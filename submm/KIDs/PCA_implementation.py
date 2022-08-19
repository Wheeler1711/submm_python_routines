#!/usr/bin/env python3

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt


def PCA_SVD(orig_array, n_comp_remove, plot=False):
    """This is a Principal Component Analysis cleaning of common modes from the 
    data. This is the implementation using Single Value Decomposition which is
    usually held as a better implementation, but may be undesireable for long
    time streams

    array is the 2D numpy array with the ith time point, and jth detector in
        array[i,j]
    n_comp_remove is the number of components to remove as the common mode
    """

    #subtract off the mean of each time stream
    mean_array = np.mean(orig_array, axis=0)
    array = orig_array - mean_array

    #normalize by variance (ensure the channels are weighted evenly rather
    #than any biasing towards the higher response channels
    var_array = np.var(array, axis=0)

    hold_res = np.copy(array)
    array = array[:, var_array !=0.] #bad calibration can create timestreams of
            #all 1 or 0, meaning variance normailization does not work
            #just ignore these channels and keep them unchanged
    array /= np.sqrt(var_array[var_array != 0])

    #convert to the transverse array since usual notation and equations have
    #array[j,i] where i is the trial/timepoint and j is the detector
    array = array.T
    #compute the svd, not using full matrices (drops zero columns which only
    #use memory and prevent easy matrix multiplication)
    U, S, Vh = np.linalg.svd(array, full_matrices=False, compute_uv=True)
    removed = Vh[0:n_comp_remove,:] #save the removed components so can be
        #inspected later, svd should sort components in decending strength
    S[0:n_comp_remove] = 0. #set the components being removed to zero
    #just remultiply the arrays together to get the results with the removed
    #components explicitely set to zero
    cleaned_array = np.dot(U * S, Vh)

    #undo the transvers and variance normalization and add back the mean in
    #case we are working in I, Q or some other case where a DC offset is
    #necessary
    cleaned_array = cleaned_array.T
    #to keep ignored channels the same, just put cleaned data into copy of 
    #original array
    hold_res[:,var_array != 0.] = cleaned_array
    cleaned_array = hold_res
    #undo normalizations and mean subtraction
    cleaned_array *= np.sqrt(var_array)
    cleaned_array += mean_array
    delta = orig_array - cleaned_array
    if plot:
        plot_PCA(U, S, Vh, var_array, mean_array)

    #return both the cleaned array and the components removed
    
    return cleaned_array, removed.T


def PCA_covariance(array, n_comp_remove):
    """This is a Principal Component Analysis cleaning of common modes from the
    data. This is using the eigenvectors of the covariance matrix to find the
    principal components.

    array is a 2D numpy array with the ith time point and jth detector in
        array[i,j]
    n_comp_remove is the number of components to remove as the common mode
    """

    #subtract off the mean_of each time stream
    mean_array = np.mean(array, axis=0)
    array -= mean_array

    #normalize each channel by its variance
    var_array = np.var(array, axis=0)
    array /= var_array

    #calculate the covariance matrix
    cov_array = np.dot(array.T, array)

    #calculate eigen vals/vectors then sort in descending order
    eigen_val, eigen_vec = np.linalg.eig(cov_array)
    
    sort_index = np.argsort(eigen_val)[::-1]
    eigen_vec = eigen_vec[:, sort_index]
    eigen_val = eigen_val[sort_index]

    #use eigen vectors to project into PCA components
    component_streams = np.dot(array, eigen_vec)


def plot_PCA(U, S, Vh, variance, mean):
    """For diagnostics and understanding what the PCA is removing, this 
    plots the timestream and psd of every principal component, along with the 
    mixing matrix (how the components map onto the original timestreams
    """
    mapping_matrix = U * S
    pdf_pages = PdfPages('./PCA_plots.pdf')

    fig = plt.figure()
    plt.suptitle('Mixing Matrix')
    mat_plot = plt.imshow(mapping_matrix)
    fig.colorbar(mat_plot)
    pdf_pages.savefig(fig)
    plt.close(fig)

    sample_rate = 488.28125
    t_vals = np.linspace(0, Vh.shape[1] / sample_rate, Vh.shape[1])
    fft_freqs = fft.fftfreq(Vh.shape[1],1./sample_rate)
    for i in range(Vh.shape[0]):
        component_psd = 2. * fft.fft(Vh[i,:]) * np.conj(fft.fft(Vh[i,:])
                    ) / (sample_rate * Vh.shape[1])
        fig = plt.figure(i + 1000,figsize = (16,6))
        plt.subplot(211)
        plt.title("Timestream of Principal component {0}".format(i))
        plt.plot(t_vals ,Vh[i,:],
                linewidth = 2)
        plt.ylabel("response (A. U.)")
        plt.xlabel("Time (s)")
        plt.subplot(212)
        plt.loglog(fft_freqs ,component_psd,
                linewidth = 2)
        WNL = np.mean(component_psd[fft_freqs > 20.])
        plt.ylim([0.05 * WNL, max(component_psd)])
        plt.ylabel("response (A. U. / Hz)")
        plt.xlabel("Frequency (Hz)")

        pdf_pages.savefig(fig)
        plt.close(fig)
    pdf_pages.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import fftpack


#code to do principal component analysis on correlated detectors.
#I find that when you use PCA every component you subtract off loses most of its signal.
# so I guess to make gains in S/N you need to have more S/N gain from the PCA then you lose from losing the pixel
# for a spectrometer you probably don't want to lose important pixels so you should pick pixels with no lines of interest


def PCA(arr,ncomp,verbose = True,sample_rate = 488.28125,filename = "",**keywords):

    pdf_pages = PdfPages('PCA_diag'+filename+'.pdf') # name of the pdf produced

    # do mean subration
    mean_vec_arr  = np.mean(arr,axis = 0)
    arr = arr - mean_vec_arr

    # normalize by the variance
    var = np.std(arr,axis = 0)**2
    arr = arr/var

    covmat = np.dot(arr.T,arr)#/data_arr.shape[0]

    #compute the eigvectors and eigenvalues
    evals,evects = np.linalg.eig(covmat)

    #sort the eigenvectors and values by largest to smallest
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evects = evects[:,idx]

    #make the eigenfunction array
    efuncarr = np.dot(arr,evects)
    # pick components to subtract
    efuncarr[:,0:ncomp] = 0
    res_index_used = idx[0:ncomp]
    #transform back
    arr_clean = np.inner(efuncarr,evects)
    
    if verbose:
        print( "the covariance matrix is")
        print( covmat)
        print("")
        print("the eigenvalues are")
        print(evals)
        print("")
        print("the eigenvectors are")
        print(evects)
        print("")
    fig = plt.figure(1,figsize = (12,6))
    plt.subplot(1,2,1)
    plt.title("Eigenvalues")
    plt.plot(np.log10(np.abs(evals)),label = "eigenvalues")
    plt.plot(np.log10(np.abs(evals[0:ncomp])),'o',label = "components removed")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue Magnitude")
    plt.legend()


    plt.subplot(1,2,2)
    plt.title("Log10(covariance matrix)")
    plt.imshow(np.log10(covmat),interpolation = 'nearest')
    plt.colorbar()

    pdf_pages.savefig(fig)
    plt.close()

    #need to unnormalize by the varaince
    arr = arr*var
    arr_clean = arr_clean*var

    #need to add the mean back in to get the right units
    arr = arr + mean_vec_arr
    arr_clean = arr_clean + mean_vec_arr
    
    #block for plotting the correlation between the two most corrrelated detectors
    #I am having some problems scaling the cross correlation though
    '''
    fig = plt.figure(2,figsize = (12,6))
    plt.title("Cross correlation of two most correlated detectors")
    # make a figure that show the correlation between the two most correlated detectors
    cov_matrix_2 = covmat
    for i in range(0,cov_matrix_2.shape[0]):
        cov_matrix_2[i,i] = 0 #zeros out diagonal obviously detecotrs are correlated with them selves
    max_corr_index1 = np.where(cov_matrix_2 == np.max(cov_matrix_2))[0][0]
    max_corr_index2 = np.where(cov_matrix_2 == np.max(cov_matrix_2))[0][1]

    #calculate the cross correlation
    npts_fft = int(2**(np.floor(np.log2(arr.shape[0]))))
    cross = fftpack.ifft(fftpack.fft(arr[:,max_corr_index1]-1,n = npts_fft)*np.conj(fftpack.fft(arr[:,max_corr_index2]-1,n = npts_fft)))
    Sxx_index1 = 2*fftpack.fft(arr[:,max_corr_index1]-1,n = npts_fft)*np.conj(fftpack.fft(arr[:,max_corr_index1]-1,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    Sxx_index2 = 2*fftpack.fft(arr[:,max_corr_index2]-1,n = npts_fft)*np.conj(fftpack.fft(arr[:,max_corr_index2]-1,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    Sxx_cross = 2*fftpack.fft(cross,n = npts_fft)*np.conj(fftpack.fft(cross,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    freqs = fftpack.fftfreq(npts_fft,1./sample_rate)
    plt.loglog(freqs[1:],Sxx_index1[1:],label = "Resonator psd index = "+str(max_corr_index1))
    plt.loglog(freqs[1:],Sxx_index2[1:],label = "Resonator psd index = "+str(max_corr_index2))
    ylim = plt.ylim()
    plt.loglog(freqs[1:],Sxx_cross[1:],label = "Cross psd")
    plt.ylim(ylim[0],ylim[1])
    plt.legend()

    pdf_pages.savefig(fig)
    plt.close()
    '''


    pdf_pages.close()


    PCA_dict = {'arr':arr,
                    'cleaned':arr_clean,
                    'ncomp':ncomp,
                    'res_index_used':res_index_used,
                    'cov_matrix':covmat}
    return PCA_dict

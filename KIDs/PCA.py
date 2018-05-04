import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import fftpack
import nitime.algorithms as tsa #pip install nitime


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
    fig = plt.figure(2,figsize = (12,6))
    plt.title("Cross correlation of two most correlated detectors")
    # make a figure that show the correlation between the two most correlated detectors
    cov_matrix_2 = covmat
    for i in range(0,cov_matrix_2.shape[0]):
        cov_matrix_2[i,i] = 0 #zeros out diagonal obviously detecotrs are correlated with them selves
    max_corr_index1 = np.where(cov_matrix_2 == np.max(cov_matrix_2))[0][0]
    max_corr_index2 = np.where(cov_matrix_2 == np.max(cov_matrix_2))[0][1]

    # this is kind of a neat package for doing cross correlation on time streams
    # see this page for explanation
    # http://nbviewer.jupyter.org/github/mattijn/pynotebook/blob/master/ipynotebooks/Python2.7/2016/2016-05-25%20cross-spectral%20analysis.ipynb
    #calculate the cross correlation
    f, pcsd_est = tsa.multi_taper_csd(np.vstack((arr[:,max_corr_index1],arr[:,max_corr_index2])), Fs=sample_rate, low_bias=True, adaptive=False, sides='onesided')
    fki = pcsd_est.diagonal().T[0]
    fkj = pcsd_est.diagonal().T[1]
    cij = pcsd_est.diagonal(+1).T.ravel()
    #calculate the coherene
    coh = np.abs(cij)**2 / (fki * fkj) 

    
    plt.loglog(f,fki,label = "Resonator psd index = "+str(max_corr_index1))
    plt.loglog(f,fkj,label = "Resonator psd index = "+str(max_corr_index2))
    ylim = plt.ylim()
    plt.loglog(f,np.abs(cij),label = "Cross psd")
    plt.ylim(ylim[0],ylim[1])
    plt.legend()

    pdf_pages.savefig(fig)
    plt.close()

    fig = plt.figure(3,figsize = (12,6))
    plt.semilogx(f,np.abs(coh))
    plt.title("Coherence")
    plt.xlabel("Frequency Hz")
    plt.ylim(0,1)

    pdf_pages.savefig(fig)
    plt.close()


    pdf_pages.close()


    PCA_dict = {'arr':arr,
                    'cleaned':arr_clean,
                    'ncomp':ncomp,
                    'res_index_used':res_index_used,
                    'cov_matrix':covmat}
    return PCA_dict

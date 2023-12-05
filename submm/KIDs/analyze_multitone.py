from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from scipy import interpolate
from scipy import signal
from scipy import fftpack
from submm.KIDs import calibrate
import pickle
from submm.KIDs import PCA_implementation as PCA
from tqdm import tqdm


def calibrate_multi(iq_sweep_data_f, iq_sweep_data_z, stream_f= None,stream_z=None,ref= None,tau = 66*10**-9,
                    skip_beginning=0, plot_period=10, decimate=1, outfile_dir="./",
                    sample_rate=1.6*10**6, plot=True,n_comp_PCA = 0,verbose = True,
                    rotation_mode = 1,**keywords):
    '''
    function for calibrating data for an array of detectors i.e. turning 
    raw timestreams of IQ data in fractional frequency units df/f

    Parameters
    ----------
    iq_sweep_data_f: numpy array shape n_pts_iq x n_res
        The frequencies of an IQ calibration sweep
    iq_sweep_data_z: numpy array  shape n_pts_iq x n_res (complex with I = real Q = imag)
        Complex IQ swep data
    stream_f: list or numpy array length n_res
        For each streaming data series at what frequency was the tone
    stream_z: numpy array shape stream_length x n_res
        IQ data streamed
    decimate: int
        factor to bin the data by
    outfile_dir: str
        where to save the file
    sample_rate: float
        The frequency at which stream data was taken in Hz
    plot: bool
        toggle making a pdf plot of the calibration process
    n_comp_PCA: int
        Allow you to clean complex data by remove n components using 
        Priciple component analysis
    rotation_mode: int 1 or 2
        = 1 rotates streaming data to be at 0,1 in IQ plane 
        This maps frequendcy noise to real part and dissipation noise
        to the imaginary part (assumes noise ball is small)
        = 2 rotates data such that the furthest off res points
        in the IQ loop will be places at +/-pi. This makes interpolation 
        from phase to frequency not have a jump as phase goes from +pi to -pi

    Returns
    -------
    calibration dictionary with following keys
    
    fine_z: complex numpy array  shape n_pts_iq x n_re
        Copy of provided iq_sweep_data_z
    stream_z: numpy array shape stream_length x n_res
        Copy of provided stream_z
    iq_sweep_data_f: numpy array shape n_pts_iq x n_res  
        Copy of provided iq_sweep_data_f
    stream_corr:numpy array shape stream_length x n_res 
        stream z moved and rotated
    fine_corr: numpy array  shape n_pts_iq x n_res
        iq_sweep_data_z moved and rotated
    stream_df_over_f: numpy array shape stream_length x n_res 
        streaming data as df/f
    stream_time: numpy array shape stream_length
        time for each stream data point in seconds
    stream_corr_cleaned: numpy array shape stream_length x n_res or None
        cleaned IQ streaming data or None if data was not cleaned        
    stream_df_over_f_cleaned: numpy array shape stream_length x n_res 
        cleaned df/f data or None if data was not cleaned
    interp_functions: list of functions with len n_res
        functions used to turn phase into frequency during the calibration
    sample_rate: float 
        frequency of the streaming data in Hz
    '''

    if stream_f is not None:
        stream_f = np.asarray(stream_f)
    
        #generate relative packet times
        stream_time = np.arange(0,stream_z.shape[0])[skip_beginning:]*1/sample_rate
    else:
        stream_time = None
        
    #bin the data if you lik
    if decimate !=1 and stream_f is not None:
        if verbose:
            print("decimating the data by factor of "+str(decimate))
        decimated_stream_z = stream_z #need deep copy?
        decimated_ref = ref
        #factors_of_10 = int(np.floor(np.log10(decimate)))
        #for k in range(0,factors_of_10): # not suppose to decimate all at once
        #    decimated_stream_z = signal.decimate(decimated_stream_z,10,axis = 0)
        #    if ref is not None:
        #        decimated_ref = signal.decimate(ref,10,axis = 0)
            
        #decimated_stream_z = signal.decimate(decimated_stream_z,
        #                                     decimate//(10**factors_of_10),axis =0)

        decimated_stream_z = signal.resample_poly(decimated_stream_z,1,decimate,padtype = 'mean')
        if ref is not None:
            #decimated_ref = signal.decimate(decimated_ref,decimate//(10**factors_of_10),axis =0)
            decimated_ref = np.round(signal.resample_poly(decimated_ref.astype('float'),1,decimate))
        decimated_stream_time = np.arange(0,decimated_stream_z.shape[0])*1/sample_rate*decimate

        stream_z = decimated_stream_z
        stream_time = decimated_stream_time
        sample_rate = sample_rate/decimate
        if ref is not None:
            ref = decimated_ref

    #initalize some arrays to hold the calibrated data
    if stream_f is not None:
        stream_corr_all = np.zeros(stream_z.shape,dtype = 'complex')
        stream_df_over_f_all = np.zeros(stream_z.shape)
    else:
        stream_corr_all = None
        stream_df_over_f_all = None
    fine_corr_all = np.zeros(iq_sweep_data_z.shape,dtype = 'complex')

    interp_functions = []
    circle_fit = np.ndarray((iq_sweep_data_z.shape[1],4))
    if n_comp_PCA>0 and stream_f is not None:
        stream_df_over_f_all_cleaned = np.zeros(stream_z.shape)

    if verbose:
        print("Calibrating...")
    for k in tqdm(range(0,iq_sweep_data_z.shape[1]),ascii = True):
        
        #remove cable delay
        fine_corr = calibrate.remove_cable_delay(iq_sweep_data_f[:,k],iq_sweep_data_z[:,k],tau)
        if stream_f is not None:
            stream_corr = calibrate.remove_cable_delay(stream_f[k],stream_z[:,k],tau)
        
        # fit a cicle to the data
        xc, yc, R, residu  = calibrate.leastsq_circle(np.real(fine_corr),np.imag(fine_corr))
        circle_fit[k,0:3] = np.array([xc, yc, R])

        #move the data to the origin
        fine_corr = fine_corr - xc -1j*yc
        if stream_f is not None:
            stream_corr = stream_corr  - xc -1j*yc

            # rotate so streaming data is at 0 pi
            phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))

        phase_fine = np.arctan2(np.imag(fine_corr),np.real(fine_corr))
        if stream_f is None:
            rotation_mode = 2
        if rotation_mode == 2: # if you have data that covers a large part of the iq loop
            med_phase = np.arctan2(np.imag(fine_corr[0]+fine_corr[-1]),
                                    np.real(fine_corr[0]+fine_corr[-1])) + np.pi
            extrap = "min_max"
        elif rotation_mode == 1:
            med_phase = np.median(phase_stream)
            extrap = "extrapolate"
        else:
            print("pick valid rotation mode 1 or 2")
            return
        circle_fit[k,-1] = med_phase

        

        fine_corr_all[:,k] = fine_corr = fine_corr*np.exp(-1j*med_phase)
        #recalc phase_fine
        phase_fine = np.arctan2(np.imag(fine_corr),np.real(fine_corr))
        if stream_f is not None:
            stream_corr_all[:,k] = stream_corr = stream_corr*np.exp(-1j*med_phase)
            phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))
        
            stream_df_over_f_all[:,k],interp_function = \
                calibrate.interp_phase_to_df_over_f(phase_fine,
                                                phase_stream,
                                                iq_sweep_data_f[:,k],
                                                extrap = extrap)
        else: #no streaming data
            ignore ,interp_function = \
                calibrate.interp_phase_to_df_over_f(phase_fine,
                                                np.ones(10),
                                                iq_sweep_data_f[:,k],
                                                extrap = extrap)
        interp_functions.append(interp_function)

    if n_comp_PCA >0 and stream_f is not None:
        interp_functions = []
        if verbose:
            print("cleaning data")
        cleaned, removed = PCA.PCA_SVD(stream_corr_all,n_comp_PCA,
                                       plot=True,sample_rate = sample_rate/decimate,
                                       outfile_dir = outfile_dir,plot_decimate = 100)
        stream_corr_all_cleaned = cleaned

        if verbose:
            print("computing cleaned df/f")
        for k in tqdm(range(0,iq_sweep_data_z.shape[1]),ascii = True):
            phase_fine = np.arctan2(np.imag(fine_corr_all[:,k]),np.real(fine_corr_all[:,k]))
            phase_stream = np.arctan2(np.imag(stream_corr_all_cleaned[:,k]),
                                      np.real(stream_corr_all_cleaned[:,k]))
            stream_df_over_f_all_cleaned[:,k],interp_function = \
                calibrate.interp_phase_to_df_over_f(phase_fine,
                                                    phase_stream,
                                                    iq_sweep_data_f[:,k],
                                                    extrap = extrap)
            interp_functions.append(interp_function)
    else:
        stream_corr_all_cleaned = None
        stream_df_over_f_all_cleaned = None
            
        
    #save everything to a dictionary
    cal_dict = {'fine_z': iq_sweep_data_z,
                'stream_z': stream_z,
                'iq_sweep_data_f':iq_sweep_data_f,
                'stream_corr':stream_corr_all,
                'fine_corr':fine_corr_all,
                'stream_df_over_f':stream_df_over_f_all,
                'stream_time':stream_time,
                'stream_corr_cleaned':stream_corr_all_cleaned,
                'stream_df_over_f_cleaned':stream_df_over_f_all_cleaned,
                'interp_functions':interp_functions,
                'sample_rate':sample_rate,
                'circle_fit':circle_fit,
                'ref':ref}

    #plot the data if desired
    if plot:
        if verbose:
            print("plotting")
        plot_calibrate(cal_dict, circle_fit,plot_period, outfile_dir)

    #save the dictionary
    if verbose:
        print("saving cal dict:")
        #print(cal_dict); 
    pickle.dump( cal_dict, open(outfile_dir+ "_cal.p", "wb" ) ) 
    return cal_dict


def plot_calibrate(cal_dict, circle_fit,plot_period, outfile_dir='./'):
    '''
    Function to plot the calibration process in general just called
    from calibrate_multitone

    Parameters                                                                                                                
    ----------
    cal_dict: calibration dictionary produced by calibrate_multitone
    circle_fit: numpy array with shape n_res x 4
        parameters that descibe a fitted circle
    plot_period: int
        plot only every nth point where n is plot period
    outfile_dir: str
        where to save the pdf

    Returns
    -------
    Nothing but produces a pdf plot called outfile_dir + cal_plots.pdf
    '''
    pdf_pages = PdfPages(outfile_dir+"cal_plots.pdf")
    #for k in range(cal_dict['fine_z'].shape[1]):
    for k in tqdm(range(0,cal_dict['fine_z'].shape[1]),ascii = True):
        fig = plt.figure(k,figsize = (10,10))

        #plot the raw data
        plt.subplot(221,aspect = 'equal')
        plt.title("Raw data")
        plt.plot(np.real(cal_dict['fine_z'][:,k]),
                np.imag(cal_dict['fine_z'][:,k]),'o')
        plt.plot(np.real(cal_dict['stream_z'][:,k][::plot_period]),
                np.imag(cal_dict['stream_z'][:,k][::plot_period]),'.')
        
        
        #unpack corr data       
        fine_corr = cal_dict['fine_corr'][:,k]
        if cal_dict['stream_corr_cleaned'] is not None:
            stream_corr = cal_dict['stream_corr_cleaned'][:,k]
        else:
            stream_corr = cal_dict['stream_corr'][:,k]
        #reverse the circle offset and rotation
        med_phase = circle_fit[k,-1]
        xc = circle_fit[k,0]
        yc = circle_fit[k,1]
        R = circle_fit[k,2]
        fine_corr0 = fine_corr * np.exp(1.j*med_phase) + xc + 1.j * yc
        stream_corr0 = stream_corr * np.exp(1.j*med_phase) + xc + 1.j * yc
        

        plt.subplot(222)
        plt.title("Cable delay removed")
        plt.plot(np.real(fine_corr0),np.imag(fine_corr0),'o')
        plt.plot(np.real(stream_corr0)[10:-10][::plot_period],
                np.imag(stream_corr0)[10:-10][::plot_period],'.')

        #center and rotate IQ circle
        plt.subplot(223)
        plt.title("Moved to 0,0 and rotated")
        plt.plot(np.real(stream_corr)[2:-1][::plot_period],
                np.imag(stream_corr)[2:-1][::plot_period],'.')
        plt.plot(np.real(fine_corr),np.imag(fine_corr),'o')
        calibrate.plot_data_circle(np.real(fine_corr)-xc,np.imag(fine_corr)-yc,
                0, 0, R)

        #redo the phase fitting
        phase_fine = np.arctan2(np.imag(fine_corr),np.real(fine_corr))
        #use_index = np.where((-np.pi/2.<phase_fine) & (phase_fine<np.pi/2))
        phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))

        phase_small = np.linspace(np.min(phase_fine),np.max(phase_fine),1000)
        freqs_stream = cal_dict['interp_functions'][k](phase_stream)

        #plot the stream phase
        plt.subplot(224)
        plt.plot(phase_fine,cal_dict['iq_sweep_data_f'][:,k],'o--')
        plt.plot(phase_small,cal_dict['interp_functions'][k](phase_small),'--')
        plt.plot(phase_stream[::plot_period],freqs_stream[::plot_period],'.')
        ylim_upper = np.max(freqs_stream)+(np.max(freqs_stream)-np.min(freqs_stream))*3
        if ylim_upper>np.max(cal_dict['iq_sweep_data_f'][:,k]):
            ylim_upper = np.max(cal_dict['iq_sweep_data_f'][:,k])
        ylim_lower = np.min(freqs_stream)-(np.max(freqs_stream)-np.min(freqs_stream))*3
        if ylim_lower<np.min(cal_dict['iq_sweep_data_f'][:,k]):
            ylim_lower = np.min(cal_dict['iq_sweep_data_f'][:,k])
        plt.ylim(ylim_lower,ylim_upper)
        plt.xlim(np.min(phase_stream)-np.pi/4,np.max(phase_stream)+np.pi/4)
        plt.xlabel("phase")
        plt.ylabel("Frequency")

        pdf_pages.savefig(fig)
        plt.close(fig)

    pdf_pages.close()


def fft_noise(z_stream,df_over_f,sample_rate):
    '''
    Funtion for ffting resonator noise
    Parameters
    ----------
    z_stream: complex numpy array shape length streaming data x n_res
        IQ streaming data assume that the resonance has been moved
        to 0,0
    df_over_f: numpy array shape length streaming data x n_res
        streaming data in df/f units
    sample_rate: float
        sample rate at which streaming data was take in Hz
    
    Returns
    -------
    fft_freqs: numpy array
        The frequencies corresponding to the fft of Sxx,S_per,and S_par
    Sxx: numpy array
        The noise power spectral density in Sxx units where x
        is df/f
    S_per: numpy array
        the noise power spectral density perpendicular to the resonacne
        iq loop for KIDs with primarily response in frequency this will
        be a measure of detector noise in whatever units your readout 
        outputs per Hz i.e. volts^2/Hz
    S_par numpy array
        The noise power spectral density parrallel to the resonance
        circle. For KIDs with primarily response in frequency this will
        likely be a measure of the readout out noise. 
    '''
    npts_fft = int(2**(np.floor(np.log2(df_over_f.size)))) 
    Sxx = 2*fftpack.fft(df_over_f,n = npts_fft)*np.conj(fftpack.fft(df_over_f,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    #perpendicular should be radial on the circle
    per_stream = np.abs(z_stream)
    # radius times angle should be distance along the circle and should work if
    # noise ball is not small
    # so circumfernce*(phase/2pi) = 2*pi*radius*(phase/(2*pi)
    # then we also don't really have to rotate the noise ball to 1,0
    par_stream = np.mean(per_stream) * np.arctan2(np.imag(z_stream), np.real(z_stream))
    
    S_per = 2*fftpack.fft(per_stream,n = npts_fft)*np.conj(
            fftpack.fft(per_stream,n = npts_fft)
            )/sample_rate*npts_fft/npts_fft**2
    S_par = 2*fftpack.fft(par_stream,n = npts_fft)*np.conj(
            fftpack.fft(par_stream,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    fft_freqs = fftpack.fftfreq(npts_fft,1./sample_rate)
    return fft_freqs,Sxx,S_per,S_par


def noise_multi(cal_dict,outfile_dir = "./",n_comp_PCA = 0,plot = True,Sxx_ylims = None,verbose = True):
    '''
    Funciton to compute the noise power spectral density of resonators
    Parameters
    ----------
    cal_dict: dictionary
         calibration dictionary produced by calibrate_multitone 
    outfile_dir: str
        desired location of plots
    n_comp_PCA: int
        how many components to remove using pricipal component analysis on 
        the df/f data. If previously cleaned in cal this will be ignored
    plot: bool
        do or do not make plots of data
    Sxx_ylims: len 2 tuple or list
        when ploting min and max of y scale for Sxx
    verbose: bool
        print progress to terminal
    
    Returns
    -------
    Dictionary with the keys

    fft_freqs: numpy array
        The frequencies corresponding to the fft of Sxx,S_per,and S_par
    Sxx: numpy array
        The noise power spectral density in Sxx units where x
        is df/f
    S_per: numpy array
        the noise power spectral density perpendicular to the resonacne
        iq loop for KIDs with primarily response in frequency this will
        be a measure of detector noise in whatever units your readout
        outputs per Hz i.e. volts^2/Hz
    S_par: numpy array
        The noise power spectral density parrallel to the resonance
        circle. For KIDs with primarily response in frequency this will
        likely be a measure of the readout out noise.
    binned_freqs: numpy array
        binned version of fft_freqs
    Sxx_binned: numpy array
        binned version of Sxx
    S_per_binned: numpy array
        binned version of S_per
    S_par_binned: numpy array
        bineed version of S_par
    amp_subtracted: numpy array
        Sxx but renormalized using S_per and S_par to subtract of noise
        in the dissipation direction nominally attributed to the readout
    Sxx_clean: numpy array
        cleaned version of Sxx
    Sxx_binned_clean: numpy array
        binned and cleaned version of Sxx
    S_per_binned_clean: numpy array
        binned and cleaned version of S_per (requires cleaning in calibrate)
    S_par_binned_clean: numpy array
        binned and cleaned version of S_par (requires cleaning in calibrate)
    amp_subtracted_clean: numpy array
        binned and cleanead version of amp_subtracted
    '''
    sample_rate = cal_dict['sample_rate']
    if cal_dict['stream_df_over_f_cleaned'] is not None:
        do_PCA = True
        cleaned = cal_dict['stream_df_over_f_cleaned']
    elif n_comp_PCA >0:
        print("Consider doing PCA in calibration step")
        do_PCA = True
        #do PCA on the data
        cleaned, removed = PCA.PCA_SVD(cal_dict['stream_df_over_f'],n_comp_PCA,
                                       plot=True,sample_rate = sample_rate,outfile_dir = outfile_dir)
    else:
        do_PCA = False

    if verbose:
        print("FFTing...")
    for k in tqdm(range(0,cal_dict['fine_corr'].shape[1]),ascii=True):
        #print(k)
        
        fft_freqs,Sxx,S_per,S_par = fft_noise(cal_dict['stream_corr'][:,k],cal_dict['stream_df_over_f'][:,k],sample_rate)
        if do_PCA:
            if cal_dict['stream_corr_cleaned'] is not None:
                fft_freqs_2,Sxx_clean,S_per_clean,S_par_clean = fft_noise(cal_dict['stream_corr_cleaned'][:,k],cleaned[:,k], sample_rate)
            else:
                fft_freqs_2,Sxx_clean,S_per_clean,S_par_clean = fft_noise(cal_dict['stream_corr'][:,k],cleaned[:,k], sample_rate)
        if k == 0:
            #intialize some arrays
            Sxx_all = np.zeros((Sxx.shape[0],cal_dict['fine_corr'].shape[1]))
            Sxx_all_clean = np.zeros((Sxx.shape[0],cal_dict['fine_corr'].shape[1]))
            S_per_all = np.zeros((S_per.shape[0],cal_dict['fine_corr'].shape[1]))
            S_par_all = np.zeros((S_par.shape[0],cal_dict['fine_corr'].shape[1]))
            S_per_all_clean = np.zeros((S_per_clean.shape[0],cal_dict['fine_corr'].shape[1]))
            S_par_all_clean = np.zeros((S_par_clean.shape[0],cal_dict['fine_corr'].shape[1]))
            
        Sxx_all[:,k] = np.abs(Sxx)
        if do_PCA:
            Sxx_all_clean[:,k] = np.abs(Sxx_clean)
            S_per_all_clean[:,k] = np.abs(S_per_clean)
            S_par_all_clean[:,k] = np.abs(S_par_clean)
        S_per_all[:,k] = np.abs(S_per)
        S_par_all[:,k] = np.abs(S_par)

        # bin it for ploting
        #plot_bins = np.logspace(-3,np.log10(250),100)
        plot_bins = np.asarray(())
        smallest = fft_freqs[1]
        largest = np.max(fft_freqs)
        current = smallest
        while current < largest:
            plot_bins = np.append(plot_bins,10**np.floor(np.log10(current))*np.linspace(1.5,10.5,10))
            current = current*10
        binnedfreq =  binned_statistic(fft_freqs, fft_freqs, bins=plot_bins)[0] #bin the frequecy against itself
        binnedpsd = binned_statistic(fft_freqs, np.abs(Sxx), bins=plot_bins)[0]
        if do_PCA:
            binnedpsd_clean = binned_statistic(fft_freqs, np.abs(Sxx_clean), bins=plot_bins)[0]
            binnedper_clean = binned_statistic(fft_freqs, np.abs(S_per_clean), bins=plot_bins)[0]
            binnedpar_clean = binned_statistic(fft_freqs, np.abs(S_par_clean), bins=plot_bins)[0]
            amp_subtracted_clean = np.abs(binnedpsd_clean)*(binnedpar_clean-binnedper_clean)/binnedpar_clean
        binnedper = binned_statistic(fft_freqs, np.abs(S_per), bins=plot_bins)[0]
        binnedpar = binned_statistic(fft_freqs, np.abs(S_par), bins=plot_bins)[0]
        amp_subtracted = np.abs(binnedpsd)*(binnedpar-binnedper)/binnedpar
        if k == 0:
            Sxx_binned_all = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))
            Sxx_binned_all_clean = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))
            S_per_binned_all = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))
            S_par_binned_all = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))
            amp_subtracted_all = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))
            S_per_binned_all_clean = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))
            S_par_binned_all_clean = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))
            amp_subtracted_all_clean = np.zeros((binnedfreq.shape[0],cal_dict['fine_corr'].shape[1]))

        Sxx_binned_all[:,k] = binnedpsd
        if do_PCA:
             Sxx_binned_all_clean[:,k] = binnedpsd_clean
             S_per_binned_all_clean[:,k] = binnedper_clean
             S_par_binned_all_clean[:,k] = binnedpar_clean
             amp_subtracted_all_clean[:,k] = amp_subtracted_clean
        S_per_binned_all[:,k] = binnedper
        S_par_binned_all[:,k] = binnedpar
        amp_subtracted_all[:,k] = amp_subtracted

    #make a psd dictionary
    psd_dict = {'fft_freqs':fft_freqs,
                    'Sxx':Sxx_all,
                    'S_per':S_per_all,
                    'S_par':S_par_all,
                    'binned_freqs':binnedfreq,
                    'Sxx_binned':Sxx_binned_all,
                    'S_per_binned':S_per_binned_all,
                    'S_par_binned':S_par_binned_all,
                    'amp_subtracted':amp_subtracted_all,
                    'Sxx_clean':Sxx_all_clean,
                    'Sxx_binned_clean':Sxx_binned_all_clean,
                    'S_per_binned_clean':S_per_binned_all_clean,
                    'S_par_binned_clean':S_par_binned_all_clean,
                    'amp_subtracted_clean':amp_subtracted_all_clean}

    #plot the stuff
    if verbose:
        print("plotting")
    if plot:
        plot_noise_multi(psd_dict, N_PCA=n_comp_PCA, outfile_dir=outfile_dir,Sxx_ylims = Sxx_ylims) 
    #save the psd dictionary
    if verbose:
        print("saving data")
    pickle.dump( psd_dict, open( outfile_dir+"psd.p", "wb" ) )

    return psd_dict


def plot_noise_multi(psd_dict, white_freq=100., N_PCA=0, outfile_dir = "./",Sxx_ylims = None):
    '''
    Funtion to plot noise in dictionary from noise multi
    
    Parameters
    ----------
    psd_dict: dictionary produced by noise_multi
    white_freq: float
        frequency at which to measure Sxx for combined plot os Sxx values
    N_PCA int:
        number of PCA components removed in cleaning for plot label
    outfile_dir: str
        where to put the pdf produced 
    Sxx_ylims: len 2 list or tuple
        ylima for Sxx plot

    Returns
    -------
    Nothing but produces a plot called outfile_dir +psd_plot.pdf
    '''
    #create the PDF file
    pdf_pages = PdfPages(outfile_dir+"psd_plots.pdf")
    #was there a PCA analysis
    do_PCA = (N_PCA > 0)
    #calculate white noise levels and plot them
    white_index = np.argmin(np.abs(psd_dict['fft_freqs']-white_freq))
    N_res = psd_dict['Sxx'].shape[1]
    Sxx_avg = np.ndarray((N_res,))
    par_per_ratio = np.ndarray((N_res,))
    for i in range(N_res):
        Sxx_avg[i] = psd_dict['Sxx'][white_index,i]
        par_per_ratio[i] = psd_dict['S_par'][white_index,i]/psd_dict['S_per'][white_index,i]
    fig = plt.figure(9000,figsize = (16,6))
    plt.subplot(122)
    plt.title("White Noise Levels")
    plt.semilogy(np.arange(N_res), Sxx_avg, 'bo')
    plt.ylabel("Sxx (1/Hz)")
    plt.xlabel("Resonator Index")
    plt.subplot(121)
    plt.title("Total to Amplifier Ratio")
    plt.plot(np.arange(N_res), par_per_ratio, 'go')
    plt.ylabel("Ratio of Parallel to Perpendicular")
    plt.xlabel("Resonator Index")
    pdf_pages.savefig(fig)
    plt.close(fig)
    #loop over the resonators
    for k in range(N_res):
        fig = plt.figure(k,figsize = (16,6))
        #plot Sxx raw and with amplifier subtracted
        plt.subplot(122)
        plt.title("Sxx")
        nan_index = np.isnan(psd_dict['binned_freqs'])
        if k == 0:
            plt.loglog(psd_dict['fft_freqs'],np.abs(psd_dict['Sxx'][:,k]))
        plt.loglog(psd_dict['binned_freqs'][~nan_index][0:-4],
                np.abs(psd_dict['Sxx_binned'][~nan_index][0:-4,k]),linewidth = 2,
                label = "Sxx raw")
        if do_PCA:
            plt.loglog(psd_dict['binned_freqs'][~nan_index][0:-4],
                    np.abs(psd_dict['Sxx_binned_clean'][~nan_index][0:-4,k]),linewidth = 2,
                    label = "PCA {0} comps".format(N_PCA))
        plt.loglog(psd_dict['binned_freqs'][~nan_index][0:-4],
                psd_dict['amp_subtracted'][~nan_index][0:-4,k],
                linewidth = 2, label="raw amp subtracted")
        plt.ylabel("Sxx (1/Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.grid(which = "both")
        if Sxx_ylims is not None:
            plt.ylim(Sxx_ylims)
        plt.legend()

        #plot noise independant quadratures
        plt.subplot(121)
        plt.title("Res indx = "+str(k))

        plt.loglog(psd_dict['binned_freqs'][~nan_index][0:-4],psd_dict['S_per_binned'][~nan_index][0:-4,k],
                label = "amp noise")
        plt.loglog(psd_dict['binned_freqs'][~nan_index][0:-4],psd_dict['S_par_binned'][~nan_index][0:-4,k],
                label = "detect noise")
        plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.grid(which = "both")

        #save and close the figure
        pdf_pages.savefig(fig)
        plt.close(fig)

    #close the pdf file so everything gets flushed to the disk
    pdf_pages.close()

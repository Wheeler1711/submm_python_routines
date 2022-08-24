from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from scipy import interpolate
from scipy import signal
from scipy import fftpack
from KIDs import calibrate
import pickle
from KIDs import PCA_implementation as PCA



def calibrate_multi(iq_sweep_data_f, iq_sweep_data_z, stream_f,stream_z,tau = 66*10**-9,
        skip_beginning=0, plot_period=10, decimate=1, outfile_dir="./",
        sample_rate=1.6*10**6, plot=True, **keywords):
    '''
    getting ride of haveing fine and gain sweeps need to have version that 
    handels a sweep with aribtrary frequency spacing
    script for calibrating data for an array of detectors
    iq_sweep_data_f shape n_pts_iq x n_res
    iq_sweep_data_z shape n_pts_iq x n_res (complex with I = real Q = imag)
    stream_f length n_res
    stream_z shape stream_length x n_res
    plot_period should probably be turnned into deimate
    '''

    stream_f = np.asarray(stream_f)
    
    #generate relative packet times
    stream_time = np.arange(0,stream_z.shape[0])[skip_beginning:]*1/sample_rate
    
    #bin the data if you like
    if decimate !=1:
        decimated_stream_z = stream_z #need deep copy?
        #decimated_stream_time = stream_time #maybe it copys when len changes
        factors_of_10 = int(np.floor(np.log10(decimate)))
        for k in range(0,factors_of_10): # not suppose to decimate all at once
            decimated_stream_z = signal.decimate(decimated_stream_z,10,axis = 0)
            #decimated_stream_time = signal.decimate(decimated_stream_time,10,axis = 0)
        decimated_stream_z = signal.decimate(decimated_stream_z,
                                                             decimate//(10**factors_of_10),axis =0)
        decimated_stream_time = np.arange(0,decimated_stream_z.shape[0])[skip_beginning:]*1/sample_rate*decimate

        stream_z = decimated_stream_z
        stream_time = decimated_stream_time

    #initalize some arrays to hold the calibrated data
    stream_corr_all = np.zeros(stream_z.shape,dtype = 'complex')
    fine_corr_all = np.zeros(iq_sweep_data_z.shape,dtype = 'complex')
    stream_df_over_f_all = np.zeros(stream_z.shape)
    circle_fit = np.ndarray((iq_sweep_data_z.shape[1],4))
    


    for k in range(0,iq_sweep_data_z.shape[1]):
        print(k)
        
        #remove cable delay
        fine_corr = calibrate.remove_cable_delay(iq_sweep_data_f[:,k],iq_sweep_data_z[:,k],tau)
        stream_corr = calibrate.remove_cable_delay(stream_f[k],stream_z[:,k],tau)
        
        # fit a cicle to the data
        xc, yc, R, residu  = calibrate.leastsq_circle(np.real(fine_corr),np.imag(fine_corr))
        circle_fit[k,0:3] = np.array([xc, yc, R])

        #move the data to the origin
        fine_corr = fine_corr - xc -1j*yc
        stream_corr = stream_corr  - xc -1j*yc

        # rotate so streaming data is at 0 pi
        phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))
        if ("rotate_fine_first" in keywords): # if you have data that covers a large part of the iq loop
            med_phase = np.arctan2(np.imag(fine_corr),np.real(fine_corr))[0]+np.pi
        else:
            med_phase = np.median(phase_stream)
        circle_fit[k,-1] = med_phase

        fine_corr_all[:,k] = fine_corr = fine_corr*np.exp(-1j*med_phase) 
        stream_corr_all[:,k] = stream_corr = stream_corr*np.exp(-1j*med_phase)


        phase_fine = np.arctan2(np.imag(fine_corr),np.real(fine_corr))
        use_index = np.where((-np.pi/2.<phase_fine) & (phase_fine<np.pi/2))
        phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))

        #interp phase to frequency
        f_interp = interpolate.interp1d(phase_fine, iq_sweep_data_f[:,k],kind = 'quadratic',bounds_error = False,fill_value = 0)

        phase_small = np.linspace(np.min(phase_fine),np.max(phase_fine),1000)
        freqs_stream = f_interp(phase_stream)
        stream_df_over_f_all[:,k] = stream_df_over_f = freqs_stream/np.mean(freqs_stream)-1.


    #save everything to a dictionary
    cal_dict = {'fine_z': iq_sweep_data_z,
                'stream_z': stream_z,
                'iq_sweep_data_f':iq_sweep_data_f,
                'stream_corr':stream_corr_all,
                'fine_corr':fine_corr_all,
                'stream_df_over_f':stream_df_over_f_all,
                'stream_time':stream_time}

    #plot the data if desired
    if plot:
        plot_calibrate(cal_dict, circle_fit,plot_period, outfile_dir)

    #save the dictionary
    pickle.dump( cal_dict, open(outfile_dir+ "cal.p", "wb" ),2 )
    return cal_dict


def plot_calibrate(cal_dict, circle_fit,plot_period, outfile_dir='./'):
    pdf_pages = PdfPages(outfile_dir+"cal_plots.pdf")
    for k in range(cal_dict['fine_z'].shape[1]):
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
        use_index = np.where((-np.pi/2.<phase_fine) & (phase_fine<np.pi/2))
        phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))

        #interp phase to frequency
        f_interp = interpolate.interp1d(phase_fine, cal_dict['iq_sweep_data_f'][:,k],
                kind = 'quadratic',bounds_error = False,fill_value = 0)

        phase_small = np.linspace(np.min(phase_fine),np.max(phase_fine),1000)
        freqs_stream = f_interp(phase_stream)

        #plot the stream phase
        plt.subplot(224)
        plt.plot(phase_fine,cal_dict['iq_sweep_data_f'][:,k],'o')
        plt.plot(phase_small,f_interp(phase_small),'--')
        plt.plot(phase_stream[::plot_period],freqs_stream[::plot_period],'.')
        plt.ylim(np.min(freqs_stream)-(np.max(freqs_stream)-np.min(freqs_stream))*3,np.max(freqs_stream)+(np.max(freqs_stream)-np.min(freqs_stream))*3)
        plt.xlim(np.min(phase_stream)-np.pi/4,np.max(phase_stream)+np.pi/4)
        plt.xlabel("phase")
        plt.ylabel("Frequency")

        pdf_pages.savefig(fig)
        plt.close(fig)

    pdf_pages.close()


def fft_noise(z_stream,df_over_f,sample_rate):
    npts_fft = int(2**(np.floor(np.log2(df_over_f.size)))) 
    Sxx = 2*fftpack.fft(df_over_f,n = npts_fft)*np.conj(fftpack.fft(df_over_f,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    #perpendicular should be radial on the circle
    per_stream = np.abs(z_stream)
    #radius times angle should be distance along the circle and should work if 
    #noise ball is not small
    par_stream = np.mean(per_stream) * np.arctan2(np.imag(z_stream), np.real(z_stream))
    S_per = 2*fftpack.fft(per_stream,n = npts_fft)*np.conj(
            fftpack.fft(per_stream,n = npts_fft)
            )/sample_rate*npts_fft/npts_fft**2
    S_par = 2*fftpack.fft(par_stream,n = npts_fft)*np.conj(
            fftpack.fft(par_stream,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    fft_freqs = fftpack.fftfreq(npts_fft,1./sample_rate)
    return fft_freqs,Sxx,S_per,S_par


def noise_multi(cal_dict, sample_rate = 1.6*10**6,outfile_dir = "./",n_comp_PCA = 0,Sxx_ylims = None):

    if n_comp_PCA >0:
        do_PCA = True
        #do PCA on the data
        cleaned, removed = PCA.PCA_SVD(cal_dict['stream_df_over_f'],n_comp_PCA,
                                       plot=True,sample_rate = sample_rate,outfile_dir = outfile_dir)
    else:
        do_PCA = False

    for k in range(0,cal_dict['fine_corr'].shape[1]):
        print(k)


        #lets fourier transfer that crap
        fft_freqs,Sxx,S_per,S_par = fft_noise(cal_dict['stream_corr'][:,k],cal_dict['stream_df_over_f'][:,k],sample_rate)
        if do_PCA:
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
    plot_noise_multi(psd_dict, N_PCA=n_comp_PCA, outfile_dir=outfile_dir,Sxx_ylims = Sxx_ylims) 
    #save the psd dictionary
    pickle.dump( psd_dict, open( outfile_dir+"psd.p", "wb" ),2 )

    return psd_dict


def plot_noise_multi(psd_dict, white_avg=100., N_PCA=0, outfile_dir = "./",Sxx_ylims = None):
    #create the PDF file
    pdf_pages = PdfPages(outfile_dir+"psd_plots.pdf")
    #was there a PCA analysis
    do_PCA = (N_PCA > 0)
    #calculate white noise levels and plot them
    freq_mask = psd_dict['fft_freqs'] > white_avg
    N_res = psd_dict['Sxx'].shape[1]
    Sxx_avg = np.ndarray((N_res,))
    par_per_ratio = np.ndarray((N_res,))
    for i in range(N_res):
        Sxx_avg[i] = np.mean(psd_dict['Sxx'][:,i][freq_mask])
        par_per_ratio[i] = np.mean(psd_dict['S_par'][:,i][freq_mask]
                ) / np.mean(psd_dict['S_per'][:,i][freq_mask])
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

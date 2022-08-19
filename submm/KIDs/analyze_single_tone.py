import numpy as np
import matplotlib.pyplot as plt
from submm.KIDs import calibrate, resonance_fitting
from scipy import interpolate
import pickle
from scipy.stats import binned_statistic



def calibrate_single_tone(fine_f,fine_z,gain_f,gain_z,stream_f,stream_z,plot_period = 1,interp = "quadratic"):

    fig = plt.figure(3,figsize = (16,10))

    plt.subplot(241,aspect = 'equal')
    plt.title("Raw data")
    plt.plot(np.real(stream_z[::plot_period]),np.real(stream_z[::plot_period]),'.')
    plt.plot(np.real(fine_z),np.imag(fine_z),'o')
    plt.plot(np.real(gain_z),np.imag(gain_z),'o')

    plt.subplot(242)
    plt.title("Raw data")
    plt.plot(np.real(stream_z[::plot_period]),np.imag(stream_z[::plot_period]),'.')
    plt.plot(np.real(fine_z),np.imag(fine_z),'o')
    
    # normalize amplistude variation in the gain scan                                        
    amp_norm_dict = resonance_fitting.amplitude_normalization_sep(gain_f,
                                                                  gain_z,
                                                                  fine_f,
                                                                  fine_z,
                                                                  stream_f,
                                                                  stream_z)

    plt.subplot(243)
    plt.title("Gain amplitude variation fit")
    plt.plot(gain_f,10*np.log10(np.abs(gain_z)**2),'o')
    plt.plot(gain_f,10*np.log10(np.abs(amp_norm_dict['normalized_gain'])**2),'o')
    plt.plot(fine_f,10*np.log10(np.abs(amp_norm_dict['normalized_fine'])**2),'o')
    plt.plot(gain_f,10*np.log10(np.abs(amp_norm_dict['poly_data'])**2))


    plt.subplot(244)
    plt.title("Data nomalized for gain amplitude variation")
    plt.plot(np.real(amp_norm_dict['normalized_fine']),np.imag(amp_norm_dict['normalized_fine']),'o')
    #plt.plot(gain_dict['freqs'][:,k]*10**6,np.log10(np.abs(amp_norm_dict['poly_data'])**2))     
    plt.plot(np.real(amp_norm_dict['normalized_stream'][::plot_period]),np.imag(amp_norm_dict['normalized_stream'][::plot_period]),'.')
    #fit the gain   
    gain_phase = np.arctan2(np.real(amp_norm_dict['normalized_gain']),np.imag(amp_norm_dict['normalized_gain']))
    tau,fit_data_phase,gain_phase_rot = calibrate.fit_cable_delay(gain_f, gain_phase)

    plt.subplot(245)
    plt.title("Gain phase fit")
    plt.plot(gain_f,gain_phase_rot,'o')
    plt.plot(gain_f,fit_data_phase)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Phase")


    #remove cable delay
    gain_corr = calibrate.remove_cable_delay(gain_f, amp_norm_dict['normalized_gain'], tau)
    fine_corr = calibrate.remove_cable_delay(fine_f, amp_norm_dict['normalized_fine'], tau)
    stream_corr = calibrate.remove_cable_delay(stream_f, amp_norm_dict['normalized_stream'], tau)


    plt.subplot(246)
    plt.title("Cable delay removed")
    plt.plot(np.real(gain_corr),np.imag(gain_corr),'o')
    plt.plot(np.real(fine_corr),np.imag(fine_corr),'o')
    plt.plot(np.real(stream_corr)[10:-10][::plot_period],np.imag(stream_corr)[10:-10][::plot_period],'.')


    # fit a cicle to the data
    xc, yc, R, residu  = calibrate.leastsq_circle(np.real(fine_corr), np.imag(fine_corr))

    #move the data to the origin 
    gain_corr = gain_corr - xc -1j*yc
    fine_corr = fine_corr - xc -1j*yc
    stream_corr = stream_corr  - xc -1j*yc

    # rotate so streaming data is at 0 pi
    phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))
    med_phase = np.median(phase_stream)

    gain_corr = gain_corr*np.exp(-1j*med_phase)
    fine_corr = fine_corr*np.exp(-1j*med_phase)
    stream_corr = stream_corr*np.exp(-1j*med_phase)


    plt.subplot(247)
    plt.title("Moved to 0,0 and rotated")
    plt.plot(np.real(stream_corr)[2:-1][::plot_period],np.imag(stream_corr)[2:-1][::plot_period],'.')
    plt.plot(np.real(gain_corr),np.imag(gain_corr),'o')
    plt.plot(np.real(fine_corr),np.imag(fine_corr),'o')
    calibrate.plot_data_circle(np.real(fine_corr) - xc, np.imag(fine_corr) - yc, 0, 0, R)


    phase_fine = np.arctan2(np.imag(fine_corr),np.real(fine_corr))
    use_index = np.where((-np.pi/2.<phase_fine) & (phase_fine<np.pi/2))
    phase_stream = np.arctan2(np.imag(stream_corr),np.real(stream_corr))

    #interp phase to frequency
    f_interp = interpolate.interp1d(phase_fine, fine_f,kind = interp,bounds_error = False,fill_value = 0)

    phase_small = np.linspace(np.min(phase_fine),np.max(phase_fine),1000)
    freqs_stream = f_interp(phase_stream)
    stream_df_over_f_all = stream_df_over_f = freqs_stream/np.mean(freqs_stream)-1.

    plt.subplot(248)
    plt.plot(phase_fine,fine_f,'o')
    plt.plot(phase_small,f_interp(phase_small),'--')
    plt.plot(phase_stream[::plot_period],freqs_stream[::plot_period],'.')
    plt.ylim(np.min(freqs_stream)-(np.max(freqs_stream)-np.min(freqs_stream))*3,np.max(freqs_stream)+(np.max(freqs_stream)-np.min(freqs_stream))*3)
    plt.xlim(np.min(phase_stream)-np.pi/4,np.max(phase_stream)+np.pi/4)
    plt.xlabel("phase")
    plt.ylabel("Frequency")

    plt.savefig("calibration.pdf")


    cal_dict = {'fine_z': fine_z,
                        'gain_z': gain_z,
                        'stream_z': stream_z,
                        'fine_freqs':fine_f,
                        'gain_freqs':gain_f,
                        'stream_corr':stream_corr,
                        'gain_corr':gain_corr,
                        'fine_corr':fine_corr,
                        'stream_df_over_f':stream_df_over_f_all}

    pickle.dump( cal_dict, open( "cal.p", "wb" ),2 )  
    return cal_dict



def noise(cal_dict, sample_rate, title=None):

    fft_freqs,Sxx,S_per,S_par = calibrate.fft_noise(cal_dict['stream_corr'], cal_dict['stream_df_over_f'], sample_rate)
    plot_bins = np.logspace(-3,np.log10(250000),1000)
    binnedfreq =  binned_statistic(fft_freqs, fft_freqs, bins=plot_bins)[0] #bin the frequecy against itself   
    binnedpsd = binned_statistic(fft_freqs, np.abs(Sxx), bins=plot_bins)[0]

    binnedper = binned_statistic(fft_freqs, np.abs(S_per), bins=plot_bins)[0]
    binnedpar = binned_statistic(fft_freqs, np.abs(S_par), bins=plot_bins)[0]
    amp_subtracted = np.abs(binnedpsd)*(binnedpar-binnedper)/binnedpar


    fig = plt.figure(4,figsize = (16,6))
    plt.subplot(122)
    if title is None:
        plt.title("Sxx")
    else:
        plt.title('Sxx ' + title)
    #plt.loglog(fft_freqs,np.abs(Sxx))                                                                      
    plt.loglog(binnedfreq,np.abs(binnedpsd),linewidth = 2,label = "Sxx raw")
    plt.loglog(binnedfreq,amp_subtracted,linewidth = 2,label = "raw amp subtracted")
    #plt.ylim(10**-18,10**-15) 
    plt.ylabel("Sxx (1/Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.legend()

    plt.subplot(121)
    #plt.loglog(fft_freqs,S_per)
    #plt.loglog(fft_freqs,S_par)
    plt.loglog(binnedfreq,binnedper,label = "amp noise")
    plt.loglog(binnedfreq,binnedpar,label = "detect noise")
    plt.legend()
    #plt.ylim(10**2,10**6)
    plt.xlabel("Frequency (Hz)")

    plt.savefig("psd.pdf")


    psd_dict = {'fft_freqs':fft_freqs,
                    'Sxx':Sxx,
                    'S_per':S_per,
                    'S_par':S_par,
                    'binned_freqs':binnedfreq,
                    'Sxx_binned':binnedpsd,
                    'S_per_binned':binnedper,
                    'S_par_binned':binnedpar,
                    'amp_subtracted':amp_subtracted}

    #save the psd dictionary
    pickle.dump( psd_dict, open("psd.p", "wb" ),2 )

    return psd_dict

import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from scipy.io.idl import readsav as readsav
from submm.KIDs import psd_fitting

# this is module to demostrate how to used the psd fitting code
#written by Jordan Wheeler on 1/6/2017

#Change Log


bins = 10000
log_bins = np.logspace(0,4,100) 

# get some data to test the fitter on (this is single tone data processed by Steve Hailey-Dunsheath's idl code)
raw = readsav("./20151106_noise_07/Set0000Pn00Fn00/rawdata.sav",verbose = False)
data = readsav("./20151106_noise_07/Set0000Pn00Fn00/psddata.sav",verbose = False)

freqs_raw = data['mixer_data_psd']['s_xx'][0]['freq_vec'][0]
psd_raw = data['mixer_data_psd']['s_xx'][0]['csd_xy'][0]

#throw out negative frequecies
greater_than_zero = np.where(freqs_raw>0)
freqs = freqs_raw[greater_than_zero]
vals = np.abs(psd_raw[greater_than_zero])

#do some binning for plotting
binned_freq =  binned_statistic(freqs, freqs, bins=bins)[0]
binned_psd = binned_statistic(freqs, vals, bins=bins)[0]

#this binning is like the binning used when log = True for for psd_fitting
binned_freq_log =  binned_statistic(freqs, freqs, bins=log_bins)[0]
binned_psd_log = binned_statistic(freqs, vals, bins=log_bins)[0]
binned_std_log = binned_statistic(freqs, vals, bins=log_bins, statistic = psd_fitting.std_of_mean)[0]

mean_index = np.where((freqs>200) & (freqs<1000))
white_guess = np.mean(vals[mean_index])

# since we are not doing a facny fit the intial guess is important or else may get stuck in a local min
x0    = np.array([1.*10**(-15.75), white_guess,1,0.00001]) #intitial guess not required will use default if not specified
use_range = [[1,10],[25,4000]] # don't want to use the data from 10 to 25Hz as there is a big ugly noise spike there
# and I don't want to use the data that is to high frequency because it seems to confuse the fitter

# fitting using unbinned data
fit = psd_fitting.fit_psd_lor(freqs, vals, x0 = x0, use_range = use_range)
print(fit[0])

plt.figure(1,figsize = (12,12))
plt.subplot(221)

plt.loglog(binned_freq,binned_psd,label = "Data",linewidth = 2)
plt.loglog(freqs, psd_fitting.noise_profile_lor(freqs, fit[0][0], fit[0][1], fit[0][2], fit[0][3]), linewidth = 2, label ="Fit")
plt.loglog(freqs, psd_fitting.noise_slope(freqs, fit[0][1], fit[0][2]), linewidth = 2, label ="1/f^" + str(fit[0][2])[0:4])
plt.loglog(freqs, psd_fitting.noise_white(freqs, fit[0][0]), linewidth = 2, label ="White" + " " + str(fit[0][0] * 10 ** 16)[0:4] + " x10^-16")

plt.title("psd fit")    
plt.legend(loc = 1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise Sxx (1/Hz)")
plt.xlim(1,10**4)
plt.ylim(10**-17,10**-13)

#fitting after log binning the data
fit = psd_fitting.fit_psd_lor(freqs, vals, x0 = x0, use_range = use_range, log = True, uniform_weight = True)
print(fit[0])

plt.subplot(222)

plt.loglog(binned_freq,binned_psd,label = "Data",linewidth = 2)
plt.errorbar(binned_freq_log,binned_psd_log,binned_std_log, fmt='o')
plt.loglog(freqs, psd_fitting.noise_profile_lor(freqs, fit[0][0], fit[0][1], fit[0][2], fit[0][3]), linewidth = 2, label ="Fit")
plt.loglog(freqs, psd_fitting.noise_slope(freqs, fit[0][1], fit[0][2]), linewidth = 2, label ="1/f^" + str(fit[0][2])[0:4])
plt.loglog(freqs, psd_fitting.noise_white(freqs, fit[0][0]), linewidth = 2, label ="White" + " " + str(fit[0][0] * 10 ** 16)[0:4] + " x10^-16")

plt.title("psd fit log binned before fitting")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise Sxx (1/Hz)")  
plt.legend(loc = 1)
plt.xlim(1,10**4)
plt.ylim(10**-17,10**-13)


#fitting without lorentzian fall off
use_range = [[1,10],[25,500]]
x0    = np.array([1.*10**(-15.75), white_guess,1]) #intitial guess not required will use default if not specified

fit = psd_fitting.fit_psd(freqs, vals, x0 = x0, use_range = use_range)
print(fit[0])

plt.subplot(223)

plt.loglog(binned_freq,binned_psd,label = "Data",linewidth = 2)
#plt.errorbar(binned_freq_log,binned_psd_log,binned_std_log, fmt='o')
plt.loglog(freqs, psd_fitting.noise_profile(freqs, fit[0][0], fit[0][1], fit[0][2]), linewidth = 2, label ="Fit")
plt.loglog(freqs, psd_fitting.noise_slope(freqs, fit[0][1], fit[0][2]), linewidth = 2, label ="1/f^" + str(fit[0][2])[0:4])
plt.loglog(freqs, psd_fitting.noise_white(freqs, fit[0][0]), linewidth = 2, label ="White" + " " + str(fit[0][0] * 10 ** 16)[0:4] + " x10^-16")

plt.title("psd fit without lorentzian")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise Sxx (1/Hz)")  
plt.legend(loc = 1)
plt.xlim(1,10**4)
plt.ylim(10**-17,10**-13)

#fitting without lorentzian fall off and log binning before fit
fit = psd_fitting.fit_psd(freqs, vals, x0 = x0, use_range = use_range, log = True)
print(fit[0])

plt.subplot(224)

plt.loglog(binned_freq,binned_psd,label = "Data",linewidth = 2)
plt.errorbar(binned_freq_log,binned_psd_log,binned_std_log, fmt='o')
plt.loglog(freqs, psd_fitting.noise_profile(freqs, fit[0][0], fit[0][1], fit[0][2]), linewidth = 2, label ="Fit")
plt.loglog(freqs, psd_fitting.noise_slope(freqs, fit[0][1], fit[0][2]), linewidth = 2, label ="1/f^" + str(fit[0][2])[0:4])
plt.loglog(freqs, psd_fitting.noise_white(freqs, fit[0][0]), linewidth = 2, label ="White" + " " + str(fit[0][0] * 10 ** 16)[0:4] + " x10^-16")

plt.title("psd fit without lorentzian")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise Sxx (1/Hz)")  
plt.legend(loc = 1)
plt.xlim(1,10**4)
plt.ylim(10**-17,10**-13)

plt.show()

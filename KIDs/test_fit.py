import numpy as np
import matplotlib.pyplot as plt
from multitone_mako import read_multitone
from KIDs import resonance_fitting


#change log
#JDW 2017-08-17 changed to include amplitude normalization option

res_num = 0


cal = read_multitone.readcal("mako20161007_223744Cal.fits")
gain = read_multitone.readcal("mako20161007_224502Cal.fits")

all_i = np.vstack((cal['I'],gain['I']))
all_q = -np.vstack((cal['Q'],gain['Q'])) # notice the negative for the Q the mako multitone seems to have the wrong sign on Q
all_freqs = np.vstack((cal['freqs'],gain['freqs']))


x = all_freqs[:,res_num]
z = all_i[:,res_num] + 1.0j*all_q[:,res_num]

#multitone glitches
x[30] = np.nan 
x[34] = np.nan
x[50] = np.nan
x[34] = np.nan
z[30] = np.nan 
z[34] = np.nan
z[50] = np.nan
z[34] = np.nan


# fitter does not like nans
z_nonan = z[~np.isnan(z)]
x_nonan = x[~np.isnan(z)]

f0_guess = x_nonan[np.argmin(np.abs(z_nonan))]

# try with or without these
x0 = [f0_guess,10000.,0.37,np.pi*-0.1,.6,np.real(z_nonan[60]),np.imag(z_nonan[60]),3*10**-7,f0_guess]
bounds = ([np.min(x_nonan),2000,.01,-4.0*np.pi,0,-5,-5,1*10**-9,np.min(x_nonan)],[np.max(x_nonan),200000,100,4.0*np.pi,5,5,5,1*10**-6,np.max(x_nonan)])

fit_dict = resonance_fitting.fit_nonlinear_iq_with_err( x_nonan, z_nonan,x0 = x0,bounds = bounds,amp_norm = True) # with defined bounds
#fit_dict = resonance_fitting.fit_nonlinear_iq_with_err( x_nonan, z_nonan) #with out defined bounds

fit = fit_dict['fit']
fit_result = fit_dict['fit_result']
x0_result = fit_dict['x0_result']


plt.figure(1)
plt.plot(np.real(x0_result),np.imag(x0_result),'+',label = "x0")
plt.plot(np.real(z_nonan),np.imag(z_nonan),'o',label = "data")
plt.plot(np.real(fit_dict['z']),np.imag(fit_dict['z']),'o',label = "data after amp_norm")
plt.plot(np.real(fit_result),np.imag(fit_result),'*',label = "fit")
plt.legend(loc = 4)

plt.figure(2)
plt.plot(x_nonan,np.sqrt(np.real(z_nonan)**2+np.imag(z_nonan)**2),'o',label = "data")
plt.plot(x_nonan,np.sqrt(np.real(fit_dict['z'])**2+np.imag(fit_dict['z'])**2),'o',label = "data after amp norm")
plt.plot(x_nonan,np.sqrt(np.real(x0_result)**2+np.imag(x0_result)**2),'+',label = "x0")
plt.plot(x_nonan,np.sqrt(np.real(fit_result)**2+np.imag(fit_result)**2),'*',label = "fit")
plt.legend()


plt.show()



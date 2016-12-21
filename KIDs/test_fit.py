import numpy as np
import matplotlib.pyplot as plt
from multitone_mako import read_multitone
from KIDs import resonance_fitting

res_num = 3


cal = read_multitone.readcal("mako20161007_224502Cal.fits")
gain = read_multitone.readcal("mako20161007_223744Cal.fits")

all_i = np.vstack((cal['I'],gain['I']*1.05))
all_q = -np.vstack((cal['Q'],gain['Q']*1.05)) # notice the negative for the Q the mako multitone seems to have the wrong sign on Q
all_freqs = np.vstack((cal['freqs'],gain['freqs']))


x = all_freqs[:,res_num]
z = all_i[:,res_num] + 1.0j*all_q[:,res_num]

#multitone glitches
x[95] = np.nan 
z[95] = np.nan
x[91] = np.nan
z[91] = np.nan
x[92] = np.nan
z[92] = np.nan
x[111] = np.nan
z[111] = np.nan


# fitter does not like nans
z_nonan = z[~np.isnan(z)]
x_nonan = x[~np.isnan(z)]

# try with or without these
x0 = [x_nonan[31],10000.,0.37,np.pi*-0.1,.6,np.real(z_nonan[60]),np.imag(z_nonan[60]),3*10**-7,x_nonan[31]]
bounds = ([np.min(x_nonan),2000,.01,-4.0*np.pi,0,-5,-5,1*10**-9,np.min(x_nonan)],[np.max(x_nonan),200000,100,4.0*np.pi,5,5,5,1*10**-6,np.max(x_nonan)])


#fit_dict = resonance_fitting.fit_nonlinear_iq( x_nonan, z,x0 = x0,bounds = bounds) # with defined bounds
fit_dict = resonance_fitting.fit_nonlinear_iq_with_err( x_nonan, z_nonan) #with out defined bounds

fit = fit_dict['fit']
fit_result = fit_dict['fit_result']
x0_result = fit_dict['x0_result']


plt.figure(1)
plt.plot(np.real(x0_result),np.imag(x0_result),'+',label = "x0")
plt.plot(np.real(z_nonan),np.imag(z_nonan),'o',label = "data")
plt.plot(np.real(fit_result),np.imag(fit_result),'*',label = "fit")
plt.legend()

plt.figure(2)
plt.plot(x_nonan,np.sqrt(np.real(fit_result)**2+np.imag(fit_result)**2),'*',label = "fit")
plt.plot(x_nonan,np.sqrt(np.real(z_nonan)**2+np.imag(z_nonan)**2),'o',label = "data")
plt.plot(x_nonan,np.sqrt(np.real(x0_result)**2+np.imag(x0_result)**2),'+',label = "x0")
plt.legend()


plt.show()



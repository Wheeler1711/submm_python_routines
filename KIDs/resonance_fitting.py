import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt
from KIDs import calibrate


# module for fitting resonances curves for kinetic inductance detectors.
# written by Jordan Wheeler 12/21/16

# for example see test_fit.py in this directory

# To Do
# I think the error analysis on the fit_nonlinear_iq_with_err probably needs some work
# add in step by step fitting i.e. first amplitude normalizaiton, then cabel delay, then i0,q0 subtraction, then phase rotation, then the rest of the fit. 

#Change log
#JDW 2017-08-17 added in a keyword/function to allow for gain varation "amp_var" to be taken out before fitting
#JDW 2017-08-30 added in fitting for magnitude fitting of resonators i.e. not in iq space
#JDW 2018-03-05 added more clever function for guessing x0 for fits
#JDW 2018-08-23 added more clever guessing for resonators with large phi into guess seperate functions

# function to descript the magnitude S21 of a non linear resonator
def nonlinear_mag(x,fr,Qr,amp,phi,a,b0,b1,flin):
    # x is the frequeciesn your iq sweep covers
    # fr is the center frequency of the resonator
    # Qr is the quality factor of the resonator
    # amp is Qr/Qc
    # phi is a rotation paramter for an impedance mismatch between the resonaotor and the readout system
    # a is the non-linearity paramter bifurcation occurs at a = 0.77
    # b0 DC level of s21 away from resonator
    # b1 Frequency dependant gain varation
    # flin is probably the frequency of the resonator when a = 0
    #
    # This is based of fitting code from MUSIC
    # The idea is we are producing a model that is described by the equation below
    # the frist two terms in the large parentasis and all other terms are farmilar to me
    # but I am not sure where the last term comes from though it does seem to be important for fitting
    #
    #                          /        (j phi)            (j phi)   \  2
    #|S21|^2 = (b0+b1 x_lin)* |1 -amp*e^           +amp*(e^       -1) |^
    #                         |   ------------      ----              |
    #                          \     (1+ 2jy)         2              /
    #
    # where the nonlineaity of y is described by the following eqution taken from Response of superconducting microresonators
    # with nonlinear kinetic inductance
    #                                     yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr
    #    
    
    xlin = (x - flin)/flin
    xg = (x-fr)/fr
    yg = Qr*xg
    y = np.zeros(x.shape[0])
    #find the roots of the y equation above
    for i in range(0,x.shape[0]):
        # 4y^3+ -4yg*y^2+ y -(yg+a)
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        #roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))   #more accurate version that doesn't seem to change the fit at al     
        # only care about real roots
        where_real = np.where(np.imag(roots) == 0)
        y[i] = np.max(np.real(roots[where_real]))
    z = (b0 +b1*xlin)*np.abs(1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*y) + amp/2.*(np.exp(1.0j*phi) -1.0))**2
    return z



# function to describe the i q loop of a nonlinear resonator
def nonlinear_iq(x,fr,Qr,amp,phi,a,i0,q0,tau,f0):
    # x is the frequeciesn your iq sweep covers
    # fr is the center frequency of the resonator
    # Qr is the quality factor of the resonator
    # amp is Qr/Qc
    # phi is a rotation paramter for an impedance mismatch between the resonaotor and the readou system
    # a is the non-linearity paramter bifurcation occurs at a = 0.77
    # i0
    # q0 these are constants that describes an overall phase rotation of the iq loop + a DC gain offset
    # tau cabel delay
    # f0 is all the center frequency, not sure why we include this as a secondary paramter should be the same as fr
    #
    # This is based of fitting code from MUSIC
    #
    # The idea is we are producing a model that is described by the equation below
    # the frist two terms in the large parentasis and all other terms are farmilar to me
    # but I am not sure where the last term comes from though it does seem to be important for fitting
    #
    #                    (-j 2 pi deltaf tau)  /        (j phi)            (j phi)   \
    #        (i0+j*q0)*e^                    *|1 -amp*e^           +amp*(e^       -1) |
    #                                         |   ------------      ----              |
    #                                          \     (1+ 2jy)         2              /
    #
    # where the nonlineaity of y is described by the following eqution taken from Response of superconducting microresonators
    # with nonlinear kinetic inductance
    #                                     yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr
    #    
    deltaf = (x - f0)
    xg = (x-fr)/fr
    yg = Qr*xg
    y = np.zeros(x.shape[0])
    #find the roots of the y equation above
    for i in range(0,x.shape[0]):
        # 4y^3+ -4yg*y^2+ y -(yg+a)
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        #roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))   #more accurate version that doesn't seem to change the fit at al     
        # only care about real roots
        where_real = np.where(np.imag(roots) == 0)
        y[i] = np.max(np.real(roots[where_real]))
    z = (i0 +1.j*q0)* np.exp(-1.0j* 2* np.pi *deltaf*tau) * (1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*y) + amp/2.*(np.exp(1.0j*phi) -1.0))
    return z



# when using a fitter that can't handel complex number one needs to return both the real and imaginary components seperatly
def nonlinear_iq_for_fitter(x,fr,Qr,amp,phi,a,i0,q0,tau,f0):    
    deltaf = (x - f0)
    xg = (x-fr)/fr
    yg = Qr*xg
    y = np.zeros(x.shape[0])
    
    for i in range(0,x.shape[0]):
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        where_real = np.where(np.imag(roots) == 0)
        y[i] = np.max(np.real(roots[where_real]))
    z = (i0 +1.j*q0)* np.exp(-1.0j* 2* np.pi *deltaf*tau) * (1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*y) + amp/2.*(np.exp(1.0j*phi) -1.0))
    real_z = np.real(z)
    imag_z = np.imag(z)
    return np.hstack((real_z,imag_z))



# function for fitting an iq sweep with the above equation
def fit_nonlinear_iq(x,z,**keywords):
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat  
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(x),50,.01,-np.pi,0,-np.inf,-np.inf,1*10**-9,np.min(x)],[np.max(x),200000,100,np.pi,5,np.inf,np.inf,np.max(x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        #fr_guess = x[np.argmin(np.abs(z))]
        #x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
        x0 = guess_x0_iq_nonlinear(x,z,verbose = True)
        print(x0)
    #Amplitude normalization?
    do_amp_norm = 0
    if ('amp_norm' in keywords):
        amp_norm = keywords['amp_norm']
        if amp_norm == True:
            do_amp_norm = 1
        elif amp_norm == False:
            do_amp_norm = 0
        else:
            print("please specify amp_norm as True or False")
    if do_amp_norm == 1:
        z = amplitude_normalization(x,z)          
    z_stacked = np.hstack((np.real(z),np.imag(z)))    
    fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,bounds = bounds)
    fit_result = nonlinear_iq(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    x0_result = nonlinear_iq(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8])

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z}
    return fit_dict

def fit_nonlinear_iq_sep(fine_x,fine_z,gain_x,gain_z,**keywords):
    # same as above funciton but takes fine and gain scans seperatly
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat  
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(fine_x),500.,.01,-np.pi,0,-np.inf,-np.inf,1*10**-9,np.min(fine_x)],[np.max(fine_x),1000000,100,np.pi,5,np.inf,np.inf,1*10**-6,np.max(fine_x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        #fr_guess = x[np.argmin(np.abs(z))]
        #x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
        x0 = guess_x0_iq_nonlinear_sep(fine_x,fine_z,gain_x,gain_z)
        #print(x0)
    #Amplitude normalization?
    do_amp_norm = 0
    if ('amp_norm' in keywords):
        amp_norm = keywords['amp_norm']
        if amp_norm == True:
            do_amp_norm = 1
        elif amp_norm == False:
            do_amp_norm = 0
        else:
            print("please specify amp_norm as True or False")

    x = np.hstack((fine_x,gain_x))
    z = np.hstack((fine_z,gain_z))

    if do_amp_norm == 1:
        z = amplitude_normalization(x,z)   
       
    z_stacked = np.hstack((np.real(z),np.imag(z)))    
    fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,bounds = bounds)
    fit_result = nonlinear_iq(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    x0_result = nonlinear_iq(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8])

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,'fit_freqs':x}
    return fit_dict



# same function but double fits so that it can get error and a proper covariance matrix out
def fit_nonlinear_iq_with_err(x,z,**keywords):
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat 
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(x),2000,.01,-np.pi,0,-5,-5,1*10**-9,np.min(x)],[np.max(x),200000,1,np.pi,5,5,5,1*10**-6,np.max(x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        fr_guess = x[np.argmin(np.abs(z))]
        x0 = guess_x0_iq_nonlinear(x,z)
    #Amplitude normalization?
    do_amp_norm = 0
    if ('amp_norm' in keywords):
        amp_norm = keywords['amp_norm']
        if amp_norm == True:
            do_amp_norm = 1
        elif amp_norm == False:
            do_amp_norm = 0
        else:
            print("please specify amp_norm as True or False")
    if do_amp_norm == 1:
        z = amplitude_normalization(x,z)  
    z_stacked = np.hstack((np.real(z),np.imag(z)))    
    fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,bounds = bounds)
    fit_result = nonlinear_iq(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    fit_result_stacked = nonlinear_iq_for_fitter(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    x0_result = nonlinear_iq(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8])
    # get error
    var = np.sum((z_stacked-fit_result_stacked)**2)/(z_stacked.shape[0] - 1)
    err = np.ones(z_stacked.shape[0])*np.sqrt(var)
    # refit
    fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,err,bounds = bounds)
    fit_result = nonlinear_iq(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    x0_result = nonlinear_iq(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8])
    

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z}
    return fit_dict


# function for fitting an iq sweep with the above equation
def fit_nonlinear_mag(x,z,**keywords):
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat  
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(x),100,.01,-np.pi,0,-np.inf,-np.inf,np.min(x)],[np.max(x),200000,100,np.pi,5,np.inf,np.inf,np.max(x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        fr_guess = x[np.argmin(np.abs(z))]
        x0 = [fr_guess,10000.,0.5,0,0,np.abs(z[0])**2,np.abs(z[0])**2,fr_guess]

    fit = optimization.curve_fit(nonlinear_mag, x, np.abs(z)**2 ,x0,bounds = bounds)
    fit_result = nonlinear_mag(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7])
    x0_result = nonlinear_mag(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7])

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z}
    return fit_dict

def fit_nonlinear_mag_sep(fine_x,fine_z,gain_x,gain_z,**keywords):
    # same as above but fine and gain scans are provided seperatly
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat  
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(fine_x),100,.01,-np.pi,0,-np.inf,-np.inf,np.min(fine_x)],[np.max(fine_x),1000000,100,np.pi,5,np.inf,np.inf,np.max(fine_x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        x0 = guess_x0_mag_nonlinear_sep(fine_x,fine_z,gain_x,gain_z)

    #stack the scans for curvefit
    x = np.hstack((fine_x,gain_x))
    z = np.hstack((fine_z,gain_z))
    fit = optimization.curve_fit(nonlinear_mag, x, np.abs(z)**2 ,x0,bounds = bounds)
    fit_result = nonlinear_mag(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7])
    x0_result = nonlinear_mag(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7])

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,'fit_freqs':x}
    return fit_dict

def amplitude_normalization(x,z):
    # normalize the amplitude varation requires a gain scan
    #flag frequencies to use in amplitude normaliztion
    index_use = np.where(np.abs(x-np.median(x))>100000) #100kHz away from resonator
    poly = np.polyfit(x[index_use],np.abs(z[index_use]),2)
    poly_func = np.poly1d(poly)
    normalized_data = z/poly_func(x)*np.median(np.abs(z[index_use]))
    return normalized_data

def amplitude_normalization_sep(gain_x,gain_z,fine_x,fine_z,stream_x,stream_z):
    # normalize the amplitude varation requires a gain scan
    # uses gain scan to normalize does not use fine scan
    #flag frequencies to use in amplitude normaliztion
    index_use = np.where(np.abs(gain_x-np.median(gain_x))>100000) #100kHz away from resonator
    poly = np.polyfit(gain_x[index_use],np.abs(gain_z[index_use]),2)
    poly_func = np.poly1d(poly)
    poly_data = poly_func(gain_x)
    normalized_gain = gain_z/poly_data*np.median(np.abs(gain_z[index_use]))
    normalized_fine = fine_z/poly_func(fine_x)*np.median(np.abs(gain_z[index_use]))
    normalized_stream = stream_z/poly_func(stream_x)*np.median(np.abs(gain_z[index_use]))
    amp_norm_dict = {'normalized_gain':normalized_gain,
                         'normalized_fine':normalized_fine,
                         'normalized_stream':normalized_stream,
                         'poly_data':poly_data}
    return amp_norm_dict

def guess_x0_iq_nonlinear(x,z,verbose = False): 
    # this is lest robust than guess_x0_iq_nonlinear_sep 
    # below. it is recommended to use that instead   
    #make sure data is sorted from low to high frequency
    sort_index = np.argsort(x)
    x = x[sort_index]
    z = z[sort_index]
    #extract just fine data
    df = np.abs(x-np.roll(x,1))
    fine_df = np.min(df[np.where(df != 0)]) 
    fine_z_index = np.where(df<fine_df*1.1)
    fine_z = z[fine_z_index]
    fine_x = x[fine_z_index]
    #extract the gain scan
    gain_z_index = np.where(df>fine_df*1.1)
    gain_z = z[gain_z_index]
    gain_x = x[gain_z_index]
    gain_phase = np.arctan2(np.real(gain_z),np.imag(gain_z))
    
    #guess f0
    fr_guess_index = np.argmin(np.abs(z))
    #fr_guess = x[fr_guess_index]
    fr_guess_index_fine = np.argmin(np.abs(fine_z))
    # below breaks if there is not a right and left side in the fine scan
    if fr_guess_index_fine == 0:
        fr_guess_index_fine = len(fine_x)//2
    elif fr_guess_index_fine == (len(fine_x)-1):
        fr_guess_index_fine = len(fine_x)//2
    fr_guess = fine_x[fr_guess_index_fine]
    
    #guess Q
    mag_max = np.max(np.abs(fine_z)**2)
    mag_min = np.min(np.abs(fine_z)**2)
    mag_3dB = (mag_max+mag_min)/2.
    half_distance = np.abs(fine_z)**2-mag_3dB
    right = half_distance[fr_guess_index_fine:-1]
    left  = half_distance[0:fr_guess_index_fine]
    right_index = np.argmin(np.abs(right))+fr_guess_index_fine
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index]-fine_x[left_index]
    Q_guess = fr_guess/Q_guess_Hz
    
    #guess amp
    d = np.max(20*np.log10(np.abs(z)))-np.min(20*np.log10(np.abs(z)))
    amp_guess = 0.0037848547850284574+0.11096782437821565*d-0.0055208783469291173*d**2+0.00013900471000261687*d**3+-1.3994861426891861e-06*d**4#polynomial fit to amp verus depth
    
    #guess impedance rotation phi
    phi_guess = 0
    
    #guess non-linearity parameter
    #might be able to guess this by ratioing the distance between min and max distance between iq points in fine sweep
    a_guess = 0
    
    #i0 and iq guess
    if np.max(np.abs(fine_z))==np.max(np.abs(z)): #if the resonator has an impedance mismatch rotation that makes the fine greater that the cabel delay
        i0_guess = np.real(fine_z[np.argmax(np.abs(fine_z))])
        q0_guess = np.imag(fine_z[np.argmax(np.abs(fine_z))])
    else:
        i0_guess = (np.real(fine_z[0])+np.real(fine_z[-1]))/2.
        q0_guess = (np.imag(fine_z[0])+np.imag(fine_z[-1]))/2.
        
    #cabel delay guess tau
    #y = mx +b
    #m = (y2 - y1)/(x2-x1)
    #b = y-mx
    m = (gain_phase - np.roll(gain_phase,1))/(gain_x-np.roll(gain_x,1))
    b = gain_phase -m*gain_x
    m_best = np.median(m[~np.isnan(m)])
    tau_guess = m_best/(2*np.pi)
        
    if verbose == True:
        print("fr guess  = %.2f MHz" %(fr_guess/10**6))
        print("Q guess   = %.2f kHz, %.1f" % ((Q_guess_Hz/10**3),Q_guess))
        print("amp guess = %.2f" %amp_guess)
        print("i0 guess  = %.2f" %i0_guess)
        print("q0 guess  = %.2f" %q0_guess)
        print("tau guess = %.2f x 10^-7" %(tau_guess/10**-7))
    
    x0 = [fr_guess,Q_guess,amp_guess,phi_guess,a_guess,i0_guess,q0_guess,tau_guess,fr_guess]
    return x0

def guess_x0_mag_nonlinear(x,z,verbose = False): 
    # this is lest robust than guess_x0_mag_nonlinear_sep 
    #below it is recommended to use that instead   
    #make sure data is sorted from low to high frequency
    sort_index = np.argsort(x)
    x = x[sort_index]
    z = z[sort_index]
    #extract just fine data
    #this will probably break if there is no fine scan
    df = np.abs(x-np.roll(x,1))
    fine_df = np.min(df[np.where(df != 0)]) 
    fine_z_index = np.where(df<fine_df*1.1)
    fine_z = z[fine_z_index]
    fine_x = x[fine_z_index]
    #extract the gain scan
    gain_z_index = np.where(df>fine_df*1.1)
    gain_z = z[gain_z_index]
    gain_x = x[gain_z_index]
    gain_phase = np.arctan2(np.real(gain_z),np.imag(gain_z))
    
    #guess f0
    fr_guess_index = np.argmin(np.abs(z))
    #fr_guess = x[fr_guess_index]
    fr_guess_index_fine = np.argmin(np.abs(fine_z))
    if fr_guess_index_fine == 0:
        fr_guess_index_fine = len(fine_x)//2
    elif fr_guess_index_fine == (len(fine_x)-1):
        fr_guess_index_fine = len(fine_x)//2
    fr_guess = fine_x[fr_guess_index_fine]
    
    #guess Q
    mag_max = np.max(np.abs(fine_z)**2)
    mag_min = np.min(np.abs(fine_z)**2)
    mag_3dB = (mag_max+mag_min)/2.
    half_distance = np.abs(fine_z)**2-mag_3dB
    right = half_distance[fr_guess_index_fine:-1]
    left  = half_distance[0:fr_guess_index_fine]
    right_index = np.argmin(np.abs(right))+fr_guess_index_fine
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index]-fine_x[left_index]
    Q_guess = fr_guess/Q_guess_Hz
    
    #guess amp
    d = np.max(20*np.log10(np.abs(z)))-np.min(20*np.log10(np.abs(z)))
    amp_guess = 0.0037848547850284574+0.11096782437821565*d-0.0055208783469291173*d**2+0.00013900471000261687*d**3+-1.3994861426891861e-06*d**4#polynomial fit to amp verus depth
    
    #guess impedance rotation phi
    phi_guess = 0
    
    #guess non-linearity parameter
    #might be able to guess this by ratioing the distance between min and max distance between iq points in fine sweep
    a_guess = 0
    
    #b0 and b1 guess
    xlin = (gain_x - fr_guess)/fr_guess
    b1_guess = (np.abs(gain_z)[-1]**2-np.abs(gain_z)[0]**2)/(xlin[-1]-xlin[0])
    b0_guess = np.median(np.abs(gain_z)**2)
        
    #cabel delay guess tau
    #y = mx +b
    #m = (y2 - y1)/(x2-x1)
    #b = y-mx
    m = (gain_phase - np.roll(gain_phase,1))/(gain_x-np.roll(gain_x,1))
    b = gain_phase -m*gain_x
    m_best = np.median(m[~np.isnan(m)])
    tau_guess = m_best/(2*np.pi)
       
    if verbose == True:
        print("fr guess  = %.2f MHz" %(fr_guess/10**6))
        print("Q guess   = %.2f kHz, %.1f" % ((Q_guess_Hz/10**3),Q_guess))
        print("amp guess = %.2f" %amp_guess)
        print("phi guess = %.2f" %phi_guess)
        print("b0 guess  = %.2f" %b0_guess)
        print("b1 guess  = %.2f" %b1_guess)
        print("tau guess = %.2f x 10^-7" %(tau_guess/10**-7))
    
    x0 = [fr_guess,Q_guess,amp_guess,phi_guess,a_guess,b0_guess,b1_guess,fr_guess]
    return x0


def guess_x0_iq_nonlinear_sep(fine_x,fine_z,gain_x,gain_z,verbose = False):   
    # this is the same as guess_x0_iq_nonlinear except that it takes
    # takes the fine scan and the gain scan as seperate variables
    # this runs into less issues when trying to sort out what part of 
    # data is fine and what part is gain for the guessing 
    #make sure data is sorted from low to high frequency

    #gain phase
    gain_phase = np.arctan2(np.real(gain_z),np.imag(gain_z))
    
    #guess f0
    fr_guess_index = np.argmin(np.abs(fine_z))
    # below breaks if there is not a right and left side in the fine scan
    if fr_guess_index == 0:
        fr_guess_index = len(fine_x)//2
    elif fr_guess_index == (len(fine_x)-1):
        fr_guess_index = len(fine_x)//2
    fr_guess = fine_x[fr_guess_index]
    
    #guess Q
    mag_max = np.max(np.abs(fine_z)**2)
    mag_min = np.min(np.abs(fine_z)**2)
    mag_3dB = (mag_max+mag_min)/2.
    half_distance = np.abs(fine_z)**2-mag_3dB
    right = half_distance[fr_guess_index:-1]
    left  = half_distance[0:fr_guess_index]
    right_index = np.argmin(np.abs(right))+fr_guess_index
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index]-fine_x[left_index]
    Q_guess = fr_guess/Q_guess_Hz
    
    #guess amp
    d = np.max(20*np.log10(np.abs(gain_z)))-np.min(20*np.log10(np.abs(fine_z)))
    amp_guess = 0.0037848547850284574+0.11096782437821565*d-0.0055208783469291173*d**2+0.00013900471000261687*d**3+-1.3994861426891861e-06*d**4#polynomial fit to amp verus depth
    
    #guess impedance rotation phi
    #phi_guess = 0
    #guess impedance rotation phi
    #fit a circle to the iq loop
    xc, yc, R, residu  = calibrate.leastsq_circle(np.real(fine_z),np.imag(fine_z))
    #compute angle between (off_res,off_res),(0,0) and (off_ress,off_res),(xc,yc) of the the fitted circle
    off_res_i,off_res_q = (np.real(fine_z[0])+np.real(fine_z[-1]))/2.,(np.imag(fine_z[0])+np.imag(fine_z[-1]))/2.
    x1, y1, = -off_res_i,-off_res_q
    x2, y2 = xc-off_res_i,yc-off_res_q
    dot = x1*x2 + y1*y2      # dot product
    det = x1*y2 - y1*x2      # determinant
    angle = np.arctan2(det, dot)
    phi_guess = angle

    # if phi is large better re guess f0
    # f0 should be the farthers from the off res point
    if (np.abs(phi_guess)>0.3):
        dist1 = np.sqrt((np.real(fine_z[0])-np.real(fine_z))**2+(np.imag(fine_z[0])-np.imag(fine_z))**2)
        dist2 = np.sqrt((np.real(fine_z[-1])-np.real(fine_z))**2+(np.imag(fine_z[-1])-np.imag(fine_z))**2)
        fr_guess_index = np.argmax((dist1+dist2))
        fr_guess = fine_x[fr_guess_index]
        #also fix the Q gues
        fine_z_derot = (fine_z-(off_res_i+1.j*off_res_q))*np.exp(1j*(-phi_guess))+(off_res_i+1.j*off_res_q)
        #fr_guess_index = np.argmin(np.abs(fine_z_derot))
        #fr_guess = fine_x[fr_guess_index]
        mag_max = np.max(np.abs(fine_z_derot)**2)
        mag_min = np.min(np.abs(fine_z_derot)**2)
        mag_3dB = (mag_max+mag_min)/2.
        half_distance = np.abs(fine_z_derot)**2-mag_3dB
        right = half_distance[np.argmin(np.abs(fine_z_derot)):-1]
        left  = half_distance[0:np.argmin(np.abs(fine_z_derot))]
        right_index = np.argmin(np.abs(right))+np.argmin(np.abs(fine_z_derot))
        left_index = np.argmin(np.abs(left))
        Q_guess_Hz = fine_x[right_index]-fine_x[left_index]
        Q_guess = fr_guess/Q_guess_Hz
        #also fix amp guess
        d = np.max(20*np.log10(np.abs(gain_z)))-np.min(20*np.log10(np.abs(fine_z_derot)))
        amp_guess = 0.0037848547850284574+0.11096782437821565*d-0.0055208783469291173*d**2+0.00013900471000261687*d**3+-1.3994861426891861e-06*d**4
    
    #guess non-linearity parameter
    #might be able to guess this by ratioing the distance between min and max distance between iq points in fine sweep
    a_guess = 0
    
    #i0 and iq guess
    if np.max(np.abs(fine_z))>np.max(np.abs(gain_z)): #if the resonator has an impedance mismatch rotation that makes the fine greater that the cabel delay
        i0_guess = np.real(fine_z[np.argmax(np.abs(fine_z))])
        q0_guess = np.imag(fine_z[np.argmax(np.abs(fine_z))])
    else:
        i0_guess = (np.real(fine_z[0])+np.real(fine_z[-1]))/2.
        q0_guess = (np.imag(fine_z[0])+np.imag(fine_z[-1]))/2.
        
    #cabel delay guess tau
    #y = mx +b
    #m = (y2 - y1)/(x2-x1)
    #b = y-mx
    m = (gain_phase - np.roll(gain_phase,1))/(gain_x-np.roll(gain_x,1))
    b = gain_phase -m*gain_x
    m_best = np.median(m[~np.isnan(m)])
    tau_guess = m_best/(2*np.pi)
        
    if verbose == True:
        print("fr guess  = %.3f MHz" %(fr_guess/10**6))
        print("Q guess   = %.2f kHz, %.1f" % ((Q_guess_Hz/10**3),Q_guess))
        print("amp guess = %.2f" %amp_guess)
        print("phi guess = %.2f" %phi_guess)
        print("i0 guess  = %.2f" %i0_guess)
        print("q0 guess  = %.2f" %q0_guess)
        print("tau guess = %.2f x 10^-7" %(tau_guess/10**-7))
    
    x0 = [fr_guess,Q_guess,amp_guess,phi_guess,a_guess,i0_guess,q0_guess,tau_guess,fr_guess]
    return x0

def guess_x0_mag_nonlinear_sep(fine_x,fine_z,gain_x,gain_z,verbose = False):   
    # this is the same as guess_x0_mag_nonlinear except that it takes
    # takes the fine scan and the gain scan as seperate variables
    # this runs into less issues when trying to sort out what part of 
    # data is fine and what part is gain for the guessing 
    #make sure data is sorted from low to high frequency 

    #phase of gain
    gain_phase = np.arctan2(np.real(gain_z),np.imag(gain_z))
    
    #guess f0
    fr_guess_index = np.argmin(np.abs(fine_z))
    #protect against guessing the first or last data points
    if fr_guess_index == 0:
        fr_guess_index = len(fine_x)//2
    elif fr_guess_index == (len(fine_x)-1):
        fr_guess_index = len(fine_x)//2
    fr_guess = fine_x[fr_guess_index]
    
    #guess Q
    mag_max = np.max(np.abs(fine_z)**2)
    mag_min = np.min(np.abs(fine_z)**2)
    mag_3dB = (mag_max+mag_min)/2.
    half_distance = np.abs(fine_z)**2-mag_3dB
    right = half_distance[fr_guess_index:-1]
    left  = half_distance[0:fr_guess_index]
    right_index = np.argmin(np.abs(right))+fr_guess_index
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index]-fine_x[left_index]
    Q_guess = fr_guess/Q_guess_Hz
    
    #guess amp
    d = np.max(20*np.log10(np.abs(gain_z)))-np.min(20*np.log10(np.abs(fine_z)))
    amp_guess = 0.0037848547850284574+0.11096782437821565*d-0.0055208783469291173*d**2+0.00013900471000261687*d**3+-1.3994861426891861e-06*d**4
    #polynomial fit to amp verus depth calculated emperically

    
    
    #guess impedance rotation phi
    #fit a circle to the iq loop
    xc, yc, R, residu  = calibrate.leastsq_circle(np.real(fine_z),np.imag(fine_z))
    #compute angle between (off_res,off_res),(0,0) and (off_ress,off_res),(xc,yc) of the the fitted circle
    off_res_i,off_res_q = (np.real(fine_z[0])+np.real(fine_z[-1]))/2.,(np.imag(fine_z[0])+np.imag(fine_z[-1]))/2.
    x1, y1, = -off_res_i,-off_res_q
    x2, y2 = xc-off_res_i,yc-off_res_q
    dot = x1*x2 + y1*y2      # dot product
    det = x1*y2 - y1*x2      # determinant
    angle = np.arctan2(det, dot)
    phi_guess = angle

    # if phi is large better re guess f0
    # f0 should be the farthers from the off res point
    if (np.abs(phi_guess)>0.3):
        dist1 = np.sqrt((np.real(fine_z[0])-np.real(fine_z))**2+(np.imag(fine_z[0])-np.imag(fine_z))**2)
        dist2 = np.sqrt((np.real(fine_z[-1])-np.real(fine_z))**2+(np.imag(fine_z[-1])-np.imag(fine_z))**2)
        fr_guess_index = np.argmax((dist1+dist2))
        fr_guess = fine_x[fr_guess_index]
        fine_z_derot = (fine_z-(off_res_i+1.j*off_res_q))*np.exp(1j*(-phi_guess))+(off_res_i+1.j*off_res_q)
        #fr_guess_index = np.argmin(np.abs(fine_z_derot))
        #fr_guess = fine_x[fr_guess_index]
        mag_max = np.max(np.abs(fine_z_derot)**2)
        mag_min = np.min(np.abs(fine_z_derot)**2)
        mag_3dB = (mag_max+mag_min)/2.
        half_distance = np.abs(fine_z_derot)**2-mag_3dB
        right = half_distance[np.argmin(np.abs(fine_z_derot)):-1]
        left  = half_distance[0:np.argmin(np.abs(fine_z_derot))]
        right_index = np.argmin(np.abs(right))+np.argmin(np.abs(fine_z_derot))
        left_index = np.argmin(np.abs(left))
        Q_guess_Hz = fine_x[right_index]-fine_x[left_index]
        Q_guess = fr_guess/Q_guess_Hz
        #also fix amp guess
        d = np.max(20*np.log10(np.abs(gain_z)))-np.min(20*np.log10(np.abs(fine_z_derot)))
        amp_guess = 0.0037848547850284574+0.11096782437821565*d-0.0055208783469291173*d**2+0.00013900471000261687*d**3+-1.3994861426891861e-06*d**4
    
    #guess non-linearity parameter
    #might be able to guess this by ratioing the distance between min and max distance between iq points in fine sweep
    a_guess = 0
    
    #b0 and b1 guess
    xlin = (gain_x - fr_guess)/fr_guess
    b1_guess = (np.abs(gain_z)[-1]**2-np.abs(gain_z)[0]**2)/(xlin[-1]-xlin[0])
    b0_guess = np.max((np.max(np.abs(fine_z)**2),np.max(np.abs(gain_z)**2)))
        
    #cabel delay guess tau
    #y = mx +b
    #m = (y2 - y1)/(x2-x1)
    #b = y-mx
    m = (gain_phase - np.roll(gain_phase,1))/(gain_x-np.roll(gain_x,1))
    b = gain_phase -m*gain_x
    m_best = np.median(m[~np.isnan(m)])
    tau_guess = m_best/(2*np.pi)
       
    if verbose == True:
        print("fr guess  = %.3f MHz" %(fr_guess/10**6))
        print("Q guess   = %.2f kHz, %.1f" % ((Q_guess_Hz/10**3),Q_guess))
        print("amp guess = %.2f" %amp_guess)
        print("phi guess = %.2f" %phi_guess)
        print("b0 guess  = %.2f" %b0_guess)
        print("b1 guess  = %.2f" %b1_guess)
        print("tau guess = %.2f x 10^-7" %(tau_guess/10**-7))
    
    x0 = [fr_guess,Q_guess,amp_guess,phi_guess,a_guess,b0_guess,b1_guess,fr_guess]
    return x0

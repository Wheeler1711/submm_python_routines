import numpy as np
import scipy.optimize as optimization


#module for fitting resonances curves for kinetic inductance detectors.
# written by Jordan Wheeler 12/21/16

# for example see test_fit.py in this directory

#To Do
#I think the error analysis on the fit_nonlinear_iq_with_err probably needs some work
# need to add in fitting in S21 space. (less parameters might be easier for an initial fit to inform the iq fit)

#Change log




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
    z = (i0 +1.j*q0)* np.exp(-1.0j* 2* np.pi *deltaf*tau) * (1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*y)+ amp/2.*(np.exp(1.0j*phi) -1.0))
    real_z = np.real(z)
    imag_z = np.imag(z)
    return np.hstack((real_z,imag_z))


# function for fitting an iq sweep with the above equation
def fit_nonlinear_iq(x,z,**keywords):
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple    
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(x),2000,.01,-4.0*np.pi,0,-5,-5,1*10**-9,np.min(x)],[np.max(x),200000,100,4.0*np.pi,5,5,5,1*10**-6,np.max(x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        fr_guess = x[np.argmin(np.abs(z))]
        x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
    z_stacked = np.hstack((np.real(z),np.imag(z)))    
    fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,bounds = bounds)
    fit_result = nonlinear_iq(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    x0_result = nonlinear_iq(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8])

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0}
    return fit_dict

# same function but double fits so that it can get error and a proper covariance matrix out
def fit_nonlinear_iq_with_err(x,z,**keywords):
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple    
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(x),2000,.01,-4.0*np.pi,0,-5,-5,1*10**-9,np.min(x)],[np.max(x),200000,100,4.0*np.pi,5,5,5,1*10**-6,np.max(x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        fr_guess = x[np.argmin(np.abs(z))]
        x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
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
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0}
    return fit_dict





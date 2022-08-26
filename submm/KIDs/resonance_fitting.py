import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt
try:
    from submm_python_routines.KIDs import calibrate
except:
    from submm.KIDs import calibrate
from numba import jit # to get working on python 2 I had to downgrade llvmlite pip install llvmlite==0.31.0
# numba seems to make the fitting 10 times faster

# module for fitting resonances curves for kinetic inductance detectors.
# written by Jordan Wheeler 12/21/16

# for example see res_fit.ipynb in this demos directory

# To Do
# I think the error analysis on the fit_nonlinear_iq_with_err probably needs some work
# add in step by step fitting i.e. first amplitude normalizaiton, then cabel delay, then i0,q0 subtraction, then phase rotation, then the rest of the fit.
# need to have fit option that just specifies tau becuase that never really changes for your cryostat

#Change log
#JDW 2017-08-17 added in a keyword/function to allow for gain varation "amp_var" to be taken out before fitting
#JDW 2017-08-30 added in fitting for magnitude fitting of resonators i.e. not in iq space
#JDW 2018-03-05 added more clever function for guessing x0 for fits
#JDW 2018-08-23 added more clever guessing for resonators with large phi into guess seperate functions


J=np.exp(2j*np.pi/3)
Jc=1/J

@jit(nopython=True) 
def cardan(a,b,c,d):
    '''
    analytical root finding fast: using numba looks like x10 speed up
    returns only the largest real root
    '''
    u=np.empty(2,np.complex128)
    z0=b/3/a
    a2,b2 = a*a,b*b    
    p=-b2/3/a2 +c/a
    q=(b/27*(2*b2/a2-9*c/a)+d)/a
    D=-4*p*p*p-27*q*q
    r=np.sqrt(-D/27+0j)        
    u=((-q-r)/2)**(1/3.)#0.33333333333333333333333
    v=((-q+r)/2)**(1/3.)#0.33333333333333333333333
    w=u*v
    w0=np.abs(w+p/3)
    w1=np.abs(w*J+p/3)
    w2=np.abs(w*Jc+p/3)
    if w0<w1: 
        if w2<w0 : v*=Jc
    elif w2<w1 : v*=Jc
    else: v*=J
    roots = np.asarray((u+v-z0, u*J+v*Jc-z0,u*Jc+v*J-z0))
    #print(roots)
    where_real = np.where(np.abs(np.imag(roots)) < 1e-15)
    #if len(where_real)>1: print(len(where_real))
    #print(D)
    if D>0: return np.max(np.real(roots)) # three real roots
    else: return np.real(roots[np.argsort(np.abs(np.imag(roots)))][0]) #one real root get the value that has smallest imaginary component
    #return np.max(np.real(roots[where_real]))

#return np.asarray((u+v-z0, u*J+v*Jc-z0,u*Jc+v*J-z0))


# function to descript the magnitude S21 of a non linear resonator
@jit(nopython=True) 
def nonlinear_mag(x,fr,Qr,amp,phi,a,b0,b1,flin):
    '''
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
    #                          /        (j phi)            (j phi)   |  2
    #|S21|^2 = (b0+b1 x_lin)* |1 -amp*e^           +amp*(e^       -1) |^
    #                         |   ------------      ----              |
    #                          \     (1+ 2jy)         2              /
    #
    # where the nonlineaity of y is described by the following eqution taken from Response of superconducting microresonators
    # with nonlinear kinetic inductance
    #                                     yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr
    #    
    '''
    xlin = (x - flin)/flin
    xg = (x-fr)/fr
    yg = Qr*xg
    y = np.zeros(x.shape[0])
    #find the roots of the y equation above
    for i in range(0,x.shape[0]):
        # 4y^3+ -4yg*y^2+ y -(yg+a)
        #roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        #roots = cardan(4.0,-4.0*yg[i],1.0,-(yg[i]+a))
        #print(roots)
        #roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))   #more accurate version that doesn't seem to change the fit at al     
        # only care about real roots
        #where_real = np.where(np.imag(roots) == 0)
        #where_real = np.where(np.abs(np.imag(roots)) < 1e-10) #analytic version has some floating point error accumulation
        y[i] = cardan(4.0,-4.0*yg[i],1.0,-(yg[i]+a))#np.max(np.real(roots[where_real]))
    z = (b0 +b1*xlin)*np.abs(1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*y) + amp/2.*(np.exp(1.0j*phi) -1.0))**2
    return z

jit(nopython=True)
def linear_mag(x,fr,Qr,amp,phi,b0):
    '''
    # simplier version for quicker fitting when applicable
    # x is the frequeciesn your iq sweep covers
    # fr is the center frequency of the resonator
    # Qr is the quality factor of the resonator
    # amp is Qr/Qc
    # phi is a rotation paramter for an impedance mismatch between the resonaotor and the readout system
    # b0 DC level of s21 away from resonator
    #
    # This is based of fitting code from MUSIC
    # The idea is we are producing a model that is described by the equation below
    # the frist two terms in the large parentasis and all other terms are farmilar to me
    # but I am not sure where the last term comes from though it does seem to be important for fitting
    #
    #                 /        (j phi)            (j phi)   |  2
    #|S21|^2 = (b0)* |1 -amp*e^           +amp*(e^       -1) |^
    #                |   ------------      ----              |
    #                 \     (1+ 2jxg)         2              /
    #
    # no y just xg
    # with no nonlinear kinetic inductance
    '''
    if not np.isscalar(fr): #vectorize breaks numba though
        x = np.reshape(x,(x.shape[0],1,1,1,1,1))
    xg = (x-fr)/fr
    z = (b0)*np.abs(1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*xg*Qr) + amp/2.*(np.exp(1.0j*phi) -1.0))**2
    return z


 

# function to describe the i q loop of a nonlinear resonator
@jit(nopython=True) 
def nonlinear_iq(x,fr,Qr,amp,phi,a,i0,q0,tau,f0):
    '''
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
    #                    (-j 2 pi deltaf tau)  /        (j phi)            (j phi)   |
    #        (i0+j*q0)*e^                    *|1 -amp*e^           +amp*(e^       -1) |
    #                                         |   ------------      ----              |
    #                                          \     (1+ 2jy)         2              /
    #
    # where the nonlineaity of y is described by the following eqution taken from Response of superconducting microresonators
    # with nonlinear kinetic inductance
    #                                     yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr
    #    
    '''
    deltaf = (x - f0)
    xg = (x-fr)/fr
    yg = Qr*xg
    y = np.zeros(x.shape[0])
    #find the roots of the y equation above
    for i in range(0,x.shape[0]):
        # 4y^3+ -4yg*y^2+ y -(yg+a)
        #roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        #roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))   #more accurate version that doesn't seem to change the fit at al     
        # only care about real roots
        #where_real = np.where(np.imag(roots) == 0)
        #y[i] = np.max(np.real(roots[where_real]))
        y[i] = cardan(4.0,-4.0*yg[i],1.0,-(yg[i]+a))
    z = (i0 +1.j*q0)* np.exp(-1.0j* 2* np.pi *deltaf*tau) * (1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*y) + amp/2.*(np.exp(1.0j*phi) -1.0))
    return z



@jit(nopython=True) 
def nonlinear_iq_for_fitter(x,fr,Qr,amp,phi,a,i0,q0,tau,f0):
    '''
    when using a fitter that can't handel complex number 
    one needs to return both the real and imaginary components seperatly
    '''

    deltaf = (x - f0)
    xg = (x-fr)/fr
    yg = Qr*xg
    y = np.zeros(x.shape[0])
    
    for i in range(0,x.shape[0]):
        #roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        #where_real = np.where(np.imag(roots) == 0)
        #y[i] = np.max(np.real(roots[where_real]))
        y[i] = cardan(4.0,-4.0*yg[i],1.0,-(yg[i]+a))
    z = (i0 +1.j*q0)* np.exp(-1.0j* 2* np.pi *deltaf*tau) * (1.0 - amp*np.exp(1.0j*phi)/ (1.0 +2.0*1.0j*y) + amp/2.*(np.exp(1.0j*phi) -1.0))
    real_z = np.real(z)
    imag_z = np.imag(z)
    return np.hstack((real_z,imag_z))


def brute_force_linear_mag_fit(x,z,ranges,n_grid_points,error = None, plot = False,**keywords):
    '''
    x frequencies Hz
    z complex or abs of s21
    ranges is the ranges for each parameter i.e. np.asarray(([f_low,Qr_low,amp_low,phi_low,b0_low],[f_high,Qr_high,amp_high,phi_high,b0_high]))
    n_grid_points how finely to sample each parameter space.
    this can be very slow for n>10
    an increase by a factor of 2 will take 2**5 times longer
    to marginalize over you must minimize over the unwanted axies of sum_dev
    i.e for fr np.min(np.min(np.min(np.min(fit['sum_dev'],axis = 4),axis = 3),axis = 2),axis = 1)
    '''
    if error is None:
        error = np.ones(len(x))

    fs = np.linspace(ranges[0][0],ranges[1][0],n_grid_points)
    Qrs = np.linspace(ranges[0][1],ranges[1][1],n_grid_points)
    amps = np.linspace(ranges[0][2],ranges[1][2],n_grid_points)
    phis = np.linspace(ranges[0][3],ranges[1][3],n_grid_points)
    b0s = np.linspace(ranges[0][4],ranges[1][4],n_grid_points)
    evaluated_ranges = np.vstack((fs,Qrs,amps,phis,b0s))

    a,b,c,d,e = np.meshgrid(fs,Qrs,amps,phis,b0s,indexing = "ij") #always index ij

    evaluated = linear_mag(x,a,b,c,d,e)
    data_values = np.reshape(np.abs(z)**2,(abs(z).shape[0],1,1,1,1,1))
    error = np.reshape(error,(abs(z).shape[0],1,1,1,1,1))
    sum_dev = np.sum(((np.sqrt(evaluated)-np.sqrt(data_values))**2/error**2),axis = 0) # comparing in magnitude space rather than magnitude squared
    
    min_index = np.where(sum_dev == np.min(sum_dev))
    index1 = min_index[0][0]
    index2 = min_index[1][0]
    index3 = min_index[2][0]
    index4 = min_index[3][0]
    index5 = min_index[4][0]
    fit_values = np.asarray((fs[index1],Qrs[index2],amps[index3],phis[index4],b0s[index5]))
    fit_values_names = ('f0','Qr','amp','phi','b0')
    fit_result = linear_mag(x,fs[index1],Qrs[index2],amps[index3],phis[index4],b0s[index5])

    marginalized_1d = np.zeros((5,n_grid_points))
    marginalized_1d[0,:] = np.min(np.min(np.min(np.min(sum_dev,axis = 4),axis = 3),axis = 2),axis = 1)
    marginalized_1d[1,:] = np.min(np.min(np.min(np.min(sum_dev,axis = 4),axis = 3),axis = 2),axis = 0)
    marginalized_1d[2,:] = np.min(np.min(np.min(np.min(sum_dev,axis = 4),axis = 3),axis = 1),axis = 0)
    marginalized_1d[3,:] = np.min(np.min(np.min(np.min(sum_dev,axis = 4),axis = 2),axis = 1),axis = 0)
    marginalized_1d[4,:] = np.min(np.min(np.min(np.min(sum_dev,axis = 3),axis = 2),axis = 1),axis = 0)

    marginalized_2d = np.zeros((5,5,n_grid_points,n_grid_points))
    #0 _
    #1 x _
    #2 x x _
    #3 x x x _ 
    #4 x x x x _
    #  0 1 2 3 4
    marginalized_2d[0,1,:] = marginalized_2d[1,0,:] = np.min(np.min(np.min(sum_dev,axis = 4),axis = 3),axis = 2)
    marginalized_2d[2,0,:] = marginalized_2d[0,2,:] = np.min(np.min(np.min(sum_dev,axis = 4),axis = 3),axis = 1)
    marginalized_2d[2,1,:] = marginalized_2d[1,2,:] = np.min(np.min(np.min(sum_dev,axis = 4),axis = 3),axis = 0)
    marginalized_2d[3,0,:] = marginalized_2d[0,3,:] = np.min(np.min(np.min(sum_dev,axis = 4),axis = 2),axis = 1)
    marginalized_2d[3,1,:] = marginalized_2d[1,3,:] = np.min(np.min(np.min(sum_dev,axis = 4),axis = 2),axis = 0)
    marginalized_2d[3,2,:] = marginalized_2d[2,3,:] = np.min(np.min(np.min(sum_dev,axis = 4),axis = 1),axis = 0)
    marginalized_2d[4,0,:] = marginalized_2d[0,4,:] = np.min(np.min(np.min(sum_dev,axis = 3),axis = 2),axis = 1)
    marginalized_2d[4,1,:] = marginalized_2d[1,4,:] = np.min(np.min(np.min(sum_dev,axis = 3),axis = 2),axis = 0)
    marginalized_2d[4,2,:] = marginalized_2d[2,4,:] = np.min(np.min(np.min(sum_dev,axis = 3),axis = 1),axis = 0)
    marginalized_2d[4,3,:] = marginalized_2d[3,4,:] = np.min(np.min(np.min(sum_dev,axis = 2),axis = 1),axis = 0)

    if plot:
        levels = [2.3,4.61] #delta chi squared two parameters 68 90 % confidence
        fig_fit = plt.figure(-1)
        axs = fig_fit.subplots(5, 5)
        for i in range(0,5): # y starting from top
            for j in range(0,5): #x starting from left
                if i > j:
                    #plt.subplot(5,5,i+1+5*j)
                    #axs[i, j].set_aspect('equal', 'box')
                    extent = [evaluated_ranges[j,0],evaluated_ranges[j,n_grid_points-1],evaluated_ranges[i,0],evaluated_ranges[i,n_grid_points-1]]
                    axs[i,j].imshow(marginalized_2d[i,j,:]-np.min(sum_dev),extent =extent,origin = 'lower', cmap = 'jet')
                    axs[i,j].contour(evaluated_ranges[j],evaluated_ranges[i],marginalized_2d[i,j,:]-np.min(sum_dev),levels = levels,colors = 'white')
                    axs[i,j].set_ylim(evaluated_ranges[i,0],evaluated_ranges[i,n_grid_points-1])
                    axs[i,j].set_xlim(evaluated_ranges[j,0],evaluated_ranges[j,n_grid_points-1])
                    axs[i,j].set_aspect((evaluated_ranges[j,0]-evaluated_ranges[j,n_grid_points-1])/(evaluated_ranges[i,0]-evaluated_ranges[i,n_grid_points-1]))
                    if j == 0:
                        axs[i, j].set_ylabel(fit_values_names[i])
                    if i == 4:
                        axs[i, j].set_xlabel("\n"+fit_values_names[j])
                    if i<4:
                        axs[i,j].get_xaxis().set_ticks([])
                    if j>0:
                        axs[i,j].get_yaxis().set_ticks([])

                elif i < j:
                    fig_fit.delaxes(axs[i,j])

        for i in range(0,5):
            #axes.subplot(5,5,i+1+5*i)
            axs[i,i].plot(evaluated_ranges[i,:],marginalized_1d[i,:]-np.min(sum_dev))
            axs[i,i].plot(evaluated_ranges[i,:],np.ones(len(evaluated_ranges[i,:]))*1.,color = 'k')
            axs[i,i].plot(evaluated_ranges[i,:],np.ones(len(evaluated_ranges[i,:]))*2.7,color = 'k')
            axs[i,i].yaxis.set_label_position("right")
            axs[i,i].yaxis.tick_right()
            axs[i,i].xaxis.set_label_position("top")
            axs[i,i].xaxis.tick_top()
            axs[i,i].set_xlabel(fit_values_names[i])

        #axs[0,0].set_ylabel(fit_values_names[0])
        #axs[4,4].set_xlabel(fit_values_names[4])
        axs[4,4].xaxis.set_label_position("bottom")
        axs[4,4].xaxis.tick_bottom()
                                                    

    #make a dictionary to return
    fit_dict = {'fit_values': fit_values,'fit_values_names':fit_values_names, 'sum_dev': sum_dev, 'fit_result': fit_result,'marginalized_2d':marginalized_2d,'marginalized_1d':marginalized_1d,'evaluated_ranges':evaluated_ranges}#, 'x0':x0, 'z':z}
    return fit_dict


# function for fitting an iq sweep with the above equation
def fit_nonlinear_iq(x,z,verbose = True,**keywords):
    '''
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat 
    # tau forces tau to specific value
    # tau_guess fixes the guess for tau without have to specifiy all of x0
    '''
    if ('tau' in keywords):
        use_given_tau = True
        tau = keywords['tau']
    else:
        use_given_tau = False
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        if verbose:
            print("default bounds used")
        bounds = ([np.min(x),50,.01,-np.pi,0,-np.inf,-np.inf,0,np.min(x)],[np.max(x),200000,1,np.pi,5,np.inf,np.inf,1*10**-6,np.max(x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        if verbose:
            print("default initial guess used")
        #fr_guess = x[np.argmin(np.abs(z))]
        #x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
        x0 = guess_x0_iq_nonlinear(x,z,verbose = verbose)
        #print(x0)
    if ('fr_guess' in keywords):
        x0[0] = keywords['fr_guess']
    if ('tau_guess' in keywords):
        x0[7] = keywords['tau_guess']
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
    
    if use_given_tau == True:
        del bounds[0][7]
        del bounds[1][7]
        del x0[7]
        fit = optimization.curve_fit(lambda x_lamb,a,b,c,d,e,f,g,h: nonlinear_iq_for_fitter(x_lamb,a,b,c,d,e,f,g,tau,h), x, z_stacked,x0,bounds = bounds)
        
        fit = list(fit)
        fit[0] = np.insert(fit[0],7,tau)
        #fill covariance matrix#
        cov = np.ones((fit[1].shape[0]+1,fit[1].shape[1]+1))*-1
        cov[0:7,0:7] = fit[1][0:7,0:7]
        cov[8,8] = fit[1][7,7]
        cov[8,0:7] = fit[1][7,0:7]
        cov[0:7,8] = fit[1][0:7,7]
        fit[1] = cov
        fit = tuple(fit)

        #print(fit[1])

        x0 = np.insert(x0,7,tau)

    else:
        fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,bounds = bounds)

    # human readable results
    fr = fit[0][0]
    Qr = fit[0][1]
    amp = fit[0][2]
    phi = fit[0][3]
    a = fit[0][4]
    i0 = fit[0][5]
    q0 = fit[0][6]
    tau = fit[0][7]
    Qc = Qr / amp
    Qi = 1.0 / ((1.0 / Qr) - (1.0 / Qc))
           
    fit_result = nonlinear_iq(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    x0_result = nonlinear_iq(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8])

    if verbose:
        print_fit_string_nonlinear_iq(fit[0],print_header = False,label = "Fit  ")

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,
                    'fr':fr,'Qr':Qr,'amp':amp,'phi':phi,'a':a,'i0':i0,'q0':q0,'tau':tau,'Qi':Qi,'Qc':Qc}
    return fit_dict


def fit_nonlinear_iq_sep(fine_x,fine_z,gain_x,gain_z,**keywords):
    '''
    # same as above funciton but takes fine and gain scans seperatly
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat  
    '''
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(fine_x),500.,.01,-np.pi,0,-np.inf,-np.inf,1*10**-9,np.min(fine_x)],[np.max(fine_x),1000000,1,np.pi,5,np.inf,np.inf,1*10**-6,np.max(fine_x)])
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
            
    if (('fine_z_err' in keywords) & ('gain_z_err' in keywords)):
        use_err = True
        fine_z_err = keywords['fine_z_err']
        gain_z_err = keywords['gain_z_err']
    else:
        use_err = False

    x = np.hstack((fine_x,gain_x))
    z = np.hstack((fine_z,gain_z))
    if use_err:
        z_err = np.hstack((fine_z_err,gain_z_err))
                

    if do_amp_norm == 1:
        z = amplitude_normalization(x,z)   
       
    z_stacked = np.hstack((np.real(z),np.imag(z)))
    if use_err:
        z_err_stacked = np.hstack((np.real(z_err),np.imag(z_err)))
        fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,sigma = z_err_stacked,bounds = bounds)
    else:
        fit = optimization.curve_fit(nonlinear_iq_for_fitter, x, z_stacked,x0,bounds = bounds) 
        
    fit_result = nonlinear_iq(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7],fit[0][8])
    x0_result = nonlinear_iq(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7],x0[8])

    if use_err:
        #only do it for fine data
        #red_chi_sqr = np.sum(z_stacked-np.hstack((np.real(fit_result),np.imag(fit_result))))**2/z_err_stacked**2)/(len(z_stacked)-8.)
        #only do it for fine data
        red_chi_sqr = np.sum((np.hstack((np.real(fine_z),np.imag(fine_z)))-np.hstack((np.real(fit_result[0:len(fine_z)]),np.imag(fit_result[0:len(fine_z)]))))**2/np.hstack((np.real(fine_z_err),np.imag(fine_z_err)))**2)/(len(fine_z)*2.-8.)


    #make a dictionary to return
    if use_err:
        fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,'fit_freqs':x,'red_chi_sqr':red_chi_sqr}
    else:
        fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,'fit_freqs':x}
    return fit_dict



# same function but double fits so that it can get error and a proper covariance matrix out
def fit_nonlinear_iq_with_err(x,z,**keywords):
    '''
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat 
    '''
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
def fit_nonlinear_mag(x,z,verbose = True,**keywords):
    '''
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat  
    '''
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(x),100,.01,-np.pi,0,-np.inf,-np.inf,np.min(x)],[np.max(x),200000,1,np.pi,5,np.inf,np.inf,np.max(x)])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        fr_guess = x[np.argmin(np.abs(z))]
        #x0 = [fr_guess,10000.,0.5,0,0,np.abs(z[0])**2,np.abs(z[0])**2,fr_guess]
        x0 = guess_x0_mag_nonlinear(x,z,verbose = verbose)

    fit = optimization.curve_fit(nonlinear_mag, x, np.abs(z)**2 ,x0,bounds = bounds)
    fit_result = np.sqrt(nonlinear_mag(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7]))
    x0_result = np.sqrt(nonlinear_mag(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7]))

    if verbose:
        print_fit_string_nonlinear_mag(fit[0],print_header = False,label = "Fit  ")

    # human readable results
    fr = fit[0][0]
    Qr = fit[0][1]
    amp = fit[0][2]
    phi = fit[0][3]
    a = fit[0][4]
    b0 = fit[0][5]
    b1 = fit[0][6]
    Qc = Qr / amp
    Qi = 1.0 / ((1.0 / Qr) - (1.0 / Qc))

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,
                    'fr':fr,'Qr':Qr,'amp':amp,'phi':phi,'a':a,'b0':b0,'b1':b1,'Qi':Qi,'Qc':Qc}
    return fit_dict

def fit_linear_mag(x,z,verbose = True,**keywords):
    '''
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat  
    '''
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([np.min(x),100,.01,-np.pi,-np.inf],[np.max(x),200000,1,np.pi,np.inf])
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")
        fr_guess = x[np.argmin(np.abs(z))]
        #x0 = [fr_guess,10000.,0.5,0,0,np.abs(z[0])**2,np.abs(z[0])**2,fr_guess]
        x0 = guess_x0_mag_nonlinear(x,z,verbose = verbose)
        x0 = np.delete(x0,[4,6,7])

    fit = optimization.curve_fit(linear_mag, x, np.abs(z)**2 ,x0,bounds = bounds)
    fit_result = np.sqrt(linear_mag(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4]))
    x0_result = np.sqrt(linear_mag(x,x0[0],x0[1],x0[2],x0[3],x0[4]))

    if verbose:
        print_fit_string_linear_mag(fit[0],print_header = False,label = "Fit  ")

    # human readable results
    fr = fit[0][0]
    Qr = fit[0][1]
    amp = fit[0][2]
    phi = fit[0][3]
    b0 = fit[0][4]
    Qc = Qr / amp
    Qi = 1.0 / ((1.0 / Qr) - (1.0 / Qc))

    #make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,
                    'fr':fr,'Qr':Qr,'amp':amp,'phi':phi,'b0':b0,'Qi':Qi,'Qc':Qc}
    return fit_dict

def fit_nonlinear_mag_sep(fine_x,fine_z,gain_x,gain_z,**keywords):
    '''
    # same as above but fine and gain scans are provided seperatly
    # keywards are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat 
    '''
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
    if (('fine_z_err' in keywords) & ('gain_z_err' in keywords)):
        use_err = True
        fine_z_err = keywords['fine_z_err']
        gain_z_err = keywords['gain_z_err']
    else:
        use_err = False
        

    #stack the scans for curvefit
    x = np.hstack((fine_x,gain_x))
    z = np.hstack((fine_z,gain_z))
    if use_err:
        z_err = np.hstack((fine_z_err,gain_z_err))
        z_err = np.sqrt(4*np.real(z_err)**2*np.real(z)**2+4*np.imag(z_err)**2*np.imag(z)**2) #propogation of errors left out cross term  
        fit = optimization.curve_fit(nonlinear_mag, x, np.abs(z)**2 ,x0,sigma = z_err,bounds = bounds)
    else:
        fit = optimization.curve_fit(nonlinear_mag, x, np.abs(z)**2 ,x0,bounds = bounds)
    fit_result = nonlinear_mag(x,fit[0][0],fit[0][1],fit[0][2],fit[0][3],fit[0][4],fit[0][5],fit[0][6],fit[0][7])
    x0_result = nonlinear_mag(x,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6],x0[7])

    #compute reduced chi squared
    print(len(z))
    if use_err:
        #red_chi_sqr = np.sum((np.abs(z)**2-fit_result)**2/z_err**2)/(len(z)-7.)
        # only use fine scan for reduced chi squared.
        red_chi_sqr = np.sum((np.abs(fine_z)**2-fit_result[0:len(fine_z)])**2/z_err[0:len(fine_z)]**2)/(len(fine_z)-7.)
    
    #make a dictionary to return
    if use_err:
        fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,'fit_freqs':x,'red_chi_sqr':red_chi_sqr}
    else:
        fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0':x0, 'z':z,'fit_freqs':x}
    return fit_dict

def amplitude_normalization(x,z):
    '''
    # normalize the amplitude varation requires a gain scan
    #flag frequencies to use in amplitude normaliztion
    '''
    index_use = np.where(np.abs(x-np.median(x))>100000) #100kHz away from resonator
    poly = np.polyfit(x[index_use],np.abs(z[index_use]),2)
    poly_func = np.poly1d(poly)
    normalized_data = z/poly_func(x)*np.median(np.abs(z[index_use]))
    return normalized_data

def amplitude_normalization_sep(gain_x,gain_z,fine_x,fine_z,stream_x,stream_z):
    '''
    # normalize the amplitude varation requires a gain scan
    # uses gain scan to normalize does not use fine scan
    #flag frequencies to use in amplitude normaliztion
    '''
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
    '''
    # this is lest robust than guess_x0_iq_nonlinear_sep 
    # below. it is recommended to use that instead   
    #make sure data is sorted from low to high frequency
    '''
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
    if len(gain_z)>1: #is there a gain scan?
        m = (gain_phase - np.roll(gain_phase,1))/(gain_x-np.roll(gain_x,1))
        b = gain_phase -m*gain_x
        m_best = np.median(m[~np.isnan(m)])
        tau_guess = m_best/(2*np.pi)
    else:
        tau_guess = 3*10**-9

    x0 = [fr_guess,Q_guess,amp_guess,phi_guess,a_guess,i0_guess,q0_guess,tau_guess,fr_guess]
        
    if verbose == True:
        print_fit_string_nonlinear_iq(x0)

    

    return x0

def guess_x0_mag_nonlinear(x,z,verbose = False):
    '''
    # this is lest robust than guess_x0_mag_nonlinear_sep 
    #below it is recommended to use that instead   
    #make sure data is sorted from low to high frequency
    '''
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
    
    if len(gain_z)>1:
        xlin = (gain_x - fr_guess)/fr_guess
        b1_guess = (np.abs(gain_z)[-1]**2-np.abs(gain_z)[0]**2)/(xlin[-1]-xlin[0])
    else:
        xlin = (fine_x - fr_guess)/fr_guess
        b1_guess = (np.abs(fine_z)[-1]**2-np.abs(fine_z)[0]**2)/(xlin[-1]-xlin[0])
    b0_guess = np.median(np.abs(gain_z)**2)
        
    x0 = [fr_guess,Q_guess,amp_guess,phi_guess,a_guess,b0_guess,b1_guess,fr_guess]
    
    if verbose == True:
        print_fit_string_nonlinear_mag(x0)
    

    return x0


def guess_x0_iq_nonlinear_sep(fine_x,fine_z,gain_x,gain_z,verbose = False):
    '''
    # this is the same as guess_x0_iq_nonlinear except that it takes
    # takes the fine scan and the gain scan as seperate variables
    # this runs into less issues when trying to sort out what part of 
    # data is fine and what part is gain for the guessing 
    #make sure data is sorted from low to high frequency
    '''

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
    xc, yc, R, residu  = calibrate.leastsq_circle(np.real(fine_z), np.imag(fine_z))
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
    '''
    # this is the same as guess_x0_mag_nonlinear except that it takes
    # takes the fine scan and the gain scan as seperate variables
    # this runs into less issues when trying to sort out what part of 
    # data is fine and what part is gain for the guessing 
    #make sure data is sorted from low to high frequency 
    '''

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
    xc, yc, R, residu  = calibrate.leastsq_circle(np.real(fine_z), np.imag(fine_z))
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


def fit_nonlinear_iq_multi(f,z,tau = None):
    '''
    wrapper for handling n resonator fits at once
    f and z should have shape n_iq_points x n_res points 
    return same thing as fitter but in arrays for all resonators
    '''

    center_freqs = f[f.shape[0]//2,:]

    all_fits = np.zeros((f.shape[1],9))
    all_fit_results = np.zeros((f.shape[1],f.shape[0]),dtype=np.complex_)
    all_x0_results = np.zeros((f.shape[1],f.shape[0]),dtype=np.complex_)
    all_masks = np.zeros((f.shape[1],f.shape[0]))
    all_x0 = np.zeros((f.shape[1],9))
    all_fr = np.zeros(f.shape[1])
    all_Qr = np.zeros(f.shape[1])
    all_amp = np.zeros(f.shape[1])
    all_phi = np.zeros(f.shape[1])
    all_a = np.zeros(f.shape[1])
    all_i0 = np.zeros(f.shape[1])
    all_q0 = np.zeros(f.shape[1])
    all_tau = np.zeros(f.shape[1])
    all_Qi = np.zeros(f.shape[1])
    all_Qc = np.zeros(f.shape[1])
        
    for i in range(0,f.shape[1]):
        f_single = f[:,i]
        z_single = z[:,i]
        #flag data that is too close to other resonators              
        distance = center_freqs-center_freqs[i]
        if center_freqs[i] != np.min(center_freqs): #don't do if lowest frequency resonator
            closest_lower_dist = -np.min(np.abs(distance[np.where(distance<0)]))
            closest_lower_index = np.where(distance ==closest_lower_dist)[0][0]
            halfway_low = (center_freqs[i] + center_freqs[closest_lower_index])/2.
        else:
            halfway_low = 0

        if center_freqs[i] != np.max(center_freqs): #don't do if highest frequenct
            closest_higher_dist = np.min(np.abs(distance[np.where(distance>0)]))
            closest_higher_index = np.where(distance ==closest_higher_dist)[0][0]
            halfway_high = (center_freqs[i] + center_freqs[closest_higher_index])/2.
        else:
            halfway_high = np.inf
           
        use_index = np.where(((f_single>halfway_low) & (f_single<halfway_high)))
        mask = np.zeros(len(f_single))
        mask[use_index] = 1
        f_single = f_single[use_index]
        z_single= z_single[use_index]
        

        try:
            if tau is not None:
                fit_dict_iq = fit_nonlinear_iq(f_single,z_single,tau = tau)
            else:
                fit_dict_iq = fit_nonlinear_iq(f_single,z_single)
                
            all_fits[i,:] = fit_dict_iq['fit'][0]
            #all_fit_results[i,:] = fit_dict_iq['fit_result']
            #all_x0_results[i,:] = fit_dict_iq['x0_result']
            all_fit_results[i,:]  = nonlinear_iq(f[:,i],all_fits[i,0],all_fits[i,1],all_fits[i,2],all_fits[i,3],all_fits[i,4],
                                                     all_fits[i,5],all_fits[i,6],all_fits[i,7],all_fits[i,8])
            all_x0_results[i,:] = nonlinear_iq(f[:,i],fit_dict_iq['x0'][0],fit_dict_iq['x0'][1],fit_dict_iq['x0'][2],
                                                   fit_dict_iq['x0'][3],fit_dict_iq['x0'][4],fit_dict_iq['x0'][5],
                                                   fit_dict_iq['x0'][6],fit_dict_iq['x0'][7],fit_dict_iq['x0'][8])
            all_masks[i,:] = mask
            all_x0[i,:] = fit_dict_iq['x0']
            all_fr[i] = fit_dict_iq['fr']
            all_Qr[i] = fit_dict_iq['Qr']
            all_amp[i] = fit_dict_iq['amp']
            all_phi[i] = fit_dict_iq['phi']
            all_a[i] = fit_dict_iq['a']
            all_i0[i] = fit_dict_iq['i0']
            all_q0[i] = fit_dict_iq['q0']
            all_tau[i] = fit_dict_iq['tau']
            all_Qc[i] = all_Qr[i]/all_amp[i]
            all_Qi[i] = 1.0 / ((1.0 / all_Qr[i]) - (1.0 / all_Qc[i]))
            
        except Exception as e:
            print(e)
            print("failed to fit")

    all_fits_dict = {'fits': all_fits, 'fit_results': all_fit_results, 'x0_results': all_x0_results, 'masks':all_masks,'x0':all_x0,
                    'fr':all_fr,'Qr':all_Qr,'amp':all_amp,'phi':all_phi,'a':all_a,'i0':all_i0,'q0':all_q0,'tau':all_tau,'Qi':all_Qi,'Qc':all_Qc}

    return all_fits_dict

def fit_linear_mag_multi(f,z):
    '''
    wrapper for handling n resonator fits at once
    f and z should have shape n_iq_points x n_res points 
    return same thing as fitter but in arrays for all resonators
    '''

    center_freqs = f[f.shape[0]//2,:]

    all_fits = np.zeros((f.shape[1],5))
    all_fit_results = np.zeros((f.shape[1],f.shape[0]))
    all_x0_results = np.zeros((f.shape[1],f.shape[0]))
    all_masks = np.zeros((f.shape[1],f.shape[0]))
    all_x0 = np.zeros((f.shape[1],5))
    all_fr = np.zeros(f.shape[1])
    all_Qr = np.zeros(f.shape[1])
    all_amp = np.zeros(f.shape[1])
    all_phi = np.zeros(f.shape[1])
    all_b0 = np.zeros(f.shape[1])
    all_Qi = np.zeros(f.shape[1])
    all_Qc = np.zeros(f.shape[1])
        
    for i in range(0,f.shape[1]):
        f_single = f[:,i]
        z_single = z[:,i]
        #flag data that is too close to other resonators              
        distance = center_freqs-center_freqs[i]
        if center_freqs[i] != np.min(center_freqs): #don't do if lowest frequency resonator
            closest_lower_dist = -np.min(np.abs(distance[np.where(distance<0)]))
            closest_lower_index = np.where(distance ==closest_lower_dist)[0][0]
            halfway_low = (center_freqs[i] + center_freqs[closest_lower_index])/2.
        else:
            halfway_low = 0

        if center_freqs[i] != np.max(center_freqs): #don't do if highest frequenct
            closest_higher_dist = np.min(np.abs(distance[np.where(distance>0)]))
            closest_higher_index = np.where(distance ==closest_higher_dist)[0][0]
            halfway_high = (center_freqs[i] + center_freqs[closest_higher_index])/2.
        else:
            halfway_high = np.inf
           
        use_index = np.where(((f_single>halfway_low) & (f_single<halfway_high)))
        mask = np.zeros(len(f_single))
        mask[use_index] = 1
        f_single = f_single[use_index]
        z_single= z_single[use_index]
        

        try:
            fit_dict_iq = fit_linear_mag(f_single,z_single)
            #ranges = np.asarray(([300*10**6,10,0,-3.14,8000],[500*10**6,20,1,3.14,10000]))
            #fit_dict_iq = brute_force_linear_mag_fit(f_single,z_single,ranges = ranges,n_grid_points = 10)
                
            all_fits[i,:] = fit_dict_iq['fit'][0]
            all_fit_results[i,:]  = np.sqrt(linear_mag(f[:,i],all_fits[i,0],all_fits[i,1],all_fits[i,2],
                                                           all_fits[i,3],all_fits[i,4]))
            all_x0_results[i,:] = np.sqrt(linear_mag(f[:,i],fit_dict_iq['x0'][0],fit_dict_iq['x0'][1],
                                                         fit_dict_iq['x0'][2],fit_dict_iq['x0'][3],fit_dict_iq['x0'][4]))
            all_masks[i,:] = mask
            all_x0[i,:] = fit_dict_iq['x0']
            all_fr[i] = fit_dict_iq['fr']
            all_Qr[i] = fit_dict_iq['Qr']
            all_amp[i] = fit_dict_iq['amp']
            all_phi[i] = fit_dict_iq['phi']
            all_b0[i] = fit_dict_iq['b0']
            all_Qc[i] = all_Qr[i]/all_amp[i]
            all_Qi[i] = 1.0 / ((1.0 / all_Qr[i]) - (1.0 / all_Qc[i]))
            
        except Exception as e:
            print("problem")
            print(e)
            print("failed to fit")

    all_fits_dict = {'fits': all_fits, 'fit_results': all_fit_results, 'x0_results': all_x0_results, 'masks':all_masks,'x0':all_x0,
                    'fr':all_fr,'Qr':all_Qr,'amp':all_amp,'phi':all_phi,'b0':all_b0,'Qi':all_Qi,'Qc':all_Qc}

    return all_fits_dict

def print_fit_string_nonlinear_iq(vals,print_header = True,label = "Guess"):
    Qc_guess = vals[1] / vals[2]
    Qi_guess = 1.0 / ((1.0 / vals[1]) - (1.0 / Qc_guess))
    if print_header:
        print("Resonator at %.2f MHz" %(vals[0]/10**6))
        print(f'     |                             Variables fit                           '+
                  '\033[1;30;42m|Derived variables|\033[0;0m')
    guess_header_str  =  '     |'
    guess_header_str += ' fr (MHz)|' 
    guess_header_str += '   Qr   |'
    guess_header_str += ' amp |'
    guess_header_str += ' phi  |'
    guess_header_str += ' a   |'
    guess_header_str += '   i0     |'
    guess_header_str += '   q0     |'
    guess_header_str += ' tau (ns)\033[1;30;42m|'
    guess_header_str += '   Qi   |'
    guess_header_str += '   Qc   |\033[0;0m'

    guess_str  =  label
    guess_str += f'| {"%3.4f" % (vals[0]/10**6)}' 
    guess_str += f'| {"%7.0f" % (vals[1])}'
    guess_str += f'| {"%0.2f" % (vals[2])}'
    guess_str += f'| {"% 1.2f" % (vals[3])}'
    guess_str += f'| {"%0.2f" % (vals[4])}'
    guess_str += f'| {"% .2E" % (vals[5])}'
    guess_str += f'| {"% .2E" % (vals[6])}'
    guess_str += f'| {"%6.2f" % (vals[7]*10**9)}  '
    guess_str += f'\033[1;30;42m| {"%7.0f" % (Qi_guess)}'
    guess_str += f'| {"%7.0f" % (Qc_guess)}|\033[0;0m'
    
    if print_header:
        print(guess_header_str)
    print(guess_str)

def print_fit_string_nonlinear_mag(vals,print_header = True,label = "Guess"):
    Qc_guess = vals[1] / vals[2]
    Qi_guess = 1.0 / ((1.0 / vals[1]) - (1.0 / Qc_guess))
    if print_header:
        print("Resonator at %.2f MHz" %(vals[0]/10**6))
        print(f'     |                       Variables fit                       '+
                  '\033[1;30;42m|Derived variables|\033[0;0m')
    guess_header_str  =  '     |'
    guess_header_str += ' fr (MHz)|' 
    guess_header_str += '   Qr   |'
    guess_header_str += ' amp |'
    guess_header_str += ' phi  |'
    guess_header_str += ' a   |'
    guess_header_str += '   b0     |'
    guess_header_str += '   b1     \033[1;30;42m|'
    guess_header_str += '   Qi   |'
    guess_header_str += '   Qc   |\033[0;0m'

    guess_str  =  label
    guess_str += f'| {"%3.4f" % (vals[0]/10**6)}' 
    guess_str += f'| {"%7.0f" % (vals[1])}'
    guess_str += f'| {"%0.2f" % (vals[2])}'
    guess_str += f'| {"% 1.2f" % (vals[3])}'
    guess_str += f'| {"%0.2f" % (vals[4])}'
    guess_str += f'| {"% .2E" % (vals[5])}'
    guess_str += f'| {"% .2E" % (vals[6])}'
    guess_str += f'\033[1;30;42m| {"%7.0f" % (Qi_guess)}'
    guess_str += f'| {"%7.0f" % (Qc_guess)}|\033[0;0m'
    
    if print_header:
        print(guess_header_str)
    print(guess_str)

def print_fit_string_linear_mag(vals,print_header = True,label = "Guess"):
    Qc_guess = vals[1] / vals[2]
    Qi_guess = 1.0 / ((1.0 / vals[1]) - (1.0 / Qc_guess))
    if print_header:
        print("Resonator at %.2f MHz" %(vals[0]/10**6))
        print(f'     |                       Variables fit                       '+
                  '\033[1;30;42m|Derived variables|\033[0;0m')
    guess_header_str  =  '     |'
    guess_header_str += ' fr (MHz)|' 
    guess_header_str += '   Qr   |'
    guess_header_str += ' amp |'
    guess_header_str += ' phi  |'
    guess_header_str += ' a   |'
    guess_header_str += '   b0     |'
    guess_header_str += '   b1     \033[1;30;42m|'
    guess_header_str += '   Qi   |'
    guess_header_str += '   Qc   |\033[0;0m'

    guess_str  =  label
    guess_str += f'| {"%3.4f" % (vals[0]/10**6)}' 
    guess_str += f'| {"%7.0f" % (vals[1])}'
    guess_str += f'| {"%0.2f" % (vals[2])}'
    guess_str += f'| {"% 1.2f" % (vals[3])}'
    guess_str += f'| --- '
    guess_str += f'| {"% .2E" % (vals[4])}'
    guess_str += f'| -------- '
    guess_str += f'\033[1;30;42m| {"%7.0f" % (Qi_guess)}'
    guess_str += f'| {"%7.0f" % (Qc_guess)}|\033[0;0m'
    
    if print_header:
        print(guess_header_str)
    print(guess_str)

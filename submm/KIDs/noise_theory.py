import numpy as np
import scipy.special as special
import scipy.optimize as optimization
import matplotlib.pyplot as plt

# this is a list of definitions that can be used to predict noise in KIDS
# right now it just contains the nessasary requirements for perdicting G-R noise in TiN
# I should expand it to include some of the definitions in Jason code
# this is adapted from some IDL code I got from Matt Bradford

#Written by Jordan on 12/12/2016

#Change log
# 1/6/2017 added nqp_min as a specified parameter for grnoise 



def nqp(t,tc,v,nqp_min):
    '''
    if ~keyword_set(nqp_min) then nqp_min=400. ; per cubic micron: zero-Temp residual QP density
    V is in cubic microns
    N0=1.72e10 for aluminum
    '''
    N0=4.e10 # for TiN
    N0 = N0/1.6e-19 # now microns^3 / Joule

    #Delta = double (3.5 * 1.381e-23 * tc)
    Delta = 1.74 * 1.381e-23 * tc #factor in delta is suspect but it would be canceled out by a factor in tc
    Nqp = v*2*N0*np.sqrt(2*np.pi*Delta*1.381e-23*t)*np.exp(-1*Delta/(1.381e-23*t))+v*nqp_min

    return Nqp

def deltaf_f(t, tc, nu, alpha, gamma):
    '''
    From Steve Hailey-Dunsheath, 2 Februay 2016
    Calculate a model for the fractional frequency shift due to change in bath temperature
    This is (alpha)*(gamma/2)*[sigma2(T)/sigma2(T=0)-1], where sigma2(T) is
    from equation 2.92 in Gao thesis, and equation 20 in Gao+08 JLTP

    nu is in MHz
    if ~keyword_set(alpha) then alpha=1
    if ~keyword_set(gamma) then gamma=1
    '''
    d_0 = 1.762*tc #factor of 1.762 is suspect
    xi = 6.626e-34*nu*1.e6/(2.*1.38e-23*t)
    model = -1.*alpha*gamma/2.*np.exp(-1.*d_0/t)*((2.*np.pi*t/d_0)**0.5 + 2.*np.exp(-1.*xi)*special.iv(0,xi))
    return model

def deltaf_f_vec(t,tc,nu,alpha,gamma):
    d_0 = 1.762*tc #factor of 1.762 is suspect
    t = np.reshape(t,(t.shape[0],1,1))
    xi = 6.626e-34*nu*1.e6/(2.*1.38e-23*t)
    model = -1.*alpha*gamma/2.*np.exp(-1.*d_0/t)*((2.*np.pi*t/d_0)**0.5 + 2.*np.exp(-1.*xi)*special.iv(0,xi))
    return model


def df_response(t,tc,f):
    '''
    calculate d (df/f) / dT via finite difference
    f is in MHz
    calls deltaf_f which computes frequency shift
    '''
    delta_t = t/100.
    dff_dt = (deltaf_f(t+delta_t,tc,f,1,1) - deltaf_f(t-delta_t,tc,f,1,1))/(2*delta_t)
    return dff_dt


def grnoise(t,tc,V,tau_qp,N0,f,nqp_min):
    '''
    this function calculates gr noise assuming constant tau
    fuction below responsivity probably does a better job. 
    V (volume in cubic microns)
    if ~keyword_set(N0) then N0=4.e10 ; microns^3 / eV
    if ~keyword_set(tau_qp) then tau_qp = 5e-6 ; sec
    if ~keyword_set(nqp_min) then nqp_min=400. ; QP per cubic micron at zero Temp
    '''
    N0 = N0/1.6e-19 # now microns^3 / Joule

    #ef^2 = 4 beta^2 Nqp tau_qp
    #beta = df_0 / d Nqp , so use (df_0/ dT) (dT / dNqp)

    #Delta = double (3.5 * 1.381e-23 * tc)
    Delta = 1.74*1.381e-23*tc
    #Nqp = v * 2 * N0 * sqrt(2*!pi*delta * 1.381e-23 * T) *exp(-1*Delta / (1.381e-23 * T)) + v * nqp_min
    dNqp_dt = V*2.*N0*np.sqrt(2*np.pi*Delta*1.318e-23)*np.exp(-1.*Delta/(1.381e-23*t))*(1./(2.*np.sqrt(t)) + np.sqrt(t)*Delta/1.381e-23/t**2)
    delta_t = t/100.
    dNqp_dt = (nqp(t+delta_t,tc,V,nqp_min)-nqp(t-delta_t,tc,V,nqp_min))/(2*delta_t)

    beta = df_response(t,tc,f)/dNqp_dt
    #assume a frequency of 100 MHz here, shouldn't matter.

    ef2 = 4. * beta**2 * nqp(t,tc,V,nqp_min) * tau_qp
    return ef2


def responsivity(temp,pabs,tc = 1.,N0 = 4.*10**10,nstar =100.,tau_max = 100.,eta_pb = 0.7,vol = 1.,fr = 100.,alpha_k = 1.,gamma_t = 1.,nu_opt = 250, n_gamma = 0.):
    '''
    Special thanks to Steve Hailey Dunsheath whose made this orginal function in idl 
    '''
    #Define various constants
    k_B = 1.381*10**-23 #Boltzmann constant [J K^-1]
    ev_joule = 1.6022*10**-19 #eV/J ratio [eV J^-1]
    h_p = 6.626*10**-34 #Planck constant [J s]

    #Compute the ratio of Delta_0/k_B [K]
    d0_kB = 1.764*tc

    # Compute n thermal
    nth = 2.*N0*k_B/ev_joule*np.sqrt(2.*np.pi*temp*d0_kB)*np.exp(-1.*d0_kB/temp)

    # Compute nqp
    # This expression has a term of the form [sqrt(1 + eps) - 1], where eps is small when nth and pabs are small.
    # When eps is small this term is not computed accurately, and here we explicitly linearize it.
    #nqp = nstar*(-1. + sqrt(1. + 2.*nth/nstar + (nth/nstar)^2. + 2.*eta_pb*pabs*1.e-12*tau_max*1.e-6/(nstar*vol*d0_kB*k_B)))
    eps = 2.*nth/nstar + (nth/nstar)**2. + 2.*eta_pb*pabs*1.*10**-12*tau_max*1.*10**-6/(nstar*vol*d0_kB*k_B)
    term = np.sqrt(1. + eps) - 1.
    indx = np.where(eps < 1*10**-8)
    count = len(indx)
    if (count != 0):
        term[indx] = 0.5*eps[indx]
    nqp = nstar*term
    
    #Compute tau_qp
    tau_qp = tau_max/(1. + nqp/nstar)

    # Compute S1 and S2
    xi = h_p*fr*1*10**6/(2.*k_B*temp)
    s1 = (2./np.pi)*np.sqrt(2.*d0_kB/(np.pi*temp))*np.sinh(xi)*special.kv(0,xi)
    s2 = 1. + np.sqrt(2.*d0_kB/(np.pi*temp))*np.exp(-1.*xi)*special.iv(0,xi)
    #s2 = 3.

    #Compute xr and Qi_inv
    #Note that xr refers to the frequency shift from the nqp = 0 state
    xr = -1.*alpha_k*gamma_t*s2*nqp/(4.*N0*d0_kB*k_B/ev_joule)
    Qi_inv = -1.*xr*2.*s1/s2

    #Compute the frequency and Qinv responsivity
    r_x = -1.*alpha_k*gamma_t*s2/(4.*N0*d0_kB*k_B/ev_joule)*eta_pb*tau_qp*1*10**-6/(d0_kB*k_B*vol)
    r_qinv = -1.*r_x*2.*s1/s2

    #Compute Sxx_gr and Sxx_gr0
    tau_th = tau_max/(1. + nth/nstar) #quasiparticle lifetime for a superconductor in thermal equilibrium at the specified temperature [microsec]
    gamma_th = nth*vol/2.*(1./tau_max + 1./tau_th)*1*10**6 #quasiparticle generation rate due to thermal fluctuations ;[sec^-1]
    gamma_r = nqp*vol/2.*(1./tau_max + 1./tau_qp)*1*10**6 #quasiparticle recombination rate ;[sec^-1]
    sxx_gr = (alpha_k*gamma_t*s2/(4.*N0*d0_kB*k_B/ev_joule))**2.*4.*(tau_qp*1*10**-6)**2./vol**2.*(gamma_th + gamma_r)
    sxx_gr0 = (alpha_k*gamma_t*s2/(4.*N0*d0_kB*k_B/ev_joule))**2.*4.*nqp/vol*tau_qp*1*10**-6

    #Compute Sxx_gamma
    sxx_gamma = (r_x)**2.*2.*h_p*nu_opt*1*10**9*pabs*1*10**-12*(1. + n_gamma)

    #Define the output dictionary and return
    dict = {'nth':nth, 
          'nqp':nqp, 
          'tau_qp':tau_qp, 
          's1':s1, 
          's2':s2, 
          'xr':xr, 
          'Qi_inv':Qi_inv, 
          'r_x':r_x, 
          'r_qinv':r_qinv, 
          'sxx_gr':sxx_gr, 
          'sxx_gr0':sxx_gr0, 
          'sxx_gamma':sxx_gamma}
        
    return dict

def responsivity_help():
    print("The input variables are temp,pabs,tc = 1.,N0 = 4.*10**10,nstar =100.,tau_max = 100.,eta_pb = 0.7,vol = 1.,fr = 100.,alpha_k = 1.,gamma_t = 1.,nu_opt = 250, n_gamma = 0.")
    print("The output variables are nth, nqp, tau_qp, s1, s2, xr, Qi_inv, r_x, r_qinv, sxx_gr, sxx_gr0, sxx_gamma")


def f0dirshort(T, f00, Fdelta):
    '''
    fucntion that computes TLS frequency shift 
    f00 is center frequency in Hz at zero temperature, p2 is the product of filling factor and
    loss tangent Fdelta_TLS, returns the frequency in Hz
    Taken from Jiansong Gao's Matlab code
    '''
    if f00<10:
        print("WARNING f should be in Hz")
    if T[0] >100:
        print("Warning T should be in Kelvin")
    f01K = 20.8366*10**9
    ref0 =np.real(special.digamma(1/2 + 1/(2*np.pi*1j)*f00/f01K/T))-np.log(f00/f01K/T/(2*np.pi)); 
    y = f00 + f00*Fdelta*1/np.pi*ref0;
    return y

def fit_tls(T,f,sigma = None,plot = True,**keywords):
    '''
    T in kelvin 
    f in Hz
    '''
    x0 = np.asarray((f[0],1e-5))
    if sigma is not None:
        print("using error")
        print(sigma)
        fit = optimization.curve_fit(f0dirshort, T, f,x0,sigma=sigma,absolute_sigma = True)
    else:
        fit = optimization.curve_fit(f0dirshort, T, f, x0)

    if plot:
        plt.figure()
        plt.plot(T,(f-f[0])/f[0],'o',label = "data")
        fit_f = f0dirshort(T,fit[0][0],fit[0][1])
        plt.plot(T,(fit_f-fit_f[0])/fit_f[0],label = f"fit FDelta = %.2E" % fit[0][1])
        plt.xlabel("Temperature (K)")
        plt.ylabel("df/f")
        plt.legend()
    return fit

def fit_tc_brute(t,df_over_f,nuref,tc_range = (0.5,1.5),alpha_range = (0,1), n_grid_points=100, error=None, plot = True,Verbose = False,**keywords):
    '''
    brute force fitter for fitting Tc and alpha assuming gamma =1
    but of alpha and gamma are degenerate
    t is temperature in kelvin
    df_over_f is f-f0/f0 
    '''

    if error is None:
        error = np.ones(len(t))

    tc_values = np.linspace(tc_range[0], tc_range[1], n_grid_points)
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_grid_points)
    evaluated_ranges = np.vstack((tc_values, alpha_values))

    a, b= np.meshgrid(tc_values, alpha_values, indexing="ij")  # always index ij

    #evaluated = noise_profile_lor_vec(x[index_for_fitting], a, b, c, d)
    evaluated = deltaf_f_vec(t,a,nuref,b,1)
    data_values = np.reshape(df_over_f, (df_over_f.shape[0], 1, 1))
    error = np.reshape(error, (error.shape[0], 1, 1))
    #print(evaluated.shape)
    # print(data_values.shape)
    # print(error.shape)
    sum_dev = np.sum(((evaluated - data_values) ** 2 / error ** 2),
                     axis=0)  # comparing in magnitude space rather than magnitude squared


    min_index = np.where(sum_dev == np.min(sum_dev))
    if Verbose:
        print("grid values at minimum are")
        print(min_index)
    index1 = min_index[0][0]
    index2 = min_index[1][0]
    fit_values = np.asarray((tc_values[index1], alpha_values[index2]))

    fit_values_names = ('tc', 'alpha')
    fit_result = deltaf_f(t, tc_values[index1], nuref,alpha_values[index2],1)

    if plot:
        extent = [evaluated_ranges[1,0],evaluated_ranges[1,n_grid_points-1],evaluated_ranges[0,0],evaluated_ranges[0,n_grid_points-1]]
        aspect = (evaluated_ranges[1,n_grid_points-1]-evaluated_ranges[1,0])/(evaluated_ranges[0,n_grid_points-1]-evaluated_ranges[0,0])
        plt.figure()
        plt.imshow(np.log10(sum_dev-np.min(sum_dev)),extent =extent,aspect = aspect,origin = 'lower', cmap = 'jet')
        plt.xlabel("alpha")
        plt.ylabel("Tc")
        plt.colorbar(label = 'Log10(sum residuals squared)')

    fit_dict = {'fit_values': fit_values, 'fit_values_names': fit_values_names, 'sum_dev': sum_dev,
                'fit_result': fit_result,'evaluated_ranges': evaluated_ranges}  #'marginalized_2d':marginalized_2d,'marginalized_1d':marginalized_1d,
    return fit_dict

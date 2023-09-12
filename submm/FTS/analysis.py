'''
FTSanalysis.py

Package for doing anlaysis of Fourier Transform spectrometer interferograms
specifically desinged to do phase correction of the spectra such that all
signal is contained and in the real part of the FFT and just noise is 
contained in the imaginary part of the FFT 

Many contributions from Jason Austermann and Johannes Hubmayr

Main program is FTSanalysis
examples are
FTSanalysis(df.x,df.y,fmin = 0,fmax = 30,algorithm='Richards',plot = True,theta_fit_method = 'interp',theta_fit_degree = 1,fmin_filter = 4,fmax_filter = 8)

FTSanalysis(df.x,df.y,0,fmin = 0,fmax = 30,autofindZPD = True,algorithm='Mertz',plot = True,theta_fit_method = 'interp',theta_fit_degree = 1,fmin_filter = 3,fmax_filter = 20)
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fftpack
import scipy.integrate
from scipy import signal
from scipy.optimize import curve_fit



def poly_mod_2pi(f,*coeffs):
    '''
    polynomial function of degree deg deg modulus +/-pi
    i.e. a function for fitting a polynomical to angle where -pi and +pi are 
    equivalent
    '''
    index = np.where(f<0)
    #print(len(coeffs))
    if len(coeffs)>1: #some weirdness in how curve fit calls a function versus a regular call
        coeffs = coeffs
    else:
        coeffs = coeffs[0]
    #print(coeffs)
    y = np.zeros(len(f))
    for i in range(0,len(coeffs)):
        y = y + coeffs[i]*np.abs(f)**(i)

    y_mod = np.mod(y+np.pi,2*np.pi)-np.pi
    y_mod[index]= -y_mod[index]

    return y_mod

def poly_angle_complex_plane(f,*coeffs):
    '''
    polynomial function of degree deg deg modulus 2pi
    i.e. a function for fitting a polynomical to angle where 0 and 2pi are 
    equivalent
    '''
    #print(len(coeffs))
    if len(coeffs)>1: #some weirdness in how curve fit calls a function versus a regular call
        coeffs = coeffs
    else:
        coeffs = coeffs[0]
    #print(coeffs)
    y = np.zeros(len(f))
    for i in range(0,len(coeffs)):
        y = y + coeffs[i]*f**(i)

    y_mod = np.mod(y+np.pi,2*np.pi)-np.pi
    y_sin = np.sin(y_mod)
    y_cos = np.cos(y_mod)

    return np.hstack((y_sin,y_cos))
        

def fit_angle(frequencies,theta,min_poly,max_poly,deg = 1,plot = False,units = "wavenumber"):
    '''
    function to fit angle versus frequency with a polynomial
    probably best to fit in complex plane
    
    inputs:
    frequencies- 
    theta-
    fmin-
    fmax-
    deg-

    outputs:
    fit_dict-
         fit_dict['fit']-                output of curve fit
         fit_dict['fit_result']            y data for the fitted result
         fit_dict['initial_guess']    y data for the inital guess for fitting
    '''

    if units == "wavenumber":
        unit_scale = 1
    elif units == "GHz":
        unit_scale = 30
    elif units == "THz":
        unit_scale = 30/1000.
    else:
        print("please select a valid unit type")
        return

    theta_complex = np.hstack((np.sin(theta),np.cos(theta)))

    print("min_poly,max_poly",min_poly,max_poly)
    
    frequencies_use = frequencies[(frequencies*unit_scale>min_poly) & (frequencies*unit_scale<max_poly)]
    theta_use = theta[(frequencies*unit_scale>min_poly) & (frequencies*unit_scale<max_poly)]
    theta_complex = np.hstack((np.sin(theta_use),np.cos(theta_use)))

    
    if deg == 0:
        p0 = np.asarray([np.median(theta_use)])
        upper_bound = (np.pi)
        lower_bound = (-np.pi)
        bounds = (lower_bound,upper_bound)
    elif deg == 1:
        p0 = np.asarray((np.median(theta_use),0))
        upper_bound = (np.pi,np.inf)
        lower_bound = (-np.pi,-np.inf)
        bounds = (lower_bound,upper_bound)
    elif deg == 2:
        p0 = np.asarray([np.median(theta_use),0,0])
        upper_bound = (np.pi,np.inf,np.inf)
        lower_bound = (-np.pi,-np.inf,-np.inf)
        bounds = (lower_bound,upper_bound)
    else:
        print("Please choose a polynomial degree of 0,1, or 2")
        return None

    print(p0)
          

    
    fit = curve_fit(poly_angle_complex_plane,frequencies_use,theta_complex,p0 = p0,bounds = bounds)
    initial_guess = poly_mod_2pi(frequencies,p0)
    fit_result = poly_mod_2pi(frequencies,fit[0])
    fit_dict = {"fit":fit,"fit_result":fit_result,"initial_guess":initial_guess}

    if plot:
        plt.figure(figsize = (12,6))
        plt.subplot(211)
        plt.title("fit_angle function output")
        plt.xlabel("Fitted domain "+units)
        plt.ylabel("Phase (Radians)")
        plt.plot(np.sort(frequencies*unit_scale),initial_guess[np.argsort(frequencies)],label = "initial guess")
        plt.plot(np.sort(frequencies*unit_scale),fit_result[np.argsort(frequencies)],label = "fit result")
        plt.plot(frequencies*unit_scale,theta,"o",mec = "k",label = "data")
        plt.xlim(min_poly,max_poly)
        plt.legend()
        plt.subplot(212)
        plt.xlabel("Entire domain "+units)
        plt.ylabel("Phase (Radians)")
        plt.plot(np.sort(frequencies*unit_scale),initial_guess[np.argsort(frequencies)],label = "initial guess")
        plt.plot(np.sort(frequencies*unit_scale),fit_result[np.argsort(frequencies)],label = "fit result")
        plt.plot(frequencies*unit_scale,theta,".",label = "data")
        plt.fill([min_poly,max_poly,max_poly,min_poly], [-1.25*np.pi,-1.25*np.pi,1.25*np.pi,1.25*np.pi], 'grey', alpha=0.4)
        #plt.xlim(fmin,fmax)
        plt.ylim(-1.25*np.pi,1.25*np.pi)
        plt.legend()
        plt.show()
        
    return fit_dict


def interp_angle(f_for_interpolation,f_to_interp_from,theta_to_interp_from,debug = False):
    '''
    basically a standard interpolation of an angle behaves poorly when +pi changes to -pi and vice versa
    solution is to plot theta on a unit circle in the complex plane.
    then interpolate the new location in the complex plane from the nearest two points
    then to get the angle from that new point on the complex plane
    this is just a simple linear interpolation

    inputs:
    f_for_interpolation   is the data for which you want to generate new angles from 
    f_to_interp_from      is the domain from which you interpolate from
    theta_to_interp_from  is the range from which you will interpolate from

    outputs:
    theta_interpolated    is new angles interpolated from at f_for_interpolation
    '''
    sin_theta = np.sin(theta_to_interp_from)
    cos_theta = np.cos(theta_to_interp_from)
    theta_interpolated = np.array(())
    count = 0
    for f in f_for_interpolation:
        difference = f-f_to_interp_from
        closest_index = np.argmin(np.abs(difference))
        difference[closest_index] = np.max(np.abs(difference)) 
        second_closest_index = np.argmin(np.abs(difference))
        length_to_closest = np.abs(f-f_to_interp_from[closest_index])
        length_to_second_closest = np.abs(f-f_to_interp_from[second_closest_index])
        total_length = length_to_closest + length_to_second_closest
        sin_theta_interpolated = sin_theta[closest_index]*(length_to_second_closest/total_length)+sin_theta[second_closest_index]*(length_to_closest/total_length)
        cos_theta_interpolated = cos_theta[closest_index]*(length_to_second_closest/total_length)+cos_theta[second_closest_index]*(length_to_closest/total_length)
        theta_interpolated = np.append(theta_interpolated,np.arctan2(sin_theta_interpolated,cos_theta_interpolated))
        if debug:
            if count == 5:
                print(np.arctan2(sin_theta_interpolated,cos_theta_interpolated))
                print(theta_interpolated)
                plt.figure()
                plt.title(str(f)+","+str(f_to_interp_from[closest_index])+","+str(f_to_interp_from[second_closest_index]))
                plt.plot(sin_theta[closest_index],cos_theta[closest_index],"o",label = "closest")
                plt.plot(sin_theta[second_closest_index],cos_theta[second_closest_index],"o",label = "2nd closest")
                plt.plot(sin_theta_interpolated,cos_theta_interpolated,"o",label = "interpolated")
                plt.ylim(-1,1)
                plt.xlim(-1,1)
                plt.legend()
                plt.show()
                
        count = count+1
    print(theta_interpolated)

    return theta_interpolated

# loading data functions
def LoadFTSdata(filename):
    ''' current data format is the ascii output of the bluesky software '''
    f=open(filename,'r')
    lines=f.readlines()
    f.close()
    
    N=len(lines)
    #print 'Number of lines in file',filename,' : ',N
    
    # cut out the header
    found_data_start = False
    i=0
    while not found_data_start:
        if i==N:
            print('Start of data after header not found')
            return False
        l=lines[i]
        if 'OPD (cm)' in l:
            start_dex=i+1
            found_data_start=True
        i=i+1
    
    data = np.empty((N-start_dex,2))    
    for i in range(N-start_dex):
        dex=i+start_dex
        x=lines[dex][:-1].split(',')
        data[i]=np.array([float(x[0]),float(x[1])])
    data=data.transpose()
    return data



def RemoveDrift(x,y,deg=1,plot=False):
    ''' remove linear drift and DC level from interferogram x,y. 
    '''
#    xcut = np.concatenate((x[0:10],x[-10:]))
#    ycut = np.concatenate((y[0:10],y[-10:]))
    xcut=x
    ycut=y
    p = scipy.polyfit(x=xcut, y=ycut, deg=deg)
    
    if plot:
        plt.figure(figsize = (12,6))
        ax1 = plt.subplot(211)
        plt.plot(x,y,'b.-',label = "all data")
        plt.plot(xcut,ycut,'r+',label = "data used")
        plt.plot(xcut,scipy.polyval(p,xcut),'g-',label = "fitted polynomial")
        plt.xlabel('OPD (cm)')
        plt.ylabel('Detector response (arb)')
        plt.title('Offset and drift removal')
        plt.legend()
        plt.subplot(212,sharex = ax1)
        plt.plot(x,y-scipy.polyval(p,x),'b.-',label = "final data")
        plt.xlabel('OPD (cm)')
        plt.ylabel('Detector response (arb)')
        plt.title('Offset and drift removed')
        plt.legend()
        #print 'Drift: slope = ', p[0], ' offset = ',p[1]
        plt.show()
    
    y=y-scipy.polyval(p,x)
    return y

def FindZPD(x,y,plot=False):
    ''' find the zero path difference from the maximum intensity of the IFG '''
    z=(y-y.mean())**2
    ZPD=x[z==z.max()]
    if len(ZPD)>1:
        print('warning, more than one maximum found.  Estimated ZPD not single valued!  Using first index')
        ZPD = ZPD[0]
    ZPD_index=list(x).index(ZPD)
    if plot:
        plt.figure(figsize = (12,6))
        plt.plot(x,y,'b.-')
        plt.plot(x[ZPD_index],y[ZPD_index],'ro')
        plt.xlabel('OPD (cm)')
        plt.ylabel('Detector response (arb)')
        plt.title('Location of ZPD')
        plt.show()
    
    return ZPD_index,ZPD

def ReturnSymmetricIFG(x,y,ZPD_index,plot=False):
    ''' return only the symmetric portion of an asymmetric IFG given the index "ZPDdex" of the zero 
        path difference 
    '''
    N=ZPD_index*2
    xsym=x[0:N+1]
    ysym=y[0:N+1]
    
    if plot:
        plt.figure(figsize = (12,6))
        plt.plot(x,y,label = "Entire interferogram")
        plt.plot(xsym,ysym,label = "Symmetric interferogram")
        plt.xlabel('OPD (cm)')
        plt.ylabel('Detector response (arb)')
        plt.title('Symmetric portion of IFG')
        plt.legend()
        plt.show()
    return xsym,ysym

def FFTandFrequencies(x,y,plot=False,units = "wavenumber"):
    ''' return the FFT of y and the frequencies sampled assuming equally spaced samples in x '''

    if units == "wavenumber":
        unit_scale = 1
    elif units == "GHz":
        unit_scale = 30
    elif units == "THz":
        unit_scale = 30/1000.
    else:
        print("please select a valid unit type")
        return

    
    samp_int = x[1]-x[0]
    N=len(y)
    ffty=scipy.fftpack.fft(y)
    f=scipy.fftpack.fftfreq(N,samp_int)
    if plot:
        plt.figure(figsize = (12,6))
        plt.subplot(211)
        plt.title('Time and FFT space')
        plt.plot(x,y)
        plt.subplot(212)
        plt.plot(f*unit_scale,np.abs(ffty))
        plt.plot(f*unit_scale,np.real(ffty))
        plt.plot(f*unit_scale,np.imag(ffty))
        plt.legend(('abs','real','imag'))
        plt.xlabel(units)
        plt.show()        
    return f,ffty

def PlotSN(f,ffty,fmin,fmax): #should this be removed?
    ''' plot the signal to noise ratio of the fft.  Use an out of band 
        section of the fft to determine the noise level fmin to fmax
    '''
    f_cut = f[(f>fmin) & (f<fmax)]
    ffty_cut = ffty[(f>fmin) & (f<fmax)]
    
    s=np.std(ffty_cut)
    plt.plot(f,np.real(ffty)/s)
    plt.plot(f,np.imag(ffty)/s)
    plt.plot(f_cut,np.real(ffty_cut)/s)
    plt.plot(f_cut,np.imag(ffty_cut)/s)
    plt.xlabel('Wavenumber (cm$^{-1}$')
    plt.ylabel('S/N')
    plt.legend(('real','imag','real noise', 'imag noise'))
    plt.show()
    

def GetPhaseSpectrum(y): #should this be removed
    ''' return the cos(theta)[wn] and sin(theta)[wn] '''
    ffty=scipy.fftpack.fft(y)
    theta = np.angle(ffty)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    return costheta, sintheta


def ExamineSymmetry(x,y): #sounds good does it still work?
    
    # plot original data
    plt.figure()
    plt.plot(x,y)
    plt.xlabel('OPD (cm)')
    plt.ylabel('Detector response')
    plt.title('Raw IFG')
    
    # find zero path difference
    ZPD_index, ZPD = FindZPD(x,y)
    x=x-ZPD
    ypos = y[ZPD_index:ZPD_dex*2+1]
    yneg = y[0:ZPD_index+1][::-1]
    
    print(len(ypos), len(yneg))
    
    plt.figure()
    plt.plot(ypos)
    plt.plot(yneg)
    plt.xlabel('OPD (cm)')
    plt.ylabel('Detector response')
    plt.title('ZPD mirror')
    
    plt.figure()
    plt.plot(ypos-yneg)
    plt.xlabel('OPD (cm)')
    plt.ylabel('$\delta_{+}$ - $\delta_{-}$')
    plt.title('ZPD difference')
    plt.show()

def AverageMultipleIFG(filenames,v=5.0,deg=3, notch_freqs=[60.0,120.0,180.0,240.0,300.0,420.0,480.0,540.0], PLOT=False):
    ''' remove drift and noise pickup from several IFG and then average together '''
    # this is probably not working correctly
    # I like to average after FFT
    print('filenames')
    print(filenames)
    if plot:
        plt.figure(figsize = (12,6))
    for i in range(len(filenames)):
        data = LoadFTSdata(filenames[i])
        x=data[0]
        y=data[1]
        y=NotchFrequencies(x,y,v=v,freqs=notch_freqs,df=.2,plot=False) # remove noise pickup
        y=RemoveDrift(x,y,deg=deg,plot=False) # remove detector drift
        if plot:
            ax1 = plt.subplot(211)
            plt.plot(x,y,label=str(i))
            plt.title('Individual IFGs post processing')
        if i==0:
            y_all = y
        else:
            y_all=np.vstack((y_all,y))

    if len(filenames) == 1:
        m = y_all * 1.
        s = y_all * 0.
    else:
        m = np.mean(y_all,axis=0)
        s = np.std(y_all,axis=0)

    #print('lengths')
    #print(len(y_all))
    #print(len(m))
    #print(len(s))
    
    if plot:
        plt.legend()
        plt.subplot(212, sharex=ax1, sharey=ax1) #get plots to zoom together
        plt.title('Averaged response')
        #plt.errorbar(x=x, y=m, yerr=s)
        plt.plot(x,m)
        plt.xlabel('OPD (cm)')
        plt.ylabel('Response (arb)')
        plt.show()
    return x,m,s

def SpillOver(theta_lyot,sigma):
    ''' return the spillover past a lyot stop at full angle theta_lyot assuming a gaussian 
        beam with beam width sigma (sigma is *NOT* FWHM)
    '''
    gaussian = lambda x,a,: scipy.exp(-1*(x/a)**2/2.0)
    N = len(sigma)
    if N>0:
        y=np.zeros(N)
        for i in range(N):
            y[i]=scipy.integrate.quad(gaussian, a=-1*theta_lyot/2.0, b=theta_lyot/2.0, args=(sigma[i]))[0]
    else:
        y=scipy.integrate.quad(gaussian, a=-1*theta_lyot/2.0, b=theta_lyot/2.0, args=(sigma))[0]
    y = y/(np.sqrt(2*np.pi)*sigma) # normalize to total area under curve
    return y

def __SpillOverVsFrequency(f,f_0,sigma_0):
    ''' return the spillover as a function of frequency f, assuming 
        diffraction theory: sin(theta) = 1.22 lambda/d
        
        inputs:
        f: frequency
        f_0: frequency at which the beam width is known
        sigma_0: beam width in degrees at frequency f_0
    '''
    sigmas = np.arcsin(f_0/f*np.sin(sigma_0*np.pi/180.0))*180.0/np.pi 
    #sigmas = f_0/f*sigma_0   
    y=SpillOver(theta_lyot=27.2,sigma=sigmas)
    return y

def CorrectSpillover(f,B,f_start,f_stop,f_0,sigma_0):
    ''' correct for spillover past the FTS from frequency f_start 
        to f_stop of the spectrum B(f).  f_0 and sigma_0 used for 
        frequency scaling (see __SpillOverVsFrequency)
    '''
    
    B=B[(f>f_start)&(f<f_stop)]
    f=f[(f>f_start)&(f<f_stop)]
    C=__SpillOverVsFrequency(f,f_0,sigma_0)
    return f,B/C

def __GetDewarFilteringSpectralResponse(f):
    ''' return the spectral response of the dewar from the thermal 
        filtering.  No absorption considered since loss tangent of 
        nylon and teflon at low temperatures is non-existent
    '''
    
    f=f*1.0e9
    
    n_teflon = 1.44
    n_nylon = 1.72 # from lamb compendium
    d1 = 2.0 # 2 cm of teflon at stage 1
    d2 = 1.0 # 1 cm of teflon at stage 2
    d_nylon = 0.15875 # in cm, 1/16 of an inch
    
    R_stage1 = mm.trans_mr(f,1,n_teflon,1,d1)
    R_stage2 = mm.trans_mr(f,1,n_teflon,1,d2)
    R_nylon =  mm.trans_mr(f,1,n_nylon,1,d_nylon)
    #Transmission = R_stage1*R_stage2*R_nylon
    Transmission = R_nylon
    
    return Transmission

def IntegratePassband(f,B,fi,ff):
    ''' integrate the passband from limits fi to ff '''
    
    B=B[(f>fi)&(f<ff)]
    f=f[(f>fi)&(f<ff)]
    z = scipy.integrate.simps(y=B, x=f)
    #print(z)
    #plt.plot(f,B,'o-')
    #plt.show()
    return z    
    
######################################################################
# end-to-end analysis functions
    
def DoubleSidedIFG(x,y,ZPD,autofindZPD=True,window=True,plot=False):
    ''' Determine the spectrum from the symmetric portion of the IFG.  The analysis 
        follows the Mertz method of phase correction described in Griffiths and Haseth 
        pgs. 85 - 93
        
        x: path length difference
        y: detector response
        ZPD: position of zero path difference 
        autofindZPD: if True, find the ZPD using FindZPD() function, which declares 
                     position of maximum signal is ZPD
        window: if True, window the data with Hanning window
        plot: if True, plot the symmetric IFG, the phase spectrum and frequency spectrum
    '''
    
    # find zero path difference if autofind==True
    if autofindZPD:
        ZPD_index, ZPD = FindZPD(x,y)
        x=x-ZPD
    
    # get symmetric portion of scan
    ZPD_index=list(x).index(ZPD)
    y=y-y.mean()
    x=x[0:ZPD_index*2+1]
    y=y[0:ZPD_index*2+1]
    
    # window if you like
    if window:
        y=y*np.hanning(len(y))
        #y=y*scipy.signal.gaussian(len(y), std=len(y)/5.0, sym=True)
    
    # shift data such that ZPD is zeroth point and mirror image the points before ZPD
#    ytemp=np.empty(len(y))
#    ytemp[0:dex]=y[dex:]
#    ytemp[dex:2*dex]=y[0:dex]
#    y=ytemp
    
    # take FFT
    sample_int=x[1]-x[0]
    ffty=scipy.fftpack.fft(y)
    f=scipy.fftpack.fftfreq(len(y),sample_int)
    
    theta = np.angle(ffty)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    
    X = np.real(ffty)*costheta
    Y = np.imag(ffty)*sintheta
    
    PCS = X+Y
    
    if plot:
        # plot original data after apodization
        plt.figure()
        plt.plot(x,y)
        plt.xlabel('OPD (cm)')
        plt.ylabel('Detector response')
        plt.title('Raw double-sided IFG')
        
        # plot angles
        plt.figure()
        plt.plot(f*30,costheta,'o-')
        plt.plot(f*30,sintheta,'o-')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('cos/sin phase')
        plt.title('Phase response')
        plt.legend(('cos(theta)','sin(theta)'))
        plt.xlim((0,1200))
        
        plt.figure()
        plt.plot(f*30,X,'bo-')
        plt.plot(f*30,Y,'go-')
        plt.plot(f*30,PCS,'ro-')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Detector response (arb)')
        plt.title('Optical Frequency Response')
        plt.legend(('cos','sin','sum'))
        plt.xlim((0,1200))
        
        
        # plot angles
        plt.figure()
        plt.plot(f*30,np.unwrap(theta),'o-')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('phase')
        plt.title('Phase response')
        plt.xlim((0,1200))
        plt.show()
    return f,PCS

def FTSanalysis(x,
                    y,
                    ZPD = None,
                    min_filter=None,
                    max_filter=None,
                    plot=False,
                    algorithm='Richards',
                    theta_fit_method = "poly",
                    theta_fit_degree = 1,
                    min_poly = 210/30.,
                    max_poly = 300/30.,
                    units = "wavenumber",
                    window = "none"):
    ''' 
    Full FTS analysis following for single sided IFGs using an algorithm of 'Mertz' or 'Richards'

    inputs:
    x-                  distances of interferogram points in cm from 0 to total throw L
    y-                  interferogram intensity values
    ZPD-                the location of the white light fringe in same units as x
                        if not specified program will try to automatically find it. 
    min_filter-        minimum frequency below which to filter signal out of if using method Richards
    max_filter-        maximum frequency above which to filter signal out of if using method Richards
    autofindZPD-        to find ZPD rather than specifying it
    plot-               make a bunch of plots or not - always make plots if it is the first run 
    algorithm-          Mertz or Richards i.e. fix phase in frequency space or interferogram space
    theta_fit_method-   either poly or interp - poly is good for extrapolating the phase correction to frequencies with little S/N
    theta_fit_degree-   the degree of polynomial for fitting the phase if using theta_fit_method = poly can be 0, 1, or 2
    min_poly-               min for which to fit phase with if using theta_fit_method = poly (units wavenumber)
    max_poly-               max for which to fit phase with if using theta_fit_method = poly (units wavenumber)
    units-              the units for plotting can be wavenumber, GHz, or THz default wavenumber cm^-1
    window-             can be none or Hanning if you would like a Hanning window applied

    outputs:
    f-                  frequencies of in wavenumber of spectrum
    B-                  complex fourier transformed spectrum with signal 
                               in the real component and noise in the imaginary component
    '''
    if theta_fit_method != "poly":
        theta_fit_method = "interp"

    if units == "wavenumber":
        unit_scale = 1
    elif units == "GHz":
        unit_scale = 30
    elif units == "THz":
        unit_scale = 30/1000.
    else:
        print("please select a valid unit type")
        return

    # remove drift
    samp_int = x[1]-x[0]
    y=RemoveDrift(x=x,y=y,plot=plot)
    if ZPD == None:
        ZPD_index,ZPD = FindZPD(x,y,plot=plot)
    else:
        ZPD_index = np.argmin(np.abs(x-ZPD))
        ZPD = x[ZPD_index]
        if plot:
            plt.figure()
            plt.title("Manually specified Zero Path Length difference (ZPD) location")
            plt.plot(x,y)
            plt.plot(x[ZPD_index],y[ZPD_index],"*",label = "ZPD")
            plt.legend()
    
    # get symmetric portion of IFG and get phase spectrum
    xsym,ysym = ReturnSymmetricIFG(x,y,ZPD_index=ZPD_index,plot=plot)

    # window for phase spectrum?
    if window == "hanning":
        if plot:
            plt.figure()
            plt.title(window +" window applied")
            plt.plot(xsym,ysym,label = 'data')
            plt.plot(xsym,np.hanning(len(ysym))*np.max(np.abs(ysym)),label = 'scaled window')
            plt.plot(xsym,ysym*np.hanning(len(ysym)),label ='windowed data')
            plt.legend()
            plt.xlabel("Path length (cm)")
            plt.ylabel("Power")
            plt.show()
        ysym = scipy.fftpack.ifftshift(ysym*np.hanning(len(ysym))) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
    else:
        ysym = scipy.fftpack.ifftshift(ysym)
        
    xsym = scipy.fftpack.ifftshift(xsym) # packing is now 0,1,2,...N/2,N/2-1,N/2-2,...1
    if plot:
        plt.figure()
        plt.title("Proper indexing for FFT\nWhite light fringe at 0\nPoint to left of white light fringe at index N")
        plt.plot(ysym)
        plt.ylabel("Power")
        plt.xlabel("Index")
        plt.show()
    fsym,Ssym = FFTandFrequencies(xsym,ysym,plot=plot,units = units)
    theta=np.angle(Ssym)

    # mirror long throw part of interferogram to make full long throw symmetric interferogram
    if plot:
        plt.figure(figsize = (12,6))
        plt.title("mirroring interferogram")
        plt.plot(x-ZPD,y,linewidth = 2,label = "full interferogram")
        plt.plot((x-ZPD)[0:2*ZPD_index+1],y[0:2*ZPD_index+1],linewidth = 2,label = "symmetric portion")
        plt.plot(-x[2*ZPD_index+1:][::-1]+ZPD,y[2*ZPD_index+1:][::-1],linewidth = 2,label = "mirrored portion")
        
    # force a symmetric IFG from the single sided IFG by mirroring the IFG
    y=np.concatenate((y[2*ZPD_index+1:][::-1],y)) # just mirror the -delta portion not measured
    #y=np.concatenate((np.zeros(len(y[2*dex+1:])),y)) # zero pad version of mirroring
    x = np.concatenate((-x[2*ZPD_index+1:][::-1]+ZPD,x-ZPD))
    
    if plot:
         plt.plot(x,y,color= "k",linewidth=0.5,label = "concatenated")
         plt.legend()
         plt.show()


    # window?
    if window == "hanning":
        if plot:
            plt.figure()
            plt.title(window +" window applied")
            plt.plot(x,y,label = 'data')
            plt.plot(x,np.hanning(len(y))*np.max(np.abs(y)),label = 'scaled window')
            plt.plot(x,y*np.hanning(len(y)),label ='windowed data')
            plt.legend()
            plt.ylabel("Power")
            plt.xlabel("Path length (cm)")
            plt.show()
        y=scipy.fftpack.ifftshift(np.hanning(len(y))*y)
    else:
        y=scipy.fftpack.ifftshift(y)

    # shift interferogram the way fft likes it
    x = scipy.fftpack.ifftshift(x)
    

    if plot: #check that white light fringe is at index 0
        plt.figure()
        plt.title("Proper indexing for FFT\nWhite light fringe at 0\nPoint to left of white light fringe at index N")
        plt.plot(y)
        plt.ylabel("Power")
        plt.xlabel("Index")
        plt.show()


    # fft before phase correction mostly to get frequencies for that we will get later
    f,S = FFTandFrequencies(x,y,plot=plot,units = units) # FFT of zeropadded data


    # need to interpolate phase information to higher resolution for full interferogram phase correction
    
    interp_func = scipy.interpolate.interp1d(scipy.fftpack.fftshift(fsym),
                                                 scipy.fftpack.fftshift(theta),
                                                 kind = "linear",
                                                 bounds_error = False,
                                                 fill_value = 0)
    
    theta_highres = scipy.fftpack.fftshift(interp_func(scipy.fftpack.fftshift(f))) #bad behavior when changing to +pi to -pi
    # interpolate between points
    theta_highres_2 = interp_angle(f,fsym,theta,debug = False) # interp_angle(scipy.fftpack.fftshift(f),scipy.fftpack.fftshift(fsym),theta,debug = False)
    # interpolate by fitting polynomial
    if theta_fit_method == "poly":
        fit_dict = fit_angle(fsym,theta,min_poly,max_poly,deg = theta_fit_degree,plot = True,units = units)
        theta_highres_3 = poly_mod_2pi(f,fit_dict['fit'][0])
    
    if plot:
        plt.figure(figsize = (12,6))
        plt.title("Fitting of phase, currently using method " + theta_fit_method)
        plt.plot(fsym*unit_scale,theta,"o",mec = "k",label = "theta from symmetric interferogram")
        plt.plot(f*unit_scale,theta_highres_2,".",label = "theta interpolated to higher resolution in complex plane")
        if theta_fit_method == "poly":
            plt.plot(np.sort(f*unit_scale),theta_highres_3[np.argsort(f)],label = "theta fitted polynomial")
        plt.ylim(-5,5)
        plt.xlabel(units)
        plt.legend()
        plt.show()
    
    # do phase correction
    if algorithm=='Mertz': # phase correct in frequency spectrum space
        if theta_fit_method == "poly":
            S_corr = S*np.exp(-1j*theta_highres_3)
        else:
            S_corr = S*np.exp(-1j*theta_highres_2)
        #B=X-Y
        B = np.real(S_corr)
        B2 = np.imag(S_corr)
        #B2 = X2-Y2
        if plot:
            plt.figure(figsize = (12,6))
            plt.title("Phase corrected spectrum using Mertz method with interpolation "+theta_fit_method)
            plt.plot(f*unit_scale,np.abs(S),label = "ABS",color = 'k', linewidth = 3)
            plt.plot(f*unit_scale,B,label = "Signal")
            plt.plot(f*unit_scale,B2,label = "Noise")
            plt.plot(f*unit_scale,np.sqrt(B2**2+B**2),label = "signal + noise",linewidth = 0.5)
            plt.legend()
            plt.xlabel(units)
            plt.show()
        
    elif algorithm=='Richards': #phase correct in interferogram space
        print('Using Richards method to correct phase in interferograms then FFT')
        


        if min_filter: # since you are Fourier transforming to correct phase you might as well do a frequency filter
            if max_filter:
                boxcar = np.zeros(len(y))
                boxcar[(f*unit_scale>min_filter) & (f*unit_scale<max_filter)] = 1
                boxcar[(f*unit_scale<-1*min_filter) & (f*unit_scale>-1*max_filter)] = 1
                if plot:
                    plt.figure(figsize = (12,6))
                    plt.title("Frequency Filter Applied")
                    plt.plot(np.sort(f*unit_scale),boxcar[np.argsort(f)])
                    plt.xlabel(units)
                    plt.show()
            else:
                print("please specify both min_filter and max_filter")
        else:
           boxcar = np.ones(len(y))  # do nothing
         
        if theta_fit_method == "poly": # ask about the phase_ifft
            phase_ifft = np.fft.ifft(np.exp(-1j*theta_highres_3)*boxcar) #poly fit
        else:
            phase_ifft = np.fft.ifft(np.exp(-1j*theta_highres_2)*boxcar) # interpolate between points


        N=len(phase_ifft)
        
        phase_ifft_sym = scipy.fftpack.fftshift(phase_ifft)


        # convolve in interferogram space and trim extra stuff from convolution
        y_corr = np.real(signal.convolve(scipy.fftpack.fftshift(y),scipy.fftpack.fftshift(phase_ifft)))[N//2:3*N//2]
        
        if window == "hanning":
            y_corr = y_corr*np.hanning(len(y_corr)) #appodize should be made into option
        else:
            y_corr = y_corr
        
        if plot:
            plt.figure(figsize = (12,6))
            plt.subplot(211)
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,scipy.fftpack.fftshift(y),label = "raw interferogram")
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,y_corr, label = "phase corrected/filtered interferogram")
            plt.xlabel("Path length (cm)")
            plt.legend()
            plt.subplot(212)
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,scipy.fftpack.fftshift(y),label = "raw interferogram")
            plt.plot(np.arange(-N//2+1,N//2+1)*samp_int,y_corr, label = "phase corrected/filtered interferogram")
            plt.xlabel("Path length (cm)")
            plt.xlim(-samp_int*100,samp_int*100)
            plt.legend()
            plt.show()

        # Do the Fourier Transform    
        f,S_corr = FFTandFrequencies(x,scipy.fftpack.ifftshift(y_corr),plot= plot,units = units)
    
    return f,S_corr

    
def NotchFrequencies(x,y,v,freqs=[60.0,120.0,180.0,240.0,300.0,420.0,480.0,540.0],df=.1,plot=False):
    ''' remove power at discrete frequencies needs to be added in '''
    t = x/v # timestream
    samp_int=t[1]-t[0]
    y=y-y.mean()
    f,ffty = FFTandFrequencies(t,y,plot=False)
    if plot:
        plt.plot(f,abs(ffty))
    
    for i in freqs:
        js1 = f[(f>(i-df/2.0)) & (f<(i+df/2.0))] # positive frequencies
        js2 = f[(f<-1*(i-df/2.0)) & (f>-1*(i+df/2.0))] # positive frequencies
        js=np.concatenate((js1,js2))
        for j in js:
            ffty[list(f).index(j)]=0
            
    if plot:
        plt.plot(f,abs(ffty))
        plt.xlabel('Frequency (Hz)')
        plt.show()
        
    y = scipy.fftpack.ifft(ffty)
    return y



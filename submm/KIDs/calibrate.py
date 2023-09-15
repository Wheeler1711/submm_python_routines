import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import interpolate

# this script is for calibrating a kinetic inductance detector to convert
# changes in i and q to the shift of the resonator

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def plot_data_circle(x,y, xc, yc, R):
    theta_fit = np.linspace(-np.pi, np.pi, 180)
    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=1)

def fit_cable_delay(gain_f,gain_phase,plot = False):
    '''
    fitting is complicated when phase wraps from
    positive pi to negative pi
    so we apply a phase offset ot get the data to be centered at 0 phase
    first we move from -pi to pi to 0 to 2pi so we can use a modulus function
    then we shift the phase so that the center point in on pi
    gain f in Hz
    '''
    if gain_f[0]<10**6:
        print("WARNING - It looks like f is not in Hz please check")
    gain_phase = np.mod(gain_phase+np.pi-((gain_phase[len(gain_phase)//2]+np.pi)-np.pi),2*np.pi)
    #shift back to -pi to pi space for fun
    gain_phase = gain_phase -np.pi
    p_phase = np.polyfit(gain_f,gain_phase,1)
    tau = p_phase[0]/(2.*np.pi)
    poly_func_phase = np.poly1d(p_phase)
    fit_data_phase = poly_func_phase(gain_f)
    if plot:
        plt.figure()
        plt.plot(gain_f, gain_phase, 'o',label = "data")
        plt.plot(gain_f, fit_data_phase,label = "fit")

    return tau,fit_data_phase,gain_phase

def fit_cable_delay_from_slope(f,phase,plot = True):
    '''
    fitting is complicated when phase wraps from
    positive pi to negative pi
    so we just look at median slope for adjacent frequency points
    gain f in Hz
    '''
    if f[0]<10**6:
        print("WARNING - It looks like f is not in Hz please check")
    phase_gradient = np.gradient(phase)/np.gradient(f)/2/np.pi
    tau = np.median(phase_gradient)
    if plot:
        plt.plot(f/10**6,phase_gradient,label = "data")
        plt.plot(f/10**6,tau*np.ones(len(f)),label = "fit")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Phase gradient")
        plt.legend()
        plt.title("Tau = "+str(tau*10**9)+" ns")
        plt.show()
    return tau,phase_gradient
        
def remove_cable_delay(f,z,tau):
    z_corr = z*np.exp(2j*np.pi*tau*f)
    return z_corr

def fft_noise(z_stream,df_over_f,sample_rate):
    npts_fft = int(2**(np.floor(np.log2(df_over_f.size)))) 
    Sxx = 2*fftpack.fft(df_over_f,n = npts_fft)*np.conj(fftpack.fft(df_over_f,n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    S_per = 2*fftpack.fft(np.real(z_stream),n = npts_fft)*np.conj(fftpack.fft(np.real(z_stream),n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    S_par = 2*fftpack.fft(np.imag(z_stream),n = npts_fft)*np.conj(fftpack.fft(np.imag(z_stream),n = npts_fft))/sample_rate*npts_fft/npts_fft**2
    fft_freqs = fftpack.fftfreq(npts_fft,1./sample_rate)
    return fft_freqs,Sxx,S_per,S_par

def interp_phase_to_df_over_f(phase_fine,phase_stream,iq_sweep_freqs,extrap = "min_max"):
    #interp phase to frequency
    if extrap == "min_max":
        fill_value = (iq_sweep_freqs[-1],iq_sweep_freqs[0])
    else:
        fill_value = "extrapolate"
        
    f_interp = interpolate.interp1d(phase_fine, iq_sweep_freqs,kind = 'quadratic',
                                    bounds_error = False,fill_value = fill_value)
    freqs_stream = f_interp(phase_stream)
    mean_freq = np.mean(freqs_stream)
    stream_df_over_f = freqs_stream/mean_freq-1.
    return stream_df_over_f,f_interp

import numpy as np
from numba import jit

from submm.KIDs.res.utils import cardan


# at import time, create a dictionary of all the fitting functions in this module
fitting_functions = {}


def fit_func(func):
    """Decorator to add a function to the fitting_functions dictionary"""
    name_type = func.__name__
    fitting_functions[name_type] = func
    return func


@fit_func
@jit(nopython=True)
def nonlinear_mag(f_hz, fr, Qr, amp, phi, a, b0, b1, flin):
    """
    function to describe the magnitude S21 of a non-linear resonator

    This is based of fitting code from MUSIC
    The idea is we are producing a model that is described by the equation below
    the first two terms in the large parenthesis and all other terms are familiar to me,
    but I am not sure where the last term comes from though it does seem to be important for fitting

                              /        (j phi)            (j phi)    |  2
    |S21|^2 = (b0+b1 x_lin)* |1 -amp*e^           +amp*(e^       -1) |^
                             |   ------------      ----              |
                             \     (1+ 2jy)         2               /

        where the nonlinearity of y is described by the following equation taken from Response of superconducting
        micro-resonators with nonlinear kinetic inductance:
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr

    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers in Hertz
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    a : float
        The nonlinearity parameter, bifurcation occurs at a = 0.77
    b0 : float
        DC level of s21 away from resonator
    b1 : float
        Frequency dependant gain variation
    flin : float
        This is probably the frequency of the resonator when a = 0

    Returns
    -------
    ResFit
    """
    xlin = (f_hz - flin) / flin
    xg = (f_hz - fr) / fr
    yg = Qr * xg
    y = np.zeros(f_hz.shape[0])
    # find the roots of the y equation above
    for i in range(0, f_hz.shape[0]):
        """
        4y^3+ -4yg*y^2+ y -(yg+a)
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        roots = cardan(4.0,-4.0*yg[i],1.0,-(yg[i]+a))
        print(roots)
        # more accurate version that doesn't seem to change the fit at all
        # only cares about real roots
        roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))       
        where_real = np.where(np.imag(roots) == 0)
        # analytic version has some floating point error accumulation
        where_real = np.where(np.abs(np.imag(roots)) < 1e-10) 
        """
        # np.max(np.real(roots[where_real]))
        y[i] = cardan(4.0, -4.0 * yg[i], 1.0, -(yg[i] + a))
    abs_val = np.abs(1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * y) + amp / 2. * (np.exp(1.0j * phi) - 1.0))
    z = (b0 + b1 * xlin) * abs_val ** 2
    return z


def nonlinear_mag_for_plot(f_hz, fr, Qr, amp, phi, a, b0, b1, flin):
    """
    The square root of the above function, nonlinear_mag(), needed for visualization and plotting
    function to describe the magnitude S21 of a non-linear resonator

    This is based of fitting code from MUSIC
    The idea is we are producing a model that is described by the equation below
    the first two terms in the large parenthesis and all other terms are familiar to me,
    but I am not sure where the last term comes from though it does seem to be important for fitting


    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers in Hertz
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    a : float
        The nonlinearity parameter, bifurcation occurs at a = 0.77
    b0 : float
        DC level of s21 away from resonator
    b1 : float
        Frequency dependant gain variation
    flin : float
        This is probably the frequency of the resonator when a = 0

    Returns
    -------
    ResFit
    """
    return np.sqrt(nonlinear_mag(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, b0=b0, b1=b1, flin=flin))


@fit_func
def linear_mag(f_hz, fr, Qr, amp, phi, b0):
    """
    This is based of fitting code from MUSIC
    The idea is we are producing a model that is described by the equation below
    the first two terms in the large parenthesis and all other terms are familiar to me,
    but I am not sure where the last term comes from though it does seem to be important for fitting

                     /        (j phi)            (j phi)    |  2
    |S21|^2 = (b0)* |1 -amp*e^           +amp*(e^       -1) |^
                    |   ------------      ----              |
                    \     (1+ 2jxg)         2              /

        no y just xg, with no non-linear kinetic inductance

    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    b0 : float
        DC level of s21 away from resonator

    """
    if not np.isscalar(fr):  # vectorize breaks numba though
        f_hz = np.reshape(f_hz, (f_hz.shape[0], 1, 1, 1, 1, 1))
    xg = (f_hz - fr) / fr
    z = (b0) * np.abs(
        1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * xg * Qr) + amp / 2. * (np.exp(1.0j * phi) - 1.0)) ** 2
    return z


def linear_mag_for_plot(f_hz, fr, Qr, amp, phi, b0):
    """
    The square root of the above function, linear_mag(), need for visualization and plotting
    This is based of fitting code from MUSIC

    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    b0 : float
        DC level of s21 away from resonator

    """
    return np.sqrt(linear_mag(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, b0=b0))


@fit_func
@jit(nopython=True)
def nonlinear_iq(f_hz, fr, Qr, amp, phi, a, i0, q0, tau, f0):
    """
    To describe the I-Q loop of a nonlinear resonator

    This is based of fitting code from MUSIC

    The idea is we are producing a model that is described by the equation below
    the first two terms in the big parenthesis and all other terms are familiar to me,
    but I am not sure where the last term comes from though it does seem to be important for fitting

                       (-j 2 pi deltaf tau)  /        (j phi)            (j phi)   |
           (i0+j*q0)*e^                    *|1 -amp*e^           +amp*(e^       -1) |
                                            |   ------------      ----              |
                                             \     (1+ 2jy)         2              /

        where the nonlinearity of y is described by the following equation taken from Response of superconducting
        micro resonators with nonlinear kinetic inductance:
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr

    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    a : float
        The nonlinearity parameter, bifurcation occurs at a = 0.77
    i0 : float
    q0 : float
        these are constants that describes an overall phase rotation of the iq loop + a DC gain offset
    tau : float
        The cable delay
    f0 : float
        The center frequency, not sure why we include this as a secondary parameter should be the same as fr
    """
    deltaf = (f_hz - f0)
    xg = (f_hz - fr) / fr
    yg = Qr * xg
    y = np.zeros(f_hz.shape[0])
    # find the roots of the y equation above
    for i in range(0, f_hz.shape[0]):
        """
        4y^3+ -4yg*y^2+ y -(yg+a)
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        # more accurate version that doesn't seem to change the fit at al     
        roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))
        only care about real roots
        where_real = np.where(np.imag(roots) == 0)
        y[i] = np.max(np.real(roots[where_real]))
        """
        y[i] = cardan(4.0, -4.0 * yg[i], 1.0, -(yg[i] + a))
    big_parenthesis = (1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * y) + amp / 2. * (np.exp(1.0j * phi) - 1.0))
    z = (i0 + 1.j * q0) * np.exp(-1.0j * 2 * np.pi * deltaf * tau) * big_parenthesis
    return z


@fit_func
@jit(nopython=True)
def nonlinear_iq_for_fitter(f_hz, fr, Qr, amp, phi, a, i0, q0, tau, f0):
    """
    when using a fitter that can't handel complex number
    one needs to return both the real and imaginary components separately
    """

    deltaf = (f_hz - f0)
    xg = (f_hz - fr) / fr
    yg = Qr * xg
    y = np.zeros(f_hz.shape[0])

    for i in range(0, f_hz.shape[0]):
        """
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        where_real = np.where(np.imag(roots) == 0)
        y[i] = np.max(np.real(roots[where_real]))
        """
        y[i] = cardan(4.0, -4.0 * yg[i], 1.0, -(yg[i] + a))
    z = (i0 + 1.j * q0) * np.exp(-1.0j * 2 * np.pi * deltaf * tau) * (
            1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * y) + amp / 2. * (np.exp(1.0j * phi) - 1.0))
    real_z = np.real(z)
    imag_z = np.imag(z)
    return np.hstack((real_z, imag_z))

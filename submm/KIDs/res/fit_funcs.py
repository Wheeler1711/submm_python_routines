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


"""
Simons Observatory fitting functions.
Copied from https://github.com/simonsobs/sodetlib/blob/master/sodetlib/resonator_fitting.py

See LICENSE file at: https://github.com/simonsobs/sodetlib/blob/master/LICENSE
Copyright (c) 2019, Simons Observatory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


@fit_func
@jit(nopython=True)
def so_linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    """
    Function for a resonator with asymmetry parameterized by the imaginary
    part of ``Q_e``. The real part of ``Q_e`` is what we typically refer to as
    the coupled Q, ``Q_c``.
    """
    Q_e = Q_e_real + 1j * Q_e_imag
    return 1 - (Q * Q_e ** (-1) / (1 + 2j * Q * (f - f_0) / f_0))


@fit_func
@jit(nopython=True)
def so_cable_delay(f, delay, phi, f_min):
    """
    Function implements a time delay (phase variation linear with frequency).
    """
    return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))


@fit_func
@jit(nopython=True)
def so_general_cable(f, delay, phi, f_min, A_mag, A_slope):
    """
    Function implements a time delay (phase variation linear with frequency) and
    attenuation slope characterizing a background RF cable transfer function.
    """
    phase_term = so_cable_delay(f, delay, phi, f_min)
    magnitude_term = ((f - f_min) * A_slope + 1) * A_mag
    return magnitude_term * phase_term


@fit_func
@jit(nopython=True)
def so_resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope):
    """
    Function that includes asymmetric resonator (``linear_resonator``) and cable
    transfer functions (``general_cable``). Which most closely matches our full
    measured transfer function.
    """
    resonator_term = so_linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag)
    cable_term = so_general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return resonator_term * cable_term


@fit_func
@jit(nopython=True)
def so_resonator_cable_for_fitter(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope):
    """
    Function that includes asymmetric resonator (``linear_resonator``) and cable
    transfer functions (``general_cable``). Which most closely matches our full
    measured transfer function.
    """
    z = so_resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope)
    real_z = np.real(z)
    imag_z = np.imag(z)
    return np.hstack((real_z, imag_z))

"""
Fitting functions for KIDs corrected to match Khalil+12.
This removes the extra term from nonlinear_iq that is not physical, and corrects
The definition of Qc to be given by 1 / Qc = Re(1 / Qe), rather than Qc = abs(Qe).
See Khalil 2012 for an explanation of why this definiion of Qc should be used:
https://doi.org/10.1063/1.3692073
See Seth Siegel's thesis (2016) for a complete derivaiton of the fitting equation,
including nonlinearity: https://thesis.library.caltech.edu/9238/

Adapted from the Jordan's code by Joanna Perido and Logan Foote, Fall 2022
"""
@fit_func
@jit(nopython=True)
def nonlinear_iq_ss(f, fr, Qr, amp, phi, a, i0, q0, tau):
    """
    To describe the I-Q loop of a nonlinear resonator
    The resonance equation is fully derived in Seth Siegel's 2016 thesis. See
    Khalil+12 for an explanation of the Qc calculation (the cos(phi) term)


                        (-j 2 pi f tau)    /                           (j phi)   \
            (i0+j*q0)*e^                * |1 -        Qr             e^           |
                                          |     --------------  X  ------------   |
                                           \     Qc * cos(phi)       (1+ 2jy)    /

        where the nonlinearity of y is described by
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr

    Parameters
    ----------
    f : numpy.array
        The frequencies in your iq sweep covers
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Qr / Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator
        and the readout system
    a : float
        The nonlinearity parameter, bifurcation occurs at a = 0.77
    i0 : float
    q0 : float
        these are constants that describes an overall phase rotation of the iq
        loop + a DC gain offset
    tau : float
        The cable delay
    """
    deltaf = f - fr
    fg = deltaf / fr
    yg = Qr * fg
    y = np.zeros(f.shape[0])
    #find the roots of the y equation above
    for i in range(0, f.shape[0]):
        y[i] = cardan(4.0, -4.0*yg[i], 1.0, -(yg[i]+a))
    Q_term = amp / np.cos(phi)
    s21_readout = (i0 + 1.j * q0) * np.exp(-2.0j * np.pi * deltaf * tau)
    z = s21_readout * (1.0 - Q_term * np.exp(1.0j * phi)/ (1.0 + 2.0j * y))
    return z

@fit_func
@jit(nopython=True)
def nonlinear_iq_ss_for_fitter(f, fr, Qr, amp, phi, a, i0, q0, tau):
    """
    when using a fitter that can't handel complex number
    one needs to return both the real and imaginary components separately
    """
    z = nonlinear_iq_ss(f, fr, Qr, amp, phi, a, i0, q0, tau)
    return np.hstack((np.real(z), np.imag(z)))

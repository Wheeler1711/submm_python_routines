"""
numba seems to make the fitting 10 times faster

module for fitting resonances curves for kinetic inductance detectors.
written by Jordan Wheeler 12/21/16

for example see res_fit.ipynb in this demos directory

To Do
I think the error analysis on the fit_nonlinear_iq_with_err probably needs some work
add in step by step fitting i.e. first amplitude normalization, then cable delay, then i0,q0 subtraction,
     then phase rotation, then the rest of the fit.
need to have fit option that just specifies tau because that never really changes for your cryostat

Change log
JDW 2017-08-17 added in a keyword/function to allow for gain variation "amp_var" to be taken out before fitting
JDW 2017-08-30 added in fitting for magnitude fitting of resonators i.e. not in iq space
JDW 2018-03-05 added more clever function for guessing x0 for fits
JDW 2018-08-23 added more clever guessing for resonators with large phi into guess separate functions
CHW 2022-09-02 PEP-8 formatting, spelling corrections, and minor code cleanup
"""
import inspect

import numpy as np
import scipy.optimize as optimization
import scipy.stats as stats

from submm.KIDs.res.fit_funcs import linear_mag, nonlinear_mag, nonlinear_iq, nonlinear_iq_for_fitter, \
    nonlinear_mag_for_plot, linear_mag_for_plot
from submm.KIDs.res.data_io import Fit, ResSet, NonlinearIQRes, NonlinearMagRes, LinearMagRes
from submm.KIDs.res.utils import amplitude_normalization, calc_qc_qi, guess_x0_iq_nonlinear, guess_x0_mag_nonlinear, \
    guess_x0_iq_nonlinear_sep, guess_x0_mag_nonlinear_sep


def bounds_check(x0, bounds):
    lower_bounds = []
    upper_bounds = []
    for x, lb, ub in zip(x0, bounds[0], bounds[1]):
        if x < lb:
            lower_bounds.append(x * 0.9)
        else:
            lower_bounds.append(lb)
        if x > ub:
            upper_bounds.append(x * 1.1)
        else:
            upper_bounds.append(ub)
    return lower_bounds, upper_bounds


def chi_squared(z, z_fit):
    real_fit = np.real(z_fit)
    imag_fit = np.imag(z_fit)
    real_meas = np.real(z)
    imag_meas = np.imag(z)
    obs_delta = np.sqrt((real_fit - real_meas) ** 2.0 + (imag_fit - imag_meas) ** 2.0)
    chi_sq, p_value = stats.chisquare(f_obs=obs_delta)
    return chi_sq, p_value


def fit_nonlinear_iq(f_hz, z, bounds=None, x0: list = None, fr_guess: float = None, tau=None, tau_guess=None,
                     amp_norm: bool = False, verbose: bool = True):
    """Fit a nonlinear IQ with from an S21 sweep.

    Parameters
    ----------
    f_hz : numpy.array
        frequencies Hz
    z : numpy.array
        complex s21
    bounds : tuple, option (default None)
        A 2d tuple of low values bounds[0] the high values bounds[1] to bound the fitting problem.
    x0 : list, optional (default None)
        The initial guesses for all parameters:
        fr_guess  = x0[0]
        Qr_guess  = x0[1]
        amp_guess = x0[2]
        phi_guess = x0[3]
        a_guess   = x0[4]
        i0_guess  = x0[5]
        q0_guess  = x0[6]
        tau_guess = x0[7]
        f0_guess  = x0[8]
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    fr_guess : float, optional (default None)
        The center resonator frequency in Hz. If None, the center frequency is calculated from the data.
        This overrides the value (x0[0]) specified in the x0 parameter list.
    tau : float, optional (default None)
        If not None, this the fitter to use a fixed value for tau (phase delay), speeding up the calculation.
    tau_guess: float, optional (default None)
        Set the initial guess for tau without. This overrides the value (xo[7]) specified in the x0 parameter list.
    amp_norm: bool, optional (default False)
        When True, a normalization is preformed for the amplitude variable. This parameter is useful when the transfer
        function of the cryostat is not flat.
    verbose : bool, optional (default True)
        Uses the print function to display fit results when true, no prints to the console when false.

    Returns
    -------
    fit : Fit
        A Fit NamedTuple containing the fit results.
    """
    if bounds is None:
        # define default bounds
        if verbose:
            print("default bounds used")
        bounds = ([np.min(f_hz), 50, .01, -np.pi, 0, -np.inf, -np.inf, -1.0e-6, np.min(f_hz)],
                  [np.max(f_hz), 200000, 1, np.pi, 5, np.inf, np.inf, 1.0e-6, np.max(f_hz)])
    if x0 is None:
        # define default initial guess
        if verbose:
            print("default initial guess used")
        # fr_guess = x[np.argmin(np.abs(z))]
        # x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
        x0 = guess_x0_iq_nonlinear(f_hz, z)
        # print(x0)
    if fr_guess is not None:
        x0[0] = fr_guess
    if tau is None:
        use_given_tau = False
    else:
        use_given_tau = True
    if tau_guess is not None:
        x0[7] = tau_guess
    if amp_norm:
        z = amplitude_normalization(f_hz, z)
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    # map the initial guess to the standard data record
    guess = NonlinearIQRes(*x0)
    if verbose:
        guess.console(label='Guess', print_header=True)
    # bounds check
    bounds = bounds_check(x0, bounds)
    # fitter choices
    if use_given_tau:
        del bounds[0][7]
        del bounds[1][7]
        del x0[7]
        popt, pcov = optimization.curve_fit(
            lambda x_lamb, a, b, c, d, e, f, g, h: nonlinear_iq_for_fitter(x_lamb, a, b, c, d, e, f, g, tau, h), f_hz,
            z_stacked, x0, bounds=bounds)
        popt = np.insert(popt, 7, tau)
        # fill covariance matrix#
        cov = np.ones((pcov.shape[0] + 1, pcov.shape[1] + 1)) * -1
        cov[0:7, 0:7] = pcov[0:7, 0:7]
        cov[8, 8] = pcov[7, 7]
        cov[8, 0:7] = pcov[7, 0:7]
        cov[0:7, 8] = pcov[0:7, 7]
        pcov = cov
    else:
        popt, pcov = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, bounds=bounds)
    # human-readable results
    fr, Qr, amp, phi, a, i0, q0, tau, f0 = popt
    Qc, Qi = calc_qc_qi(qr=Qr, amp=amp)
    z_fit = nonlinear_iq(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0)
    chi_sq, p_value = chi_squared(z=z, z_fit=z_fit)
    result = NonlinearIQRes(fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0,
                            chi_sq=chi_sq, p_value=p_value, Qc=Qc, Qi=Qi)
    if verbose:
        result.console(label='Fit', print_header=True)
    # make a packaged result (NamedTuple) to return
    fit = Fit(origin=inspect.currentframe().f_code.co_name, func=nonlinear_iq,
              guess=guess, result=result, popt=popt, pcov=pcov, f_data=f_hz, z_data=z)
    return fit


def fit_nonlinear_iq_sep(fine_f_hz, fine_z, gain_f_hz, gain_z,
                         fine_z_err=None, gain_z_err=None, bounds=None, x0=None, amp_norm: bool = False,
                         verbose: bool = True):
    """Same as fit_nonlinear_iq() but takes fine and gain scans separately

    Parameters
    ----------
    fine_f_hz : numpy.array
        frequencies Hz
    fine_z : numpy.array
        complex s21
    gain_f_hz : numpy.array
        frequencies Hz
    gain_z : numpy.array
        complex s21
    fine_z_err : numpy.array, optional (default None)
        The error in the fine scan, expected to bve the same size as fine_z. If either fine_z_err or gain_z_err is None,
        the error is not used.
    gain_z_err : numpy.array, optional (default None)
        The error in the gain scan, expected to bve the same size as gain_z. If either fine_z_err or gain_z_err is None,
        the error is not used.
    bounds : tuple, option (default None)
        A 2d tuple of low values bounds[0] the high values bounds[1] to bound the fitting problem.
    x0 : list, optional (default None)
        The initial guesses for all parameters:
        fr_guess  = x0[0]
        Qr_guess  = x0[1]
        amp_guess = x0[2]
        phi_guess = x0[3]
        a_guess   = x0[4]
        i0_guess  = x0[5]
        q0_guess  = x0[6]
        tau_guess = x0[7]
        f0_guess  = x0[8]
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    amp_norm : bool, optional (default False)
        When True, a normalization is preformed for the amplitude variable. This parameter is useful when the transfer
        function of the cryostat is not flat.
    verbose : bool, optional (default True)
        Uses the print function to display fit results when true, no prints to the console when false.

    Returns
    -------
    fit : Fit
        A Fit NamedTuple containing the fit results.
    """
    if bounds is None:
        # define default bounds
        print("default bounds used")
        bounds = ([np.min(fine_f_hz), 500., .01, -np.pi, 0, -np.inf, -np.inf, 1 * 10 ** -9, np.min(fine_f_hz)],
                  [np.max(fine_f_hz), 1000000, 1, np.pi, 5, np.inf, np.inf, 1 * 10 ** -6, np.max(fine_f_hz)])
    if x0 is None:
        # define default initial guess
        print("default initial guess used")
        # fr_guess = x[np.argmin(np.abs(z))]
        # x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
        x0 = guess_x0_iq_nonlinear_sep(fine_f_hz, fine_z, gain_f_hz, gain_z)
    f_hz = np.hstack((fine_f_hz, gain_f_hz))
    z = np.hstack((fine_z, gain_z))
    if amp_norm:
        z = amplitude_normalization(f_hz, z)
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    # map the initial guess to the standard data record
    guess = NonlinearIQRes(*x0)
    if verbose:
        guess.console(label='Guess', print_header=True)
    # bounds check
    bounds = bounds_check(x0, bounds)
    # error
    if fine_z_err is not None and gain_z_err is not None:
        z_err = np.hstack((fine_z_err, gain_z_err))
        z_err_stacked = np.hstack((np.real(z_err), np.imag(z_err)))
        popt, pcov = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, sigma=z_err_stacked,
                                            bounds=bounds)
        fr, Qr, amp, phi, a, i0, q0, tau, f0 = popt
        fit_result = nonlinear_iq(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0)
        # only do it for fine data
        red_chi_sqr = np.sum((np.hstack((np.real(fine_z), np.imag(fine_z))) - np.hstack(
            (np.real(fit_result[0:len(fine_z)]), np.imag(fit_result[0:len(fine_z)])))) ** 2 / np.hstack(
            (np.real(fine_z_err), np.imag(fine_z_err))) ** 2) / (len(fine_z) * 2. - 8.)

    else:
        popt, pcov = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, bounds=bounds)
        fr, Qr, amp, phi, a, i0, q0, tau, f0 = popt
        red_chi_sqr = None

    Qc, Qi = calc_qc_qi(qr=Qr, amp=amp)
    z_fit = nonlinear_iq(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0)
    chi_sq, p_value = chi_squared(z=z, z_fit=z_fit)
    result = NonlinearIQRes(fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0,
                            red_chi_sqr=red_chi_sqr, chi_sq=chi_sq, p_value=p_value, Qc=Qc, Qi=Qi)
    if verbose:
        result.console(label='Fit', print_header=True)
    fit = Fit(origin=inspect.currentframe().f_code.co_name, func=nonlinear_iq,
              guess=guess, result=result, popt=popt, pcov=pcov, f_data=f_hz, z_data=z)
    return fit


def fit_nonlinear_iq_with_err(f_hz, z, bounds=None, x0=None, amp_norm: bool = False, verbose: bool = True):
    """Same as fit_nonlinear_iq(), but double fits so that it can get error and a proper covariance matrix out.

    Parameters
    ----------
    f_hz : numpy.array
        frequencies Hz
    z : numpy.array
        complex s21
    bounds : tuple, option (default None)
        A 2d tuple of low values bounds[0] the high values bounds[1] to bound the fitting problem.
    x0 : list, optional (default None)
        The initial guesses for all parameters:
        fr_guess  = x0[0]
        Qr_guess  = x0[1]
        amp_guess = x0[2]
        phi_guess = x0[3]
        a_guess   = x0[4]
        i0_guess  = x0[5]
        q0_guess  = x0[6]
        tau_guess = x0[7]
        f0_guess  = x0[8]
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    amp_norm : bool, optional (default False)
        When True, a normalization is preformed for the amplitude variable. This parameter is useful when the transfer
        function of the cryostat is not flat.
    verbose : bool, optional (default True)
        Uses the print function to display fit results when true, no prints to the console when false.

    Returns
    -------
    fit : Fit
        A Fit NamedTuple containing the fit results.
    """
    if bounds is None:
        # define default bounds
        print("default bounds used")
        bounds = ([np.min(f_hz), 2000, .01, -np.pi, 0, -5, -5, 1 * 10 ** -9, np.min(f_hz)],
                  [np.max(f_hz), 200000, 1, np.pi, 5, 5, 5, 1 * 10 ** -6, np.max(f_hz)])
    if x0 is None:
        # define default initial guess
        print("default initial guess used")
        x0 = guess_x0_iq_nonlinear(f_hz, z)
    if amp_norm:
        z = amplitude_normalization(f_hz, z)
    guess = NonlinearIQRes(*x0)
    if verbose:
        guess.console(label='Guess', print_header=True)
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    # bounds check
    bounds = bounds_check(x0, bounds)
    # fit
    popt_first, pcov_first = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, bounds=bounds)
    fr_first, Qr_first, amp_first, phi_first, a_first, i0_first, q0_first, tau_first, f0_first = popt_first

    fit_result_stacked = nonlinear_iq_for_fitter(f_hz=z, fr=fr_first, Qr=Qr_first, amp=amp_first, phi=phi_first,
                                                 a=a_first, i0=i0_first, q0=q0_first, tau=tau_first, f0=f0_first)
    # get error
    var = np.sum((z_stacked - fit_result_stacked) ** 2) / (z_stacked.shape[0] - 1)
    err = np.ones(z_stacked.shape[0]) * np.sqrt(var)
    # refit
    popt, pcov = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, err, bounds=bounds)
    fr, Qr, amp, phi, a, i0, q0, tau, f0 = popt
    # make a dictionary to return
    Qc, Qi = calc_qc_qi(qr=Qr, amp=amp)
    z_fit = nonlinear_iq(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0)
    chi_sq, p_value = chi_squared(z=z, z_fit=z_fit)
    result = NonlinearIQRes(fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0,
                            chi_sq=chi_sq, p_value=p_value, Qc=Qc, Qi=Qi)
    if verbose:
        result.console(label='Fit', print_header=True)
    fit = Fit(origin=inspect.currentframe().f_code.co_name, func=nonlinear_iq,
              guess=guess, result=result, popt=popt, pcov=pcov, f_data=f_hz, z_data=z)
    return fit


# function for fitting an iq sweep with the above equation
def fit_nonlinear_mag(f_hz, z, bounds=None, x0=None, verbose=True):
    """
    Parameters
    ----------
    f_hz : numpy.array
        frequencies Hz
    z : numpy.array
        complex s21
    bounds : tuple, option (default None)
        A 2d tuple of low values bounds[0] the high values bounds[1] to bound the fitting problem.
    x0 : list, optional (default None)
        The initial guesses for all parameters:
        fr_guess  = x0[0]
        Qr_guess  = x0[1]
        amp_guess = x0[2]
        phi_guess = x0[3]
        a_guess   = x0[4]
        b0_guess  = x0[5]
        b1_guess  = x0[6]
        flin_guess= x0[7]
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    verbose : bool, optional (default True)
        Uses the print function to display fit results when true, no prints to the console when false.

    Returns
    -------
    fit : Fit
        A Fit NamedTuple containing the fit results.
    """
    if bounds is None:
        print("default bounds used")
        bounds = ([np.min(f_hz), 100, .01, -np.pi, 0, -np.inf, -np.inf, np.min(f_hz)],
                  [np.max(f_hz), 200000, 1, np.pi, 5, np.inf, np.inf, np.max(f_hz)])
    if x0 is None:
        # define default initial guess
        print("default initial guess used")
        # x0 = [fr_guess,10000.,0.5,0,0,np.abs(z[0])**2,np.abs(z[0])**2,fr_guess]
        x0 = guess_x0_mag_nonlinear(f_hz, z, verbose=verbose)

    guess = NonlinearMagRes(*x0)
    if verbose:
        guess.console(label='Guess', print_header=True)
    # bounds check
    bounds = bounds_check(x0, bounds)
    # fit
    popt, pcov = optimization.curve_fit(nonlinear_mag, f_hz, np.abs(z) ** 2, x0, bounds=bounds)
    fr, Qr, amp, phi, a, b0, b1, flin = popt
    Qc, Qi = calc_qc_qi(qr=Qr, amp=amp)
    z_fit = nonlinear_mag_for_plot(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, b0=b0, b1=b1, flin=flin)
    chi_sq, p_value = chi_squared(z=z, z_fit=z_fit)
    result = NonlinearMagRes(fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, b0=b0, b1=b1, flin=flin,
                             chi_sq=chi_sq, p_value=p_value, Qc=Qc, Qi=Qi)
    if verbose:
        result.console(label='Fit', print_header=True)
    fit = Fit(origin=inspect.currentframe().f_code.co_name, func=nonlinear_mag_for_plot,
              guess=guess, result=result, popt=popt, pcov=pcov, f_data=f_hz, z_data=z)
    return fit


def fit_linear_mag(f_hz, z, bounds=None, x0=None, verbose=True):
    """
    Parameters
    ----------
    f_hz : numpy.array
        frequencies Hz
    z : numpy.array
        complex s21
    bounds : tuple, option (default None)
        A 2d tuple of low values bounds[0] the high values bounds[1] to bound the fitting problem.
    x0 : list, optional (default None)
        The initial guesses for all parameters:
        fr_guess  = x0[0]
        Qr_guess  = x0[1]
        amp_guess = x0[2]
        phi_guess = x0[3]
        b0_guess  = x0[4]
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    verbose : bool, optional (default True)
        Uses the print function to display fit results when true, no prints to the console when false.


    Returns
    -------
    fit : Fit
        A Fit NamedTuple containing the fit results.
    """
    if bounds is None:
        # define default bounds
        print("default bounds used")
        bounds = ([np.min(f_hz), 100, .01, -np.pi, -np.inf], [np.max(f_hz), 200000, 1, np.pi, np.inf])
    if x0 is None:
        # define default initial guess
        if verbose:
            print("default initial guess used")
        # x0 = [fr_guess,10000.,0.5,0,0,np.abs(z[0])**2,np.abs(z[0])**2,fr_guess]
        x0 = guess_x0_mag_nonlinear(f_hz, z, verbose=verbose)
        x0 = np.delete(x0, [4, 6, 7])
    guess = LinearMagRes(*x0)
    if verbose:
        guess.console(label='Guess', print_header=True)
    # bounds check
    bounds = bounds_check(x0, bounds)
    # fit
    popt, pcov = optimization.curve_fit(linear_mag, f_hz, np.abs(z) ** 2, x0, bounds=bounds)
    # human-readable results
    fr, Qr, amp, phi, b0 = popt
    Qc, Qi = calc_qc_qi(qr=Qr, amp=amp)
    z_fit = linear_mag_for_plot(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, b0=b0)
    chi_sq, p_value = chi_squared(z=z, z_fit=z_fit)
    result = LinearMagRes(fr=fr, Qr=Qr, amp=amp, phi=phi, b0=b0, chi_sq=chi_sq, p_value=p_value, Qc=Qc, Qi=Qi)
    if verbose:
        result.console(label='Fit', print_header=True)
    fit = Fit(origin=inspect.currentframe().f_code.co_name, func=linear_mag_for_plot,
              guess=guess, result=result, popt=popt, pcov=pcov, f_data=f_hz, z_data=z)
    return fit


def fit_nonlinear_mag_sep(fine_f_hz, fine_z, gain_f_hz, gain_z,
                          fine_z_err=None, gain_z_err=None, bounds=None, x0=None, verbose=True):
    """Same as, fit_nonlinear_mag(), above but fine and gain scans are provided separately.
    
    Parameters
    ----------
    fine_f_hz : numpy.array
        frequencies Hz
    fine_z : numpy.array
        complex s21
    gain_f_hz : numpy.array
        frequencies Hz
    gain_z : numpy.array
        complex s21
    fine_z_err : numpy.array, optional (default None)
        The error in the fine scan, expected to bve the same size as fine_z. If either fine_z_err or gain_z_err is None,
        the error is not used.
    gain_z_err : numpy.array, optional (default None)
        The error in the gain scan, expected to bve the same size as gain_z. If either fine_z_err or gain_z_err is None,
        the error is not used.
    bounds : tuple, option (default None)
        A 2d tuple of low values bounds[0] the high values bounds[1] to bound the fitting problem.
    x0 : list, optional (default None)
        The initial guesses for all parameters:
        fr_guess   = x0[0]
        Qr_guess   = x0[1]
        amp_guess  = x0[2]
        phi_guess  = x0[3]
        a_guess    = x0[4]
        b0_guess   = x0[5]
        b1_guess   = x0[6]
        flin_guess = x0[7]
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    verbose : bool, optional (default True)
        Uses the print function to display fit results when true, no prints to the console when false.

    Returns
    -------
    fit : Fit
        A Fit NamedTuple containing the fit results.
    """
    if bounds is None:
        # define default bounds
        print("default bounds used")
        bounds = ([np.min(fine_f_hz), 100, .01, -np.pi, 0, -np.inf, -np.inf, np.min(fine_f_hz)],
                  [np.max(fine_f_hz), 1000000, 100, np.pi, 5, np.inf, np.inf, np.max(fine_f_hz)])
    if x0 is None:
        # define default initial guess
        print("default initial guess used")
        x0 = guess_x0_mag_nonlinear_sep(fine_f_hz, fine_z, gain_f_hz, gain_z)

    guess = NonlinearMagRes(*x0)
    if verbose:
        guess.console(label='Guess', print_header=True)
    # stack the scans for curve_fit
    f_hz = np.hstack((fine_f_hz, gain_f_hz))
    z = np.hstack((fine_z, gain_z))
    # bounds check
    bounds = bounds_check(x0, bounds)
    # fit
    if fine_z_err is not None and gain_z_err is not None:
        z_err = np.hstack((fine_z_err, gain_z_err))
        # propagation of errors left out cross term
        z_err = np.sqrt(4 * np.real(z_err) ** 2 * np.real(z) ** 2 + 4 * np.imag(z_err) ** 2 * np.imag(z) ** 2)
        popt, pcov = optimization.curve_fit(nonlinear_mag, f_hz, np.abs(z) ** 2, x0, sigma=z_err, bounds=bounds)
        fr, Qr, amp, phi, a, b0, b1, flin = popt
        fit_result = nonlinear_mag(f_hz=f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, b0=b0, b1=b1, flin=flin)
        red_chi_sqr = np.sum((np.abs(fine_z) ** 2 - fit_result[0:len(fine_z)]) ** 2 / z_err[0:len(fine_z)] ** 2) / (
                len(fine_z) - 7.)
    else:
        popt, pcov = optimization.curve_fit(nonlinear_mag, f_hz, np.abs(z) ** 2, x0, bounds=bounds)
        fr, Qr, amp, phi, a, b0, b1, flin = popt
        red_chi_sqr = None
    z_fit = nonlinear_mag_for_plot(f_hz=fine_f_hz, fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, b0=b0, b1=b1, flin=flin)
    chi_sq, p_value = chi_squared(z=z, z_fit=z_fit)
    # human-readable results
    Qc, Qi = calc_qc_qi(qr=Qr, amp=amp)
    result = NonlinearMagRes(fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, b0=b0, b1=b1, flin=flin,
                             chi_sq=chi_sq, p_value=p_value, red_chi_sqr=red_chi_sqr,  Qc=Qc, Qi=Qi)
    if verbose:
        result.console(label='Fit', print_header=True)
    fit = Fit(origin=inspect.currentframe().f_code.co_name, func=nonlinear_mag_for_plot,
              guess=guess, result=result, popt=popt, pcov=pcov, f_data=f_hz, z_data=z)
    return fit


def fit_nonlinear_iq_multi(f_hz, z, center_freqs=None, tau: float = None, fit_overlap=0.5, verbose: bool = True):
    """
    wrapper for handling n resonator fits at once
    mostly just a for loop for fitting but also trys to fit in a way that
    better handles collisions by not fitting to close to other resonators
    f_hz and z should have shape n_iq_points x n_res points
    center_freqs can be specified if you are fitting n resonators but you
    know there are actually more the n resonators at the frequency locations
    in center_freqs. This is useful if you didn't collect data for all of the 
    resonators but don't want collisions to screw up your fitting.
    fit_overlap: default 0.5 will only use data to halfway between the resonator
    you are trying to fit and the nearest neighbor resonators if it is close by.
    returns a class
    """

    if center_freqs is None:
        center_freqs = f_hz[f_hz.shape[0] // 2, :]
    res_fits = []
    for i in range(0, f_hz.shape[1]):
        f_single = f_hz[:, i]
        z_single = z[:, i]
        # flag data that is too close to other resonators
        if center_freqs is not None:
            center_index = np.argmin(np.abs(center_freqs - f_single[len(f_single) // 2]))
        else:
            center_index = i
        distance = center_freqs - center_freqs[center_index]
        if center_freqs[center_index] != np.min(center_freqs):  # don't do if lowest frequency resonator
            closest_lower_dist = -np.min(np.abs(distance[np.where(distance < 0)]))
            closest_lower_index = np.where(distance == closest_lower_dist)[0][0]
            halfway_low = center_freqs[center_index] - \
                          (center_freqs[center_index] - center_freqs[closest_lower_index]) * fit_overlap
        else:
            halfway_low = 0

        if center_freqs[center_index] != np.max(center_freqs):  # don't do if highest frequency resonator
            closest_higher_dist = np.min(np.abs(distance[np.where(distance > 0)]))
            closest_higher_index = np.where(distance == closest_higher_dist)[0][0]
            halfway_high = center_freqs[center_index] + \
                           (center_freqs[closest_higher_index] - center_freqs[center_index]) * fit_overlap
        else:
            halfway_high = np.inf

        use_index = np.where(((f_single > halfway_low) & (f_single < halfway_high)))
        mask = np.zeros(len(f_single))
        mask[use_index] = 1
        f_single_res = f_single[use_index]
        z_single_res = z_single[use_index]

        try:
            fit_single_res = fit_nonlinear_iq(f_single_res, z_single_res, tau=tau, verbose=verbose)
        except Exception as e:
            if verbose:
                print(e)
                print("failed to fit")
        else:
            # overwrite fit data with full data set
            fit_dict = fit_single_res._asdict()
            fit_dict['f_data'] = f_single
            fit_dict['z_data'] = z_single
            fit_dict['mask'] = mask
            fit = Fit(**fit_dict)
            res_fits.append(fit)
    res_set = ResSet(res_fits=res_fits, verbose=verbose)
    return res_set


def fit_linear_mag_multi(f_hz, z, verbose: bool = True):
    """
    wrapper for handling n resonator fits at once
    f_hz and z should have shape n_iq_points x n_res points
    return same thing as fitter but in arrays for all resonators
    """
    center_freqs = f_hz[f_hz.shape[0] // 2, :]
    res_fits = []
    for i in range(0, f_hz.shape[1]):
        f_single = f_hz[:, i]
        z_single = z[:, i]
        # flag data that is too close to other resonators              
        distance = center_freqs - center_freqs[i]
        if center_freqs[i] != np.min(center_freqs):  # don't do if lowest frequency resonator
            closest_lower_dist = -np.min(np.abs(distance[np.where(distance < 0)]))
            closest_lower_index = np.where(distance == closest_lower_dist)[0][0]
            halfway_low = (center_freqs[i] + center_freqs[closest_lower_index]) / 2.
        else:
            halfway_low = 0

        if center_freqs[i] != np.max(center_freqs):  # don't do if highest frequency
            closest_higher_dist = np.min(np.abs(distance[np.where(distance > 0)]))
            closest_higher_index = np.where(distance == closest_higher_dist)[0][0]
            halfway_high = (center_freqs[i] + center_freqs[closest_higher_index]) / 2.
        else:
            halfway_high = np.inf
        use_index = np.where(((f_single > halfway_low) & (f_single < halfway_high)))
        mask = np.zeros(len(f_single))
        mask[use_index] = 1
        f_single_res = f_single[use_index]
        z_single_res = z_single[use_index]
        try:
            fit_single_res = fit_linear_mag(f_single_res, z_single_res, verbose=verbose)
        except Exception as e:
            if verbose:
                print("problem")
                print(e)
                print("failed to fit")
        else:
            # overwrite fit data with full data set
            fit_dict = fit_single_res._asdict()
            fit_dict['f_data'] = f_single
            fit_dict['z_data'] = z_single
            fit_dict['mask'] = mask
            fit = Fit(**fit_dict)
            res_fits.append(fit)
    res_set = ResSet(res_fits=res_fits, verbose=verbose)
    return res_set


if __name__ == '__main__':
    # get demo data
    import os
    from scipy.io import loadmat
    from submm.sample_data.abs_paths import abs_path_sample_data

    data_path = os.path.join(abs_path_sample_data,
                             "Survey_Tbb20.000K_Tbath170mK_Pow-60dBm_array_temp_sweep_long.mat")
    sample_data = loadmat(data_path)

    freq_ghz = sample_data['f'][:, 0]
    freq_mhz = freq_ghz * 1.0e3
    freq_hz = freq_ghz * 1.0e9
    s21_complex = sample_data['z'][:, 0]
    s21_mag = 20 * np.log10(np.abs(s21_complex))

    freq_hz_res1 = freq_hz[21050: 21250]
    freq_mhz_res1 = freq_hz[21050: 21250]
    s21_complex_res1 = s21_complex[21050: 21250]
    s21_mag_res1 = s21_mag[21050: 21250]

    # # Caleb's testing area
    # Test the Res() instance for the fit result directly
    res_fit = fit_nonlinear_iq(f_hz=freq_hz_res1, z=s21_complex_res1)
    print(f'Res fit res_fit.result["fr"] from key = {res_fit.result["fr"]}')
    print(f'res_fit.result[0], from index = {res_fit.result[0]}')
    print(f'res_fit.result[0:2], from slice = {res_fit.result[0:2]}')
    print(f'res_fit.result.fr, from attribute = {res_fit.result.fr}\n')

    # test the Fit() instance for to get Res parameters
    print(f'Res fit fr (res_fit["fr"]) from key = {res_fit.result["fr"]}')
    print(f'Res fit Qc (res_fit["Qc"]) from key = {res_fit.result["Qc"]}\n')

    # test the console (print data to screen) output
    res_fit.console()

    # test hash ability of the result
    test_set = {res_fit.result}

    # test the plot
    # res_fit.plot()

    # test write, read, and iteration class
    res_set_test = ResSet(res_fits=[res_fit])
    res_set_test.write()
    res_set_read = ResSet(path=res_set_test.path)
    for read_result in res_set_read:
        read_result.console(label='Read', fields=['fr', 'tau', 'Qc'])
    # get array of the results for specific fields
    res_set_test('fr', 'amp', 'z_data')

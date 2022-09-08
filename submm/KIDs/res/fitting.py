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
import matplotlib.pyplot as plt

from submm.KIDs.res.fit_funcs import linear_mag, nonlinear_mag, nonlinear_iq, nonlinear_iq_for_fitter, \
    NonlinearIQRes, Fit
from submm.KIDs.res.utils import print_fit_string_nonlinear_iq, print_fit_string_nonlinear_mag, \
    print_fit_string_linear_mag, amplitude_normalization, guess_x0_iq_nonlinear, guess_x0_mag_nonlinear, \
    guess_x0_iq_nonlinear_sep, guess_x0_mag_nonlinear_sep


def brute_force_linear_mag_fit(f_hz, z, ranges, n_grid_points, error=None, plot=False):
    """
    Parameters
    ----------
    f_hz : numpy.array
        frequencies Hz
    z : numpy.array
        complex or abs of s21
    ranges : numpy.array
        The ranges for each parameter as in:
        np.asarray(([f_low,Qr_low,amp_low,phi_low,b0_low],[f_high,Qr_high,amp_high,phi_high,b0_high]))
    n_grid_points: int
        How finely to sample each parameter space. This can be very slow for n>10, an increase by a factor of 2 will
        take 2**5 times longer to marginalize over you must minimize over the unwanted axes of sum_dev
        i.e. for fr np.min(np.min(np.min(np.min(fit['sum_dev'],axis = 4),axis = 3),axis = 2),axis = 1)
    error : numpy.array, optional (default = None)
        The error on the complex or abs of s21, used for weighting squares
    plot : bool, optional (default = False)
        If true, will plot the fit and the data
    """
    if error is None:
        error = np.ones(len(f_hz))

    fs = np.linspace(ranges[0][0], ranges[1][0], n_grid_points)
    Qrs = np.linspace(ranges[0][1], ranges[1][1], n_grid_points)
    amps = np.linspace(ranges[0][2], ranges[1][2], n_grid_points)
    phis = np.linspace(ranges[0][3], ranges[1][3], n_grid_points)
    b0s = np.linspace(ranges[0][4], ranges[1][4], n_grid_points)
    evaluated_ranges = np.vstack((fs, Qrs, amps, phis, b0s))

    a, b, c, d, e = np.meshgrid(fs, Qrs, amps, phis, b0s, indexing="ij")  # always index ij

    evaluated = linear_mag(f_hz, a, b, c, d, e)
    data_values = np.reshape(np.abs(z) ** 2, (abs(z).shape[0], 1, 1, 1, 1, 1))
    error = np.reshape(error, (abs(z).shape[0], 1, 1, 1, 1, 1))
    sum_dev = np.sum(((np.sqrt(evaluated) - np.sqrt(data_values)) ** 2 / error ** 2),
                     axis=0)  # comparing in magnitude space rather than magnitude squared

    min_index = np.where(sum_dev == np.min(sum_dev))
    index1 = min_index[0][0]
    index2 = min_index[1][0]
    index3 = min_index[2][0]
    index4 = min_index[3][0]
    index5 = min_index[4][0]
    fit_values = np.asarray((fs[index1], Qrs[index2], amps[index3], phis[index4], b0s[index5]))
    fit_values_names = ('f0', 'Qr', 'amp', 'phi', 'b0')
    fit_result = linear_mag(f_hz, fs[index1], Qrs[index2], amps[index3], phis[index4], b0s[index5])

    marginalized_1d = np.zeros((5, n_grid_points))
    marginalized_1d[0, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=2), axis=1)
    marginalized_1d[1, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=2), axis=0)
    marginalized_1d[2, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=1), axis=0)
    marginalized_1d[3, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=2), axis=1), axis=0)
    marginalized_1d[4, :] = np.min(np.min(np.min(np.min(sum_dev, axis=3), axis=2), axis=1), axis=0)

    marginalized_2d = np.zeros((5, 5, n_grid_points, n_grid_points))
    # 0 _
    # 1 x _
    # 2 x x _
    # 3 x x x _ 
    # 4 x x x x _
    #  0 1 2 3 4
    marginalized_2d[0, 1, :] = marginalized_2d[1, 0, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=2)
    marginalized_2d[2, 0, :] = marginalized_2d[0, 2, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=1)
    marginalized_2d[2, 1, :] = marginalized_2d[1, 2, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=0)
    marginalized_2d[3, 0, :] = marginalized_2d[0, 3, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=2), axis=1)
    marginalized_2d[3, 1, :] = marginalized_2d[1, 3, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=2), axis=0)
    marginalized_2d[3, 2, :] = marginalized_2d[2, 3, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=1), axis=0)
    marginalized_2d[4, 0, :] = marginalized_2d[0, 4, :] = np.min(np.min(np.min(sum_dev, axis=3), axis=2), axis=1)
    marginalized_2d[4, 1, :] = marginalized_2d[1, 4, :] = np.min(np.min(np.min(sum_dev, axis=3), axis=2), axis=0)
    marginalized_2d[4, 2, :] = marginalized_2d[2, 4, :] = np.min(np.min(np.min(sum_dev, axis=3), axis=1), axis=0)
    marginalized_2d[4, 3, :] = marginalized_2d[3, 4, :] = np.min(np.min(np.min(sum_dev, axis=2), axis=1), axis=0)

    if plot:
        levels = [2.3, 4.61]  # delta chi squared two parameters 68 90 % confidence
        fig_fit = plt.figure(-1)
        axs = fig_fit.subplots(5, 5)
        for i in range(0, 5):  # y starting from top
            for j in range(0, 5):  # x starting from left
                if i > j:
                    # plt.subplot(5,5,i+1+5*j)
                    # axs[i, j].set_aspect('equal', 'box')
                    extent = [evaluated_ranges[j, 0], evaluated_ranges[j, n_grid_points - 1], evaluated_ranges[i, 0],
                              evaluated_ranges[i, n_grid_points - 1]]
                    axs[i, j].imshow(marginalized_2d[i, j, :] - np.min(sum_dev), extent=extent, origin='lower',
                                     cmap='jet')
                    axs[i, j].contour(evaluated_ranges[j], evaluated_ranges[i],
                                      marginalized_2d[i, j, :] - np.min(sum_dev), levels=levels, colors='white')
                    axs[i, j].set_ylim(evaluated_ranges[i, 0], evaluated_ranges[i, n_grid_points - 1])
                    axs[i, j].set_xlim(evaluated_ranges[j, 0], evaluated_ranges[j, n_grid_points - 1])
                    axs[i, j].set_aspect((evaluated_ranges[j, 0] - evaluated_ranges[j, n_grid_points - 1]) / (
                            evaluated_ranges[i, 0] - evaluated_ranges[i, n_grid_points - 1]))
                    if j == 0:
                        axs[i, j].set_ylabel(fit_values_names[i])
                    if i == 4:
                        axs[i, j].set_xlabel("\n" + fit_values_names[j])
                    if i < 4:
                        axs[i, j].get_xaxis().set_ticks([])
                    if j > 0:
                        axs[i, j].get_yaxis().set_ticks([])

                elif i < j:
                    fig_fit.delaxes(axs[i, j])

        for i in range(0, 5):
            # axes.subplot(5,5,i+1+5*i)
            axs[i, i].plot(evaluated_ranges[i, :], marginalized_1d[i, :] - np.min(sum_dev))
            axs[i, i].plot(evaluated_ranges[i, :], np.ones(len(evaluated_ranges[i, :])) * 1., color='k')
            axs[i, i].plot(evaluated_ranges[i, :], np.ones(len(evaluated_ranges[i, :])) * 2.7, color='k')
            axs[i, i].yaxis.set_label_position("right")
            axs[i, i].yaxis.tick_right()
            axs[i, i].xaxis.set_label_position("top")
            axs[i, i].xaxis.tick_top()
            axs[i, i].set_xlabel(fit_values_names[i])

        # axs[0,0].set_ylabel(fit_values_names[0])
        # axs[4,4].set_xlabel(fit_values_names[4])
        axs[4, 4].xaxis.set_label_position("bottom")
        axs[4, 4].xaxis.tick_bottom()

    # make a dictionary to return
    fit_dict = {'fit_values': fit_values, 'fit_values_names': fit_values_names, 'sum_dev': sum_dev,
                'fit_result': fit_result, 'marginalized_2d': marginalized_2d, 'marginalized_1d': marginalized_1d,
                'evaluated_ranges': evaluated_ranges}  # , 'x0':x0, 'z':z}
    return fit_dict


def fit_nonlinear_iq(f_hz, z, bounds=None, x0: list = None, fr_guess: float = None, tau=None, tau_guess=None,
                     amp_norm: bool = False, verbose: bool = True):
    """
    Fit a nonlinear IQ with from an S21 sweep.

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
    """

    if bounds is None:
        # define default bounds
        if verbose:
            print("default bounds used")
        bounds = ([np.min(f_hz), 50, .01, -np.pi, 0, -np.inf, -np.inf, 0, np.min(f_hz)],
                  [np.max(f_hz), 200000, 1, np.pi, 5, np.inf, np.inf, 1 * 10 ** -6, np.max(f_hz)])
    if x0 is None:
        # define default initial guess
        if verbose:
            print("default initial guess used")
        # fr_guess = x[np.argmin(np.abs(z))]
        # x0 = [fr_guess,10000.,0.5,0,0,np.mean(np.real(z)),np.mean(np.imag(z)),3*10**-7,fr_guess]
        x0 = guess_x0_iq_nonlinear(f_hz, z, verbose=verbose)
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
        # print(fit[1])
        x0 = np.insert(x0, 7, tau)

    else:
        popt, pcov = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, bounds=bounds)

    # mapp the initial guess to the standard data record
    guess = NonlinearIQRes(*x0)

    # human-readable results
    fr, Qr, amp, phi, a, i0, q0, tau, f0 = popt
    Qc = Qr / amp
    Qi = 1.0 / ((1.0 / Qr) - (1.0 / Qc))
    result = NonlinearIQRes(fr=fr, Qr=Qr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0, Qc=Qc, Qi=Qi)

    # fit_result = nonlinear_iq(f_hz=f_hz, Qr=Qr, fr=fr, amp=amp, phi=phi, a=a, i0=i0, q0=q0, tau=tau, f0=f0)
    # x0_result = nonlinear_iq(f_hz, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])

    if verbose:
        print_fit_string_nonlinear_iq(popt, print_header=False, label="Fit  ")

    # make a dictionary to return
    # fit_dict = {'fit': (popt, pcov), 'fit_result': fit_result, 'x0_result': x0_result, 'x0': x0, 'z': z,
    #             'fr': fr, 'Qr': Qr, 'amp': amp, 'phi': phi, 'a': a, 'i0': i0, 'q0': q0, 'tau': tau, 'Qi': Qi, 'Qc': Qc}
    fit = Fit(origin=inspect.currentframe().f_code.co_name, func=nonlinear_iq,
              guess=guess, result=result, pcov=pcov, f_data=f_hz, z_data=z)
    return fit


def fit_nonlinear_iq_sep(fine_f_hz, fine_z, gain_f_hz, gain_z,
                         fine_z_err=None, gain_z_err=None, bounds=None, x0=None, amp_norm: bool = False):
    """
    same as above function but takes fine and gain scans seporately

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
        # print(x0)
    # User the error?
    use_err = fine_z_err is not None and gain_z_err is not None

    f_hz = np.hstack((fine_f_hz, gain_f_hz))
    z = np.hstack((fine_z, gain_z))
    if use_err:
        z_err = np.hstack((fine_z_err, gain_z_err))

    if amp_norm:
        z = amplitude_normalization(f_hz, z)

    z_stacked = np.hstack((np.real(z), np.imag(z)))
    if use_err:
        z_err_stacked = np.hstack((np.real(z_err), np.imag(z_err)))
        fit = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, sigma=z_err_stacked, bounds=bounds)
    else:
        fit = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, bounds=bounds)


    # x0_result = nonlinear_iq(f_hz, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])

    if use_err:
        fit_result = nonlinear_iq(f_hz, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4], fit[0][5], fit[0][6],
                                  fit[0][7], fit[0][8])
        # only do it for fine data
        # red_chi_sqr = np.sum(z_stacked-np.hstack((np.real(fit_result),np.imag(fit_result))))**2/z_err_stacked**2)/(len(z_stacked)-8.)
        # only do it for fine data
        red_chi_sqr = np.sum((np.hstack((np.real(fine_z), np.imag(fine_z))) - np.hstack(
            (np.real(fit_result[0:len(fine_z)]), np.imag(fit_result[0:len(fine_z)])))) ** 2 / np.hstack(
            (np.real(fine_z_err), np.imag(fine_z_err))) ** 2) / (len(fine_z) * 2. - 8.)
        # make a dictionary to return
    else:
        # make a dictionary to return
        fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0': x0, 'z': z, 'fit_freqs': f_hz}

    return fit_dict


# same function but double fits so that it can get error and a proper covariance matrix out
def fit_nonlinear_iq_with_err(f_hz, z, bounds=None, x0=None, amp_norm: bool = False):
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
        i0_guess  = x0[5]
        q0_guess  = x0[6]
        tau_guess = x0[7]
        f0_guess  = x0[8]
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    amp_norm : bool, optional (default False)
        When True, a normalization is preformed for the amplitude variable. This parameter is useful when the transfer
        function of the cryostat is not flat.
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
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    fit = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, bounds=bounds)
    fit_result = nonlinear_iq(f_hz, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4], fit[0][5], fit[0][6], fit[0][7],
                              fit[0][8])
    fit_result_stacked = nonlinear_iq_for_fitter(f_hz, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4], fit[0][5],
                                                 fit[0][6], fit[0][7], fit[0][8])
    x0_result = nonlinear_iq(f_hz, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])
    # get error
    var = np.sum((z_stacked - fit_result_stacked) ** 2) / (z_stacked.shape[0] - 1)
    err = np.ones(z_stacked.shape[0]) * np.sqrt(var)
    # refit
    fit = optimization.curve_fit(nonlinear_iq_for_fitter, f_hz, z_stacked, x0, err, bounds=bounds)
    fit_result = nonlinear_iq(f_hz, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4], fit[0][5], fit[0][6], fit[0][7],
                              fit[0][8])
    x0_result = nonlinear_iq(f_hz, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8])

    # make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0': x0, 'z': z}
    return fit_dict


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
        The fit's initial guess can be very important because least squares fitting does not completely search the
        parameter space.
    verbose : bool, optional (default True)
        Uses the print function to display fit results when true, no prints to the console when false.
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

    fit = optimization.curve_fit(nonlinear_mag, f_hz, np.abs(z) ** 2, x0, bounds=bounds)
    fit_result = np.sqrt(
        nonlinear_mag(f_hz, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4], fit[0][5], fit[0][6], fit[0][7]))
    x0_result = np.sqrt(nonlinear_mag(f_hz, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7]))

    if verbose:
        print_fit_string_nonlinear_mag(fit[0], print_header=False, label="Fit  ")

    # human-readable results
    fr = fit[0][0]
    Qr = fit[0][1]
    amp = fit[0][2]
    phi = fit[0][3]
    a = fit[0][4]
    b0 = fit[0][5]
    b1 = fit[0][6]
    Qc = Qr / amp
    Qi = 1.0 / ((1.0 / Qr) - (1.0 / Qc))

    # make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0': x0, 'z': z,
                'fr': fr, 'Qr': Qr, 'amp': amp, 'phi': phi, 'a': a, 'b0': b0, 'b1': b1, 'Qi': Qi, 'Qc': Qc}
    return fit_dict


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
    """
    if bounds is None:
        # define default bounds
        print("default bounds used")
        bounds = ([np.min(f_hz), 100, .01, -np.pi, -np.inf], [np.max(f_hz), 200000, 1, np.pi, np.inf])
    if x0 is None:
        # define default initial guess
        print("default initial guess used")
        # x0 = [fr_guess,10000.,0.5,0,0,np.abs(z[0])**2,np.abs(z[0])**2,fr_guess]
        x0 = guess_x0_mag_nonlinear(f_hz, z, verbose=verbose)
        x0 = np.delete(x0, [4, 6, 7])

    fit = optimization.curve_fit(linear_mag, f_hz, np.abs(z) ** 2, x0, bounds=bounds)
    fit_result = np.sqrt(linear_mag(f_hz, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4]))
    x0_result = np.sqrt(linear_mag(f_hz, x0[0], x0[1], x0[2], x0[3], x0[4]))

    if verbose:
        print_fit_string_linear_mag(fit[0], print_header=False, label="Fit  ")

    # human-readable results
    fr = fit[0][0]
    Qr = fit[0][1]
    amp = fit[0][2]
    phi = fit[0][3]
    b0 = fit[0][4]
    Qc = Qr / amp
    Qi = 1.0 / ((1.0 / Qr) - (1.0 / Qc))

    # make a dictionary to return
    fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0': x0, 'z': z,
                'fr': fr, 'Qr': Qr, 'amp': amp, 'phi': phi, 'b0': b0, 'Qi': Qi, 'Qc': Qc}
    return fit_dict


def fit_nonlinear_mag_sep(fine_f_hz, fine_z, gain_f_hz, gain_z, fine_z_err=None, gain_z_err=None, bounds=None, x0=None,
                          verbose=True):
    """
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

    # same as above but fine and gain scans are provided seperatly
    # keywords are
    # bounds ---- which is a 2d tuple of low the high values to bound the problem by
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # amp_norm --- do a normalization for variable amplitude. usefull when tranfer function of the cryostat is not flat 
    """
    if bounds is None:
        # define default bounds
        print("default bounds used")
        bounds = ([np.min(fine_f_hz), 100, .01, -np.pi, 0, -np.inf, -np.inf, np.min(fine_f_hz)],
                  [np.max(fine_f_hz), 1000000, 100, np.pi, 5, np.inf, np.inf, np.max(fine_f_hz)])
    if x0 is None:
        # define default intial guess
        print("default initial guess used")
        x0 = guess_x0_mag_nonlinear_sep(fine_f_hz, fine_z, gain_f_hz, gain_z)
    # use error when fitting the data?
    use_err = fine_z_err is not None and gain_z_err is not None
    # stack the scans for curvefit
    f_hz = np.hstack((fine_f_hz, gain_f_hz))
    z = np.hstack((fine_z, gain_z))
    if use_err:
        z_err = np.hstack((fine_z_err, gain_z_err))
        # propagation of errors left out cross term
        z_err = np.sqrt(4 * np.real(z_err) ** 2 * np.real(z) ** 2 + 4 * np.imag(z_err) ** 2 * np.imag(z) ** 2)
        fit = optimization.curve_fit(nonlinear_mag, f_hz, np.abs(z) ** 2, x0, sigma=z_err, bounds=bounds)
    else:
        fit = optimization.curve_fit(nonlinear_mag, f_hz, np.abs(z) ** 2, x0, bounds=bounds)
    fit_result = nonlinear_mag(f_hz, fit[0][0], fit[0][1], fit[0][2], fit[0][3], fit[0][4], fit[0][5], fit[0][6],
                               fit[0][7])
    x0_result = nonlinear_mag(f_hz, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7])

    # compute reduced chi squared
    if verbose:
        print(f'len(z)=={len(z)}')
    if use_err:
        # red_chi_sqr = np.sum((np.abs(z)**2-fit_result)**2/z_err**2)/(len(z)-7.)
        # only use fine scan for reduced chi squared.
        red_chi_sqr = np.sum((np.abs(fine_z) ** 2 - fit_result[0:len(fine_z)]) ** 2 / z_err[0:len(fine_z)] ** 2) / (
                len(fine_z) - 7.)
        # make a dictionary to return
        fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0': x0, 'z': z, 'fit_freqs': f_hz,
                    'red_chi_sqr': red_chi_sqr}
    else:
        # make a dictionary to return
        fit_dict = {'fit': fit, 'fit_result': fit_result, 'x0_result': x0_result, 'x0': x0, 'z': z, 'fit_freqs': f_hz}
    return fit_dict


def fit_nonlinear_iq_multi(f_hz, z, tau=None):
    """
    wrapper for handling n resonator fits at once
    f_hz and z should have shape n_iq_points x n_res points
    return same thing as fitter but in arrays for all resonators
    """

    center_freqs = f_hz[f_hz.shape[0] // 2, :]

    all_fits = np.zeros((f_hz.shape[1], 9))
    all_fit_results = np.zeros((f_hz.shape[0], f_hz.shape[1]), dtype=np.complex_)
    all_x0_results = np.zeros((f_hz.shape[0], f_hz.shape[1]), dtype=np.complex_)
    all_masks = np.zeros((f_hz.shape[0], f_hz.shape[1]))
    all_x0 = np.zeros((f_hz.shape[1], 9))
    all_fr = np.zeros(f_hz.shape[1])
    all_Qr = np.zeros(f_hz.shape[1])
    all_amp = np.zeros(f_hz.shape[1])
    all_phi = np.zeros(f_hz.shape[1])
    all_a = np.zeros(f_hz.shape[1])
    all_i0 = np.zeros(f_hz.shape[1])
    all_q0 = np.zeros(f_hz.shape[1])
    all_tau = np.zeros(f_hz.shape[1])
    all_Qi = np.zeros(f_hz.shape[1])
    all_Qc = np.zeros(f_hz.shape[1])
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

        if center_freqs[i] != np.max(center_freqs):  # don't do if highest frequenct
            closest_higher_dist = np.min(np.abs(distance[np.where(distance > 0)]))
            closest_higher_index = np.where(distance == closest_higher_dist)[0][0]
            halfway_high = (center_freqs[i] + center_freqs[closest_higher_index]) / 2.
        else:
            halfway_high = np.inf

        use_index = np.where(((f_single > halfway_low) & (f_single < halfway_high)))
        mask = np.zeros(len(f_single))
        mask[use_index] = 1
        f_single = f_single[use_index]
        z_single = z_single[use_index]

        try:
            if tau is not None:
                fit_dict_iq = fit_nonlinear_iq(f_single, z_single, tau=tau)
            else:
                fit_dict_iq = fit_nonlinear_iq(f_single, z_single)

            all_fits[i, :] = fit_dict_iq['fit'][0]
            # all_fit_results[i,:] = fit_dict_iq['fit_result']
            # all_x0_results[i,:] = fit_dict_iq['x0_result']
            all_fit_results[:, i] = nonlinear_iq(f_hz[:, i], all_fits[i, 0], all_fits[i, 1], all_fits[i, 2],
                                                 all_fits[i, 3], all_fits[i, 4],
                                                 all_fits[i, 5], all_fits[i, 6], all_fits[i, 7], all_fits[i, 8])
            all_x0_results[:, i] = nonlinear_iq(f_hz[:, i], fit_dict_iq['x0'][0], fit_dict_iq['x0'][1],
                                                fit_dict_iq['x0'][2],
                                                fit_dict_iq['x0'][3], fit_dict_iq['x0'][4], fit_dict_iq['x0'][5],
                                                fit_dict_iq['x0'][6], fit_dict_iq['x0'][7], fit_dict_iq['x0'][8])
            all_masks[:, i] = mask
            all_x0[i, :] = fit_dict_iq['x0']
            all_fr[i] = fit_dict_iq['fr']
            all_Qr[i] = fit_dict_iq['Qr']
            all_amp[i] = fit_dict_iq['amp']
            all_phi[i] = fit_dict_iq['phi']
            all_a[i] = fit_dict_iq['a']
            all_i0[i] = fit_dict_iq['i0']
            all_q0[i] = fit_dict_iq['q0']
            all_tau[i] = fit_dict_iq['tau']
            all_Qc[i] = all_Qr[i] / all_amp[i]
            all_Qi[i] = 1.0 / ((1.0 / all_Qr[i]) - (1.0 / all_Qc[i]))

        except Exception as e:
            print(e)
            print("failed to fit")

    all_fits_dict = {'fits': all_fits, 'fit_results': all_fit_results, 'x0_results': all_x0_results, 'masks': all_masks,
                     'x0': all_x0,
                     'fr': all_fr, 'Qr': all_Qr, 'amp': all_amp, 'phi': all_phi, 'a': all_a, 'i0': all_i0, 'q0': all_q0,
                     'tau': all_tau, 'Qi': all_Qi, 'Qc': all_Qc}

    return all_fits_dict


def fit_linear_mag_multi(f_hz, z):
    """
    wrapper for handling n resonator fits at once
    f_hz and z should have shape n_iq_points x n_res points
    return same thing as fitter but in arrays for all resonators
    """

    center_freqs = f_hz[f_hz.shape[0] // 2, :]

    all_fits = np.zeros((f_hz.shape[1], 5))
    all_fit_results = np.zeros((f_hz.shape[0], f_hz.shape[1]))
    all_x0_results = np.zeros((f_hz.shape[0], f_hz.shape[1]))
    all_masks = np.zeros((f_hz.shape[0], f_hz.shape[1]))
    all_x0 = np.zeros((f_hz.shape[1], 5))
    all_fr = np.zeros(f_hz.shape[1])
    all_Qr = np.zeros(f_hz.shape[1])
    all_amp = np.zeros(f_hz.shape[1])
    all_phi = np.zeros(f_hz.shape[1])
    all_b0 = np.zeros(f_hz.shape[1])
    all_Qi = np.zeros(f_hz.shape[1])
    all_Qc = np.zeros(f_hz.shape[1])

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
        f_single = f_single[use_index]
        z_single = z_single[use_index]

        try:
            fit_dict_iq = fit_linear_mag(f_single, z_single)
            # ranges = np.asarray(([300*10**6,10,0,-3.14,8000],[500*10**6,20,1,3.14,10000]))
            # fit_dict_iq = brute_force_linear_mag_fit(f_single,z_single,ranges = ranges,n_grid_points = 10)

            all_fits[i, :] = fit_dict_iq['fit'][0]
            all_fit_results[:, i] = np.sqrt(linear_mag(f_hz[:, i], all_fits[i, 0], all_fits[i, 1], all_fits[i, 2],
                                                       all_fits[i, 3], all_fits[i, 4]))
            all_x0_results[:, i] = np.sqrt(linear_mag(f_hz[:, i], fit_dict_iq['x0'][0], fit_dict_iq['x0'][1],
                                                      fit_dict_iq['x0'][2], fit_dict_iq['x0'][3], fit_dict_iq['x0'][4]))
            all_masks[:, i] = mask
            all_x0[i, :] = fit_dict_iq['x0']
            all_fr[i] = fit_dict_iq['fr']
            all_Qr[i] = fit_dict_iq['Qr']
            all_amp[i] = fit_dict_iq['amp']
            all_phi[i] = fit_dict_iq['phi']
            all_b0[i] = fit_dict_iq['b0']
            all_Qc[i] = all_Qr[i] / all_amp[i]
            all_Qi[i] = 1.0 / ((1.0 / all_Qr[i]) - (1.0 / all_Qc[i]))

        except Exception as e:
            print("problem")
            print(e)
            print("failed to fit")

    all_fits_dict = {'fits': all_fits, 'fit_results': all_fit_results, 'x0_results': all_x0_results, 'masks': all_masks,
                     'x0': all_x0,
                     'fr': all_fr, 'Qr': all_Qr, 'amp': all_amp, 'phi': all_phi, 'b0': all_b0, 'Qi': all_Qi,
                     'Qc': all_Qc}

    return all_fits_dict


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

    # Caleb's testing area
    res_fit = fit_nonlinear_iq(f_hz=freq_hz_res1, z=s21_complex_res1)
    print(f'Res fit fr from key = {res_fit.result["fr"]}')
    print(f'from index = {res_fit.result[0]}')
    print(f'from slice = {res_fit.result[0:2]}')
    print(f'from attribute = {res_fit.result.fr}')
    res_fit.console()

    # test hash ability
    test_set = {res_fit.result}

    # test the plot
    # res_fit.plot()


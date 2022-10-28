import os
import getpass
from typing import Union
from multiprocessing import Pool

import numpy as np
import matplotlib as mpl
from submm.KIDs.res.data_io import Fit
from submm.KIDs.res.fitting import fit_nonlinear_iq, fit_so_resonator_cable, guess_so_resonator_cable, ResonatorCable

# Debug mode
debug_mode = False
# multiprocessing
# the 'assumption' of max threads is that we are cpu limited in processing,
# so we use should not use more than a computer's available threads
max_threads = int(os.cpu_count())
# Try to strike a balance between the best performance and computer usability during processing
balanced_threads = max(max_threads - 2, 2)
# Use onl half of the available threads for processing
half_threads = int(np.round(os.cpu_count() * 0.5))
current_user = getpass.getuser()
if debug_mode:
    # this will do standard linear processing.
    multiprocessing_threads_default = None
    mpl.use(backend='module://backend_interagg')
elif half_threads < 2:
    multiprocessing_threads_default = None
else:
    multiprocessing_threads_default = half_threads



allowed_fits = ['fit_nonlinear_iq', 'fit_so_resonator_cable']
allowed_fits_int = set(range(len(allowed_fits)))
allowed_functions = {'fit_nonlinear_iq': fit_nonlinear_iq, 'fit_so_resonator_cable': fit_so_resonator_cable}


def fit_nonlinear_iq_wrapped(f_hz, z, tau=None, verbose: bool = True):
    if debug_mode:
        fit_single_res = fit_nonlinear_iq(f_hz, z, tau=tau, verbose=False)
        if verbose:
            fit_single_res.console()
        return fit_single_res
    try:
        fit_single_res = fit_nonlinear_iq(f_hz, z, tau=tau, verbose=False)
    except Exception as excep:
        print(repr(excep))
        print(f"failed to fit freq range: {np.min(f_hz) * 1e-6} - {np.max(f_hz) * 1e-6} MHz\n")
        return None
    else:
        if verbose:
            fit_single_res.console()
        return fit_single_res


def null_func(f_list, *args):
    nan_array = np.empty(len(f_list))
    nan_array[:] = np.nan
    return nan_array


def make_null_fit(f_hz, z, failed_str=None, flag_str=None):
    real = np.real(z)
    imag = np.imag(z)
    x0, bounds = guess_so_resonator_cable(f_hz, real, imag)
    guess = ResonatorCable(*x0)
    popt = np.empty(len(x0))
    popt[:] = np.nan
    pcov = np.empty((len(x0), len(x0)))
    pcov[:] = np.nan

    null_fit = Fit(origin=f'Any Failed Fit: {failed_str}',
                   func=null_func,
                   guess=guess, result=ResonatorCable(fr=(f_hz[0] + f_hz[-1]) / 2.0),
                   popt=popt, pcov=pcov,
                   f_data=f_hz, z_data=z, mask=None,
                   flags={flag_str})
    return null_fit


def fit_so_resonator_cable_wrapped(f_hz, z, verbose: bool = True):
    if debug_mode:
        fit_single_res = fit_so_resonator_cable(f_hz, z, verbose=False)
        if verbose:
            fit_single_res.console()
        return fit_single_res
    try:
        fit_single_res = fit_so_resonator_cable(f_hz, z, verbose=False)
    except RuntimeError as excep:
        excep_str = repr(excep)
        print(excep_str)
        print(f"\nfailed to fit freq range: {np.min(f_hz) * 1e-6} - {np.max(f_hz) * 1e-6} MHz")
        null_fit = make_null_fit(f_hz=f_hz, z=z, failed_str=excep_str, flag_str='failed-fit-runtimeerror')
        return null_fit
    else:
        if verbose:
            fit_single_res.console()
        return fit_single_res


wrapped_functions = {'fit_nonlinear_iq': fit_nonlinear_iq_wrapped,
                     'fit_so_resonator_cable': fit_so_resonator_cable_wrapped}


def fit_select(function_select: Union[str, int]):
    if isinstance(function_select, str):
        function_select = function_select.lower().strip().replace(' ', '_')
        for allowed_fit in allowed_fits:
            if function_select in allowed_fit:
                function_select = allowed_fit
                break
        else:
            raise ValueError(f"function_select must be one of {allowed_fits}")
        function_select = function_select
    elif isinstance(function_select, int):
        if function_select in allowed_fits_int:
            function_select = allowed_fits[function_select]
        else:
            raise ValueError(f"function_select must be one of {allowed_fits_int}")

    else:
        raise TypeError(f"function_select must be a string or int, not {type(function_select)}")
    f_wrapped = wrapped_functions[function_select]
    return f_wrapped


def fit_pool(f_hz_list, z_list, *args, fit_selection: Union[str, int] = 0,
             multiprocessing_threads: Union[int, None] = multiprocessing_threads_default,
             verbose: bool = True) -> list:
    """ For Handling N fits at once using multiprocessing. Elements of f_hz_list and z_list, must be the same length
    per item, but not between items.

    """
    f_wrapped = fit_select(fit_selection)
    if multiprocessing_threads is None:
        res_fits = []
        for f_hz, z in zip(f_hz_list, z_list):
            fit_single_res = f_wrapped(f_hz, z, *args, verbose)
            res_fits.append(fit_single_res)
    else:
        list_len = len(f_hz_list)
        arg_lists = [f_hz_list, z_list]
        for arg in args:
            arg_lists.append([arg] * list_len)
        arg_lists.append([verbose] * list_len)
        star_args = zip(*arg_lists)
        with Pool(multiprocessing_threads) as p:
            res_fits = [fit_single_res for fit_single_res in p.starmap(f_wrapped, star_args)]
    return res_fits


def fit_nonlinear_iq_pool(f_hz_list, z_list, tau: float = None, verbose: bool = True,
                          multiprocessing_threads: Union[int, None] = multiprocessing_threads_default) -> list:
    return fit_pool(f_hz_list, z_list, tau, fit_selection='fit_nonlinear_iq',
                    multiprocessing_threads=multiprocessing_threads, verbose=verbose)


def fit_so_resonator_cable_pool(f_hz_list, z_list, verbose: bool = True,
                                multiprocessing_threads: Union[int, None] = multiprocessing_threads_default) -> list:
    return fit_pool(f_hz_list, z_list, fit_selection='fit_so_resonator_cable',
                    multiprocessing_threads=multiprocessing_threads, verbose=verbose)




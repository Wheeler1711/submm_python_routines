import os
import getpass
from typing import Union
from multiprocessing import Pool

import numpy as np
import matplotlib as mpl
from submm.KIDs.res.data_io import ResSet
from submm.KIDs.res.fitting import fit_nonlinear_iq

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


def fit_nonlinear_iq_wrapper(f_hz, z, tau, verbose):
    try:
        return fit_nonlinear_iq(f_hz, z, tau=tau, verbose=verbose)
    except Exception as e:
        if verbose:
            print(e)
            print(f"failed to fit freq range: {np.min(f_hz) * 1e-6} - {np.max(f_hz) * 1e-6} MHz\n")
        return None


def fit_nonlinear_iq_pool(f_hz_list, z_list,
                          tau: float = None, verbose: bool = True,
                          multiprocessing_threads: Union[int, None] = multiprocessing_threads_default) \
        -> (list, ResSet):
    """ For Handling N fits at once using multiprocessing. Elements of f_hz_list and z_list, must be the same length
    per item, but not between items.

    """
    res_fits = []
    if multiprocessing_threads is None:
        for f_hz, z in zip(f_hz_list, z_list):
            try:
                fit_single_res = fit_nonlinear_iq_wrapper(f_hz, z, tau, verbose)
            except Exception as e:
                res_fits.append(None)
                if verbose:
                    print(e)
                    print("failed to fit")
            else:
                res_fits.append(fit_single_res)
    else:
        star_args = zip(f_hz_list, z_list, [tau] * len(f_hz_list), [verbose] * len(f_hz_list))
        with Pool(multiprocessing_threads) as p:
            res_fits = [fit_single_res for fit_single_res in p.starmap(fit_nonlinear_iq_wrapper, star_args)]
    return res_fits, ResSet(res_fits=res_fits)

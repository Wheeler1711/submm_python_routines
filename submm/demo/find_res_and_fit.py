import os
import time
import numpy as np

from scipy.io import loadmat

from submm.KIDs import find_resonances_interactive as find_kids
from submm.sample_data.abs_paths import abs_path_sample_data
from submm.KIDs.res.sweep_tools import InteractivePlot
from submm.KIDs.res.fitting import fit_nonlinear_iq_multi, fit_linear_mag_multi


def main(linear=False):  # if resonators are well below bifurcation fitting can be much faster

    # load the sample data
    data_path = os.path.join(abs_path_sample_data,
                             "Survey_Tbb20.000K_Tbath170mK_Pow-60dBm_array_temp_sweep_long.mat")
    sample_data = loadmat(data_path)

    # recast the data
    freq_ghz = sample_data['f'][:, 0]
    freq_mhz = freq_ghz * 1.0e3
    freq_hz = freq_ghz * 1.0e9
    s21_complex = sample_data['z'][:, 0]
    s21_mag = 20 * np.log10(np.abs(s21_complex))

    # find resonators interactive
    # 1st filter
    # 2nd find peaks
    # 3rd manual correction
    ip = find_kids.find_vna_sweep(freq_hz, s21_complex)

    # slice up the vna with a span equal to q_slice where q = f/delta_f
    res_freq_array, res_array = find_kids.slice_vna(freq_hz, s21_complex, ip.kid_idx, q_slice=2000, flag_collided=False)

    # fit the resonators
    t1 = time.time()
    if not linear:
        res_set = fit_nonlinear_iq_multi(res_freq_array, res_array, tau=97 * 10 ** -9)
    else:
        res_set = fit_linear_mag_multi(res_freq_array, res_array)
    t2 = time.time()
    print("time to fit {:.2f} s".format(t2 - t1))

    # if you want to inspect just the first fit to understand the data
    result = next(iter(res_set))
    # and then for fit_result
    fit_result = res_set._fit_results[result]
    # to see what is in the class
    print("fields in a result")
    print(result._fields)
    print("fields in a fit_result")
    print(fit_result._fields)

    # plot the data, see submm/KIDs/res/data_io.py's ResSet plot function for example of sweep plotting
    ip2 = res_set.plot(flags=ip.flags)

    # below is example of retuning resonators

    # ip2 = InteractivePlot(res_freq_array,res_array,retune = True,find_min = False)


if __name__ == "__main__":
    main()

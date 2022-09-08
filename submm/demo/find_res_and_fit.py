import os
import time

import numpy as np
from scipy.io import loadmat

from submm.KIDs import find_resonances_interactive as find_kids
from submm.sample_data.abs_paths import abs_path_sample_data
from submm.KIDs.res import sweep_tools as res_sweep_tools, fitting as res_fit

linear = False  # if resonators are well below bifurcation fitting can be much faster

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

# slice up the vna with a span equal to to q_slice where q = f/delta_f
res_freq_array, res_array = find_kids.slice_vna(freq_hz, s21_complex, ip.kid_idx, q_slice=2000, flag_collided=False)

# fit the resonators
t1 = time.time()
if not linear:
    fits = res_fit.fit_nonlinear_iq_multi(res_freq_array.T, res_array.T, tau=97 * 10 ** -9)
else:
    fits = res_fit.fit_linear_mag_multi(res_freq_array.T, res_array.T)
t2 = time.time()
print("time to fit {:.2f} s".format(t2 - t1))

# stack fit data with Qi and Qc skip second f0
if not linear:
    fit_data = np.vstack((fits['fits'][:, :-1].T, fits['Qi'], fits['Qc'])).T
    fit_data[:, 0] = fit_data[:, 0] / 10 ** 6
    fit_data[:, 7] = fit_data[:, 7] * 10 ** 9
    data_names = ["Resonator Frequencies (MHz)", "Qr", "amp", "phi", "a", "i0", "q0", "tau (ns)", "Qi", "Qc"]
else:
    fit_data = np.vstack((fits['fits'][:, :].T, fits['Qi'], fits['Qc'])).T
    fit_data[:, 0] = fit_data[:, 0] / 10 ** 6
    data_names = ["Resonator Frequencies (MHz)", "Qr", "amp", "phi", "b0", "Qi", "Qc"]

# stack the data with fit data
multi_sweep_freqs = np.dstack((np.expand_dims(res_freq_array.T, axis=2), np.expand_dims(res_freq_array.T, axis=2)))
multi_sweep_z = np.dstack((np.expand_dims(res_array.T, axis=2), np.expand_dims(fits['fit_results'], axis=2)))

ip2 = res_sweep_tools.InteractivePlot(multi_sweep_freqs, multi_sweep_z, retune=False, combined_data=fit_data,
                                      combined_data_names=data_names,
                                      sweep_labels=['Data', 'Fit'])

# below is example of retuning resonators

# ip2 = res_sweep_tools.InteractivePlot(res_freq_array.T,res_array.T,retune = True,find_min = False)

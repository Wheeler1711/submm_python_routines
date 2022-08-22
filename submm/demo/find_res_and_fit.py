import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from submm.KIDs import resonance_fitting as res_fit
from submm.KIDs import find_resonances_interactive as find_kids
from submm.sample_data.abs_paths import abs_path_sample_data
from importlib import reload

#load the sample data

data_path = os.path.join(abs_path_sample_data,
                         "Survey_Tbb20.000K_Tbath170mK_Pow-60dBm_array_temp_sweep_long.mat")
sample_data = loadmat(data_path)


freq_ghz = sample_data['f'][:, 0]
freq_mhz = freq_ghz * 1.0e3
freq_hz = freq_ghz * 1.0e9
s21_complex = sample_data['z'][:, 0]
s21_mag = 20 * np.log10(np.abs(s21_complex))

ip = find_kids.find_vna_sweep(freq_hz,s21_complex)

res_freq_array,res_array = find_kids.slice_vna(freq_hz,s21_complex,ip.kid_idx,q_slice = 5000,flag_collided = False)

fits = res_fit.fit_nonlinear_iq_multi(res_freq_array.T,res_array.T)

# new plotter goes here

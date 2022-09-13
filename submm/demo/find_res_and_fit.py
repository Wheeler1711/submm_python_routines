import os
import time

import numpy as np
from scipy.io import loadmat

from submm.KIDs import find_resonances_interactive as find_kids
from submm.sample_data.abs_paths import abs_path_sample_data
from submm.KIDs.res.sweep_tools import InteractivePlot
from submm.KIDs.res.fitting import fit_nonlinear_iq_multi, fit_linear_mag_multi

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
    res_set = fit_nonlinear_iq_multi(res_freq_array, res_array, tau=97 * 10 ** -9)
else:
    res_set = fit_linear_mag_multi(res_freq_array, res_array)
t2 = time.time()
print("time to fit {:.2f} s".format(t2 - t1))

# if you want to inspect just the first fit to understand the data
result = next(iter(sorted(res_set.results)))
# and then for fit_result
fit_result = res_set.fit_results[result]
# to see what is in the class
print("fields in a fit_result")
print(fit_result._fields)

# plotter want frequencies and z values with shape n_frequency_point x n_res x n_sweeps
# also for fitted paramters it wants a list fitted names for the legend
# and a array that is n_res x len(fitted paramters)

# Make the frequency and z arrays first
fitted_frequencies = np.zeros(res_freq_array.shape)
fitted_z_values = np.zeros(res_array.shape,dtype = 'complex')
for i,result in enumerate(sorted(res_set.results)): # set needs to be sorted to match input data
    fit_result = res_set.fit_results[result]
    fitted_frequencies[:,i] = fit_result.f_data # input frequencies nominally the same as res_freq_array
    fitted_z_values[:,i] = fit_result.z_fit()
    
# now get the fitted values
# first get names of fitted values
data_names = []
result = next(iter(sorted(res_set.results))) # grab first fit for inspection
for field in result._fields:
    if getattr(result, field) != None:
        data_names.append(field)

fitted_parameters = np.zeros((len(res_set.results),len(data_names)))
for i, result in enumerate(sorted(res_set.results)): # don't forget to sort
    for j, field in enumerate(data_names): # this gets rid of the the None
        fitted_parameters[i,j] = getattr(result,field)
        

# stack the data with fit data
multi_sweep_freqs = np.dstack((np.expand_dims(res_freq_array, axis=2), np.expand_dims(fitted_frequencies, axis=2)))
multi_sweep_z = np.dstack((np.expand_dims(res_array, axis=2), np.expand_dims(fitted_z_values, axis=2)))

ip2 = InteractivePlot(multi_sweep_freqs, multi_sweep_z, retune=False, combined_data=fitted_parameters,
                      combined_data_names=data_names,
                      sweep_labels=['Data', 'Fit'])

# below is example of retuning resonators

#ip2 = InteractivePlot(res_freq_array,res_array,retune = True,find_min = False)

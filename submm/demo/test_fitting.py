import os
import time
import numpy as np

from scipy.io import loadmat

from submm.KIDs import find_resonances_interactive as find_kids
from submm.sample_data.abs_paths import abs_path_sample_data
from submm.KIDs.res.sweep_tools import InteractivePlot
from submm.KIDs.res.fitting import fit_nonlinear_iq_multi, fit_linear_mag_multi
from submm.KIDs.res.data_io import field_to_field_label, field_to_format_strs, field_to_format_strs, field_to_multiplier
from submm.KIDs.calibrate import fit_cable_delay_from_slope, fit_cable_delay


def main(linear=False,data_set = 2):  # if resonators are well below bifurcation fitting can be much faster

    # load the sample data
    # HAWC+ TiN KIDs
    if data_set == 1:
        data_path = os.path.join(abs_path_sample_data,
                             "Survey_Tbb20.000K_Tbath170mK_Pow-60dBm_array_temp_sweep_long.mat")
        q_slice=2000
    else: #CCAT aluminum array with the longest wirebonds ever
        data_path = os.path.join(abs_path_sample_data,
                             "survey_100mK_minus50dBm.mat")
        q_slice=2500
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
    #ip = find_kids.find_vna_sweep(freq_hz, s21_complex)
    tau_2,fit_data_phase,gain_phase = fit_cable_delay(freq_hz, np.arctan2(np.real(s21_complex),np.imag(s21_complex)))
    print(tau_2)
    tau,phase_gradient =  fit_cable_delay_from_slope(freq_hz, np.arctan2(np.real(s21_complex),np.imag(s21_complex)))

    #np.save("Jordan_indexes.npy",ip.kid_idx)
    #print(ip.kid_idx)
    kid_idx = np.load(os.path.join(abs_path_sample_data,"Jordan_indexes.npy"))

    # slice up the vna with a span equal to q_slice where q = f/delta_f
    res_freq_array, res_array = find_kids.slice_vna(freq_hz, s21_complex, kid_idx, q_slice=q_slice, flag_collided=False)

    # fit the resonators
    t1 = time.time()
    if not linear:
        res_set = fit_nonlinear_iq_multi(res_freq_array, res_array,tau = tau,eqn = 'standard') #eqn = standard or ss
        res_set_2 = fit_nonlinear_iq_multi(res_freq_array, res_array,tau = tau,eqn = 'ss')
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
    #ip2 = res_set.plot(show_guess = True)

    # below is example of retuning resonators

    # ip2 = InteractivePlot(res_freq_array,res_array,retune = True,find_min = False)

    show_guess = True
    # Make the frequency and z arrays for original data and fitted
    n_iq_points = len(res_set._fit_results[next(iter(res_set))].z_data)  # assumes all data is the same size
    frequencies = np.zeros((n_iq_points, len(res_set)))
    fitted_frequencies = np.zeros((n_iq_points, len(res_set)))
    z_values = np.zeros((n_iq_points, len(res_set)), dtype='complex')
    fitted_z_values = np.zeros((n_iq_points, len(res_set)), dtype='complex')
    guess_z_values = np.zeros((n_iq_points, len(res_set)), dtype='complex')
    for i, result in enumerate(res_set):
        fit_result = res_set._fit_results[result]
        frequencies[:, i] = fit_result.f_data  # input frequencies nominally the same as res_freq_array
        fitted_frequencies[:, i] = fit_result.f_data  # input frequencies nominally the same as res_freq_array
        z_values[:, i] = fit_result.z_data
        fitted_z_values[:, i] = fit_result.z_fit()
        guess_z_values[:, i] = fit_result.z_guess()

    # Make the frequency and z arrays for original data and fitted
    n_iq_points = len(res_set._fit_results[next(iter(res_set))].z_data)  # assumes all data is the same size
    frequencies_2 = np.zeros((n_iq_points, len(res_set)))
    fitted_frequencies_2 = np.zeros((n_iq_points, len(res_set)))
    z_values_2 = np.zeros((n_iq_points, len(res_set)), dtype='complex')
    fitted_z_values_2 = np.zeros((n_iq_points, len(res_set)), dtype='complex')
    guess_z_values_2 = np.zeros((n_iq_points, len(res_set)), dtype='complex')
    for i, result in enumerate(res_set_2):
        fit_result_2 = res_set_2._fit_results[result]
        frequencies_2[:, i] = fit_result_2.f_data  # input frequencies nominally the same as res_freq_array
        fitted_frequencies_2[:, i] = fit_result_2.f_data  # input frequencies nominally the same as res_freq_array
        z_values_2[:, i] = fit_result_2.z_data
        fitted_z_values_2[:, i] = fit_result_2.z_fit()
        guess_z_values_2[:, i] = fit_result_2.z_guess()


    # stack the data with fit data
    if show_guess:
        multi_sweep_freqs = np.dstack((np.expand_dims(frequencies, axis=2),
                                           np.expand_dims(fitted_frequencies,axis=2),
                                           np.expand_dims(fitted_frequencies, axis=2),
                                           np.expand_dims(fitted_frequencies, axis=2),
                                           np.expand_dims(fitted_frequencies, axis=2)))
        multi_sweep_z = np.dstack((np.expand_dims(z_values, axis=2),
                                       np.expand_dims(fitted_z_values, axis=2),
                                       np.expand_dims(guess_z_values, axis=2),
                                       np.expand_dims(fitted_z_values_2, axis=2),
                                       np.expand_dims(guess_z_values_2, axis=2)))
        sweep_labels = ['Data', 'Fit standard','Guess standard','Fit ss','Guess ss']
    else:
        multi_sweep_freqs = np.dstack((np.expand_dims(frequencies, axis=2), np.expand_dims(fitted_frequencies, axis=2)))
        multi_sweep_z = np.dstack((np.expand_dims(z_values, axis=2), np.expand_dims(fitted_z_values, axis=2)))
        sweep_labels  = ['Data', 'Fit']

    # now get the fitted values
    # first get names of fitted values
    data_names = []
    formats = []
    result = next(iter(res_set))  # grab first fit for inspection
    for field in result._fields:
        if getattr(result, field) is not None:
            data_names.append(field)
            data_names.append(field+" ss")
            formats.append(field_to_field_label[field.lower()] + ': {:' + field_to_format_strs[field.lower()] + '}')
            formats.append(field_to_field_label[field.lower()] + ': {:' + field_to_format_strs[field.lower()] + '}')

    fitted_parameters = np.zeros((len(res_set), len(data_names)*2))
    for i, result in enumerate(res_set):
        for j, field in enumerate(data_names[::2]): # this gets rid of the None
            value = getattr(result, field)
            if value is None or np.isnan(value):
                value = -99.99
            if field in field_to_multiplier.keys():
                fitted_parameters[i, 2*j+0] = value * field_to_multiplier[field.lower()]
            else:
                fitted_parameters[i, 2*j+0] = value

    for i, result in enumerate(res_set_2):
        for j, field in enumerate(data_names[::2]): # this gets rid of the None
            value = getattr(result, field)
            if value is None or np.isnan(value):
                value = -99.99
            if field in field_to_multiplier.keys():
                fitted_parameters[i, 2*j+1] = value * field_to_multiplier[field.lower()]
            else:
                fitted_parameters[i, 2*j+1] = value



    # run the plotter
    ip = InteractivePlot(multi_sweep_freqs, multi_sweep_z, retune=False, combined_data=fitted_parameters,
                         combined_data_names=data_names, combined_data_format=formats,
                         sweep_labels=sweep_labels,
                         verbose=res_set.verbose)

    return res_set


if __name__ == "__main__":
    main()

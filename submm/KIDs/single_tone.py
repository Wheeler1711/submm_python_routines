import time
import pickle
import os.path

import numpy as np
import matplotlib.pyplot as plt

from submm.lab_brick import core
from submm.instruments import NIDAQ as n
from submm.instruments import anritsu as an
from submm.KIDs.res.fitting import fit_nonlinear_mag_sep, fit_nonlinear_iq_sep
from submm.KIDs.res.utils import guess_x0_mag_nonlinear_sep, guess_x0_iq_nonlinear_sep


class single_tone(object):

    def __init__(self):

        # Declare two internal objects: Anritsu/NIDAQ
        self.anritsu = an.Anritsu()
        self.daq = n.NIDAQ()
        self.daq.sample_rate = 60000

        # Initializing Standard Configurations
        self.switch_time = 0.1
        self.iq_integration_time = 1.
        self.integration_time = 45
        self.anritsu.set_power(8)
        self.anritsu.turn_outputOn()
        self.iq_dictionary = {}
        self.power_dictionary = {}

        # Preset Span variables
        self.gain_span = 2 * 10 ** 6
        self.rough_span = 500 * 10 ** 3
        self.med_span = 100 * 10 ** 3
        self.fine_span = 20 * 10 ** 3

        # Preset number of points to be taken
        self.gain_numpoints = 100
        self.rough_numpoints = 100
        self.med_numpoints = 100
        self.fine_numpoints = 100

        # Center freq to be saved
        self.center_frequency = 0

        # Directory for Dictionary data
        self.output_dir = "c:/users/Tycho/Data/"

        # Initialize the lab brick
        # If lab brick isn't connected, output to screen
        try:
            self.input_attn = core.Attenuator(0x041f, 0x1208, "15915")
            self.input_attn.set_attenuation(30)
            time.sleep(0.5)
            self.input_attn_value = self.input_attn.get_attenuation()
        except:
            print("Unable to connect to input attenuator")
            self.input_attn = None
        try:
            self.output_attn = core.Attenuator(0x041f, 0x1208, "16776")
            self.output_attn.set_attenuation(30)
            time.sleep(0.5)
            self.output_attn_value = self.output_attn.get_attenuation()
        except:
            print("Unable to connect to output attenuator")
            self.output_attn = None

    # This method passes a frequency, span(+-), and number
    # of total points to be evaluated. The function measures
    # the profile of the resonator and returns 3 arrays:
    # 1 frequency array and 2 voltages array
    def iq_sweep(self, center_freq, span, numpoints):
        i = np.zeros(numpoints)
        q = np.zeros(numpoints)

        freqs = np.linspace(center_freq - span / 2, center_freq + span / 2, numpoints)

        for k in range(0, numpoints):
            print(k)
            self.anritsu.set_frequency(freqs[k])
            time.sleep(self.switch_time)

            i[k], q[k] = self.daq.average_2ch(self.iq_integration_time)

        return freqs, i, q

    # This method passes a frequency to be streamed
    # and returns two arrays that stores frequency
    def stream(self, center_freq):

        self.anritsu.set_frequency(center_freq)
        time.sleep(self.switch_time)

        i, q = self.daq.stream_2ch(self.integration_time)

        return i, q

    def stream3(self, center_freq):

        self.anritsu.set_frequency(center_freq)
        time.sleep(self.switch_time)

        i, q, ref = self.daq.stream_3ch(self.integration_time)

        return i, q, ref

    # This method finds the minimum frequency and
    # the index position of that frequency
    def find_min(self, freqs, i, q):
        index = np.argmin(i ** 2 + q ** 2)

        return freqs[index], index

    # This methods finds the largest i and q separation,
    # and returns the value and it's index position
    def find_max_iq_sep(self, freqs, i, q):
        i2 = np.roll(i, 1)
        q2 = np.roll(q, 1)

        dist = np.sqrt((i - i2) ** 2 + (q - q2) ** 2)

        dist[0] = 0
        dist[-1] = 0

        index = np.argmax(dist)

        return freqs[index], index

    def rough(self, center_freq):
        freqs, i, q = self.iq_sweep(center_freq, self.rough_span, self.rough_numpoints)

        return freqs, i, q

    def med(self, center_freq):
        freqs, i, q = self.iq_sweep(center_freq, self.med_span, self.med_numpoints)

        return freqs, i, q

    def fine(self, center_freq):
        freqs, i, q = self.iq_sweep(center_freq, self.fine_span, self.fine_numpoints)

        return freqs, i, q

    def gain(self, center_freq):
        freqs, i, q = self.iq_sweep(center_freq, self.gain_span, self.gain_numpoints)

        return freqs, i, q

    # This method passes a frequency to be evaluated and
    # returns a dictionary with all the values being
    # tested. It finds the best place to take data

    def take_noise_set(self, center_freq, chan3=False, take_noise=True, filename="",
                       pause_before_noise=False):

        if (len(filename) < 2):

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = os.path.join(self.output_dir + timestr + "_noiseData.txt")
            file_name2 = os.path.join(self.output_dir + timestr)
        else:
            file_name = str(filename + ".txt")
            file_name2 = str(filename)

        self.center_frequency = center_freq

        print("taking rough scan")
        freqs_rough, I_rough, Q_rough = self.rough(center_freq)
        rough_center_freq, min_pos_rough = self.find_min(freqs_rough, I_rough, Q_rough)

        print("taking med scan")
        freqs_med, I_med, Q_med = self.med(rough_center_freq)
        med_center_freq, min_pos_med = self.find_min(freqs_med, I_med, Q_med)

        print("taking fine scan")
        freqs_fine, I_fine, Q_fine = self.fine(med_center_freq)
        fine_center_freq, max_iq_pos = self.find_max_iq_sep(freqs_fine, I_fine, Q_fine)

        print("taking gain scan")
        freqs_gain, I_gain, Q_gain = self.gain(fine_center_freq)

        if (take_noise == True):

            print('\n')
            if pause_before_noise == True:
                input("Program paused before streaming. Press enter to continue....")
            if (chan3 == True):
                print("taking noise data")
                I_noise, Q_noise, ref_noise = self.stream3(fine_center_freq)
                self.iq_dictionary['ref_noise'] = ref_noise
                self.iq_dictionary['I_noise'] = I_noise
                self.iq_dictionary['Q_noise'] = Q_noise
            else:
                I_noise, Q_noise = self.stream(fine_center_freq)
                self.iq_dictionary['I_noise'] = I_noise
                self.iq_dictionary['Q_noise'] = Q_noise

        self.iq_dictionary['freqs_rough'] = freqs_rough
        self.iq_dictionary['I_rough'] = I_rough
        self.iq_dictionary['Q_rough'] = Q_rough
        self.iq_dictionary['freqs_med'] = freqs_med
        self.iq_dictionary['I_med'] = I_med
        self.iq_dictionary['Q_med'] = Q_med
        self.iq_dictionary['freqs_fine'] = freqs_fine
        self.iq_dictionary['I_fine'] = I_fine
        self.iq_dictionary['Q_fine'] = Q_fine
        self.iq_dictionary['freqs_gain'] = freqs_gain
        self.iq_dictionary['I_gain'] = I_gain
        self.iq_dictionary['Q_gain'] = Q_gain
        self.iq_dictionary['rough_center_freq'] = rough_center_freq
        self.iq_dictionary['med_center_freq'] = med_center_freq
        self.iq_dictionary['fine_center_freq'] = fine_center_freq
        self.iq_dictionary['min_pos_rough'] = min_pos_rough
        self.iq_dictionary['min_pos_med'] = min_pos_med
        self.iq_dictionary['max_iq_pos'] = max_iq_pos

        try:
            self.iq_dictionary['input_attn'] = self.input_attn_value
            self.iq_dictionary['output_attn'] = self.output_attn_value
        except:
            pass

        self.export_file(file_name, self.iq_dictionary)
        self.save_log_iq(file_name2)

        print('\n')
        print("Best Frequency: " + str(fine_center_freq / 10 ** 6) + " Mhz")

        return self.iq_dictionary

    # This method exports the object's dictionary
    # to a file to be saved
    def export_file(self, file_name, dictionary):

        file_object = open(str(file_name), "w")
        pickle.dump(dictionary, file_object)
        file_object.close()

    # This method returns a saved file of all the variables
    # and their values
    def save_log_iq(self, file_name):

        file_object = open(file_name + "_logIQ.txt", "w")

        file_object.write("switch_time: " + str(self.switch_time) + "\n")
        file_object.write("iq_integration_time: " + str(self.iq_integration_time) + "\n")
        file_object.write("integration_time: " + str(self.integration_time) + "\n")
        file_object.write("anritsu_power: " + str(self.anritsu.get_power()) + "\n")
        file_object.write("gain_span: " + str(self.gain_span) + "\n")
        file_object.write("rough_span: " + str(self.rough_span) + "\n")
        file_object.write("med_span: " + str(self.med_span) + "\n")
        file_object.write("fine_span: " + str(self.fine_span) + "\n")
        file_object.write("gain_numpoints: " + str(self.gain_numpoints) + "\n")
        file_object.write("rough_numpoints: " + str(self.rough_numpoints) + "\n")
        file_object.write("med_numpoints: " + str(self.med_numpoints) + "\n")
        file_object.write("fine_numpoints: " + str(self.fine_numpoints) + "\n")
        file_object.write("input_attn_value: " + str(self.input_attn_value) + "\n")
        try:
            file_object.write("output_attn_value: " + str(self.output_attn_value) + "\n")
        except:
            pass
        file_object.write("sample_rate: " + str(self.daq.sample_rate) + "\n")
        file_object.write("center_freq: " + str(self.center_frequency) + "\n")

        file_object.close()

    # If the input changes, then output needs to be
    # reciprocated to that input value
    def change_attn_balanced(self, input):

        if (self.output_attn != None):

            delta_attn = input - self.input_attn_value

            self.input_attn.set_attenuation(input)
            time.sleep(0.5)
            self.input_attn_value = self.input_attn.get_attenuation()

            self.output_attn.set_attenuation(self.output_attn_value - delta_attn)
            time.sleep(0.5)
            self.output_attn_value = self.output_attn.get_attenuation()
        else:
            self.input_attn.set_attenuation(input)
            time.sleep(0.5)
            self.input_attn_value = self.input_attn.get_attenuation()

    def set_input_attn(self, input):
        self.input_attn.set_attenuation(input)
        time.sleep(0.5)
        self.input_attn_value = self.input_attn.get_attenuation()

    def set_output_attn(self, output):

        if (self.output_attn != None):
            self.output_attn.set_attenuation(output)
            time.sleep(0.5)
            self.output_attn_value = self.output_attn.get_attenuation()
        else:
            print("Output not connected!")

    def power_sweep(self, low_power, high_power, step, center_freq, filename=""):

        if (filename == ""):

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = os.path.join(self.output_dir + timestr + "_powerData.txt")
        else:
            file_name = os.path.join(self.output_dir + filename + ".txt")
            file_name2 = os.path.join(self.output_dir + filename)

        powers = np.arange(low_power, high_power - step, -step)

        i_gain = np.zeros(shape=(len(powers), self.gain_numpoints))
        q_gain = np.zeros(shape=(len(powers), self.gain_numpoints))
        i_fine = np.zeros(shape=(len(powers), self.fine_numpoints))
        q_fine = np.zeros(shape=(len(powers), self.fine_numpoints))
        freqs_gain = np.linspace(center_freq - (self.gain_span / 2), center_freq + (self.gain_span / 2),
                                 self.gain_numpoints)
        freqs_fine = np.linspace(center_freq - (self.fine_span / 2), center_freq + (self.fine_span / 2),
                                 self.fine_numpoints)
        position = 0

        for k in range(0, len(powers)):
            freqs_gain, i_gain[k][:], q_gain[k][:] = self.gain(center_freq)
            freqs_fine, i_fine[k][:], q_fine[k][:] = self.fine(center_freq)

            center_freq, position = self.find_min(freqs_fine, i_fine[k][:], q_fine[k][:])

        self.power_dictionary['powers'] = powers
        self.power_dictionary['i_gain'] = i_gain
        self.power_dictionary['q_gain'] = q_gain
        self.power_dictionary['i_fine'] = i_fine
        self.power_dictionary['q_fine'] = q_fine
        self.power_dictionary['freqs_gain'] = freqs_gain
        self.power_dictionary['freqs_fine'] = freqs_fine
        self.power_dictionary['center_freq'] = center_freq
        self.power_dictionary['position'] = position

        self.export_file(file_name, self.power_dictionary)
        self.save_log_power(file_name2)
        return self.power_dictionary

    def save_log_power(self, file_name):

        file_object = open(file_name + "_logPower.txt", "w")

        file_object.write("Powers: " + str(self.power_dictionary['powers']) + "\n")
        file_object.write("i_gain: " + str(self.power_dictionary['i_gain']) + "\n")
        file_object.write("q_gain: " + str(self.power_dictionary['q_gain']) + "\n")
        file_object.write("i_fine: " + str(self.power_dictionary['i_fine']) + "\n")
        file_object.write("q_fine: " + str(self.power_dictionary['q_fine']) + "\n")
        file_object.write("freqs_gain: " + str(self.power_dictionary['freqs_gain']) + "\n")
        file_object.write("freqs_fine: " + str(self.power_dictionary['freqs_fine']) + "\n")
        file_object.write("center_freq: " + str(self.power_dictionary['center_freq']) + "\n")
        file_object.write("position: " + str(self.power_dictionary['position']) + "\n")

        file_object.close()


# This method opens up a file that contains the object's
# dictionary and stores it back into another dictionary
def import_file(file_name):
    file_object = open(file_name, "rU")
    dictionary = pickle.load(file_object)
    file_object.close()
    return dictionary


# Fitting noise and guessing
def fit_noise_set(dictionary):
    # figure out where to save
    fine_f = dict['freqs_fine']
    gain_f = dict['freqs_gain']
    fine_z = dict['I_fine'] + 1.j * dict['Q_fine']
    gain_z = dict['I_gain'] + 1.j * dict['Q_gain']

    fig = plt.figure(figsize=(8, 8))

    # Subplot 221/223 refers to noise and power
    plt.subplot(221)
    plt.plot(dict['freqs_fine'] / 10 ** 6, 10 * np.log10(dict['I_fine'] ** 2 + dict['Q_fine'] ** 2), 'o', label="fine")
    plt.plot(dict['freqs_gain'] / 10 ** 6, 10 * np.log10(dict['I_gain'] ** 2 + dict['Q_gain'] ** 2), 'o', label="gain")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")

    plt.subplot(223)
    plt.plot(dict['freqs_fine'] / 10 ** 6, 10 * np.log10(dict['I_fine'] ** 2 + dict['Q_fine'] ** 2), 'o')
    plt.plot(dict['freqs_gain'] / 10 ** 6, 10 * np.log10(dict['I_gain'] ** 2 + dict['Q_gain'] ** 2), 'o')
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.xlim(np.min(dict['freqs_fine'] / 10 ** 6), np.max(dict['freqs_fine'] / 10 ** 6))

    # fit nonlinear magnitude
    try:
        x0 = guess_x0_mag_nonlinear_sep(fine_f, fine_z, gain_f, gain_z, verbose=True)
        fit_dict_mag = fit_nonlinear_mag_sep(fine_f, fine_z, gain_f, gain_z, x0=x0)  # ,bounds =bounds)

        plt.subplot(221)
        plt.plot(fit_dict_mag['fit_freqs'] / 10 ** 6, 10 * np.log10(fit_dict_mag['fit_result']), "+", label="fit")
        plt.plot(fit_dict_mag['fit_freqs'] / 10 ** 6, 10 * np.log10(fit_dict_mag['x0_result']), "x", label="x0 guess")
        plt.title(str("Non-linearity param a = %.2f") % fit_dict_mag['fit'][0][4])

        plt.legend()
        plt.subplot(223)
        plt.plot(fit_dict_mag['fit_freqs'] / 10 ** 6, 10 * np.log10(fit_dict_mag['fit_result']), "+")
        plt.plot(fit_dict_mag['fit_freqs'] / 10 ** 6, 10 * np.log10(fit_dict_mag['x0_result']), "x")
    except Exception as e:
        print(e)
        print("could not fit the resonator")

    # Subplot 222/224 refers to just noise
    plt.subplot(222, aspect='equal')
    plt.plot(dict['I_fine'], dict['Q_fine'], 'o')
    plt.plot(dict['I_gain'], dict['Q_gain'], 'o')
    plt.xlabel("I")
    plt.ylabel("Q")

    plt.subplot(224, aspect='equal')
    plt.plot(dict['I_fine'], dict['Q_fine'], 'o')
    plt.plot(dict['I_gain'], dict['Q_gain'], 'o')

    plt.xlabel("I")
    plt.ylabel("Q")
    plt.xlim(np.min(dict['I_fine']), np.max(dict['I_fine']))
    plt.ylim(np.min(dict['Q_fine']), np.max(dict['Q_fine']))

    # fit nonlinear iq
    try:
        x0 = guess_x0_iq_nonlinear_sep(fine_f, fine_z, gain_f, gain_z, verbose=True)
        fit_dict_iq = fit_nonlinear_iq_sep(fine_f, fine_z, gain_f, gain_z, x0=x0)

        plt.subplot(222, aspect='equal')
        plt.plot(np.real(fit_dict_iq['fit_result']), np.imag(fit_dict_iq['fit_result']), "+")
        plt.plot(np.real(fit_dict_iq['x0_result']), np.imag(fit_dict_iq['x0_result']), "x")

        plt.title(str("Non-linearity param a = %0.2f") % fit_dict_iq['fit'][0][4])

        plt.subplot(224, aspect='equal')
        plt.plot(np.real(fit_dict_iq['fit_result']), np.imag(fit_dict_iq['fit_result']), "+")
        plt.plot(np.real(fit_dict_iq['x0_result']), np.imag(fit_dict_iq['x0_result']), "x")
        plt.plot(dict['I_fine'], dict['Q_fine'])
    except Exception as e:
        print(e)
        print("could not fit the resonator")

    plt.show()


# This method passes a dictionary parameter and plots
# the dictionary values and returns 2 figures to be
# evaluated
def plot_iq_dict(dictionary):
    plt.figure(1)
    plt.title("IQ")
    plt.xlabel("I")
    plt.ylabel("Q")

    try:
        plt.plot(dictionary['I_noise'][::100], dictionary['Q_noise'][::100], 'm-', label="IQ noise")

    except:
        pass

    plt.plot(dictionary['I_rough'], dictionary['Q_rough'], 'r-', label="rough")
    plt.plot(dictionary['I_med'], dictionary['Q_med'], 'b-', label="med")
    plt.plot(dictionary['I_fine'], dictionary['Q_fine'], 'g-', label="fine")
    plt.plot(dictionary['I_gain'], dictionary['Q_gain'], 'y-', label="gain")
    plt.plot(dictionary['I_fine'][dictionary['max_iq_pos']], dictionary['Q_fine'][dictionary['max_iq_pos']], '*',
             label="max sep")
    plt.legend()

    plt.figure(2)
    plt.title("Magnitude")
    plt.xlabel("Frequency (Mhz)")
    plt.ylabel("Magnitude")

    plt.plot(dictionary['freqs_rough'] / 10 ** 6, dictionary['I_rough'] ** 2 + dictionary['Q_rough'] ** 2, 'r-',
             label="rough")
    plt.plot(dictionary['freqs_rough'][dictionary['min_pos_rough']] / 10 ** 6,
             dictionary['I_rough'][dictionary['min_pos_rough']] ** 2 + dictionary['Q_rough'][
                 dictionary['min_pos_rough']] ** 2, '*')
    plt.plot(dictionary['freqs_med'] / 10 ** 6, dictionary['I_med'] ** 2 + dictionary['Q_med'] ** 2, 'b-', label="med")
    plt.plot(dictionary['freqs_med'][dictionary['min_pos_med']] / 10 ** 6,
             dictionary['I_med'][dictionary['min_pos_med']] ** 2 + dictionary['Q_med'][dictionary['min_pos_med']] ** 2,
             '*', label="med min")

    plt.plot(dictionary['freqs_fine'] / 10 ** 6, dictionary['I_fine'] ** 2 + dictionary['Q_fine'] ** 2, 'g-',
             label="fine")
    plt.plot(dictionary['freqs_fine'][dictionary['max_iq_pos']] / 10 ** 6,
             dictionary['I_fine'][dictionary['max_iq_pos']] ** 2 + dictionary['Q_fine'][dictionary['max_iq_pos']] ** 2,
             '*', label="iq pos")

    plt.plot(dictionary['freqs_gain'] / 10 ** 6, dictionary['I_gain'] ** 2 + dictionary['Q_gain'] ** 2, 'y-',
             label="gain")

    plt.legend()

    plt.show()


def plot_power_dict(dictionary):
    plt.figure(1)
    plt.title("I&Q w/ Powers")
    plt.xlabel("I_fine")
    plt.ylabel("Q_fine")

    for k in range(0, len(dictionary['powers'])):
        plt.plot(dictionary['i_fine'][k, :], dictionary['q_fine'][k, :], label=str(dictionary['powers'][k]) + " db")

    plt.legend()
    plt.show()


def plot_iq_single(freq, i, q):
    plt.figure(1)
    plt.title("Magnitude")
    plt.xlabel("Frequency (Mhz)")
    plt.ylabel("Power (dB)")

    plt.plot((freq / 10 ** 6), (10 * np.log10(i ** 2 + q ** 2)))

    plt.show()

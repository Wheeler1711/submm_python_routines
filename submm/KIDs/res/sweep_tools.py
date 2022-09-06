"""
Tools for handling resonator iq sweeps
i.e. finding minimum, maximum of iq sweep arrays
plotting iq sweep arrays
retuning resonators
finding correct readout power level
modified from https://github.com/sbg2133/kidPy/
"""

import copy
import platform

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def find_max_didq(z, look_around):
    Is = np.real(z)
    Qs = np.imag(z)
    pos_offset_I = np.roll(Is, 1, axis=0)
    neg_offset_I = np.roll(Is, -1, axis=0)
    pos_offset_Q = np.roll(Qs, 1, axis=0)
    neg_offset_Q = np.roll(Qs, -1, axis=0)
    pos_dist = np.sqrt((Is - pos_offset_I) ** 2 + (Qs - pos_offset_Q) ** 2)
    neg_dist = np.sqrt((Is - neg_offset_I) ** 2 + (Qs - neg_offset_Q) ** 2)
    ave_dist = (pos_dist + neg_dist) / 2.
    # zero out the last and first values
    ave_dist[0, :] = 0
    ave_dist[ave_dist.shape[0] - 1, :] = 0
    min_index = np.argmax(ave_dist[Is.shape[0] // 2 - look_around:Is.shape[0] // 2 + look_around],
                          axis=0) + (Is.shape[0] // 2 - look_around)
    return min_index


class InteractivePlot(object):
    """
    interactive plot for plot many resonators iq data
    chan freqs and z should have dimension n_iq points by n_res
    also can provide multiple sweeps for plotting by adding extra dimension
    i.e. chan freqs and z could have dimension n_iq points by n_res by n_sweeps
    combined data should have dimension n_res by n_different_types of data
    """

    def __init__(self, chan_freqs, z, look_around=2, stream_data=None, retune=True, find_min=True,
                 combined_data=None, combined_data_names=None, sweep_labels=None, sweep_line_styles=None,
                 combined_data_format=None):
        if len(z.shape) < 3:  # only one sweep
            self.z = z.reshape((z.shape[0], z.shape[1], 1))
            self.chan_freqs = chan_freqs.reshape((chan_freqs.shape[0], chan_freqs.shape[1], 1))
        else:
            self.z = z
            self.chan_freqs = chan_freqs

        self.Is = np.real(self.z)
        self.Qs = np.imag(self.z)
        self.find_min = find_min
        self.retune = retune
        self.combined_data = combined_data
        self.combined_data_names = combined_data_names
        self.stream_data = stream_data
        self.targ_size = chan_freqs.shape[0]
        self.look_around = look_around
        self.plot_index = 0
        self.combined_data_index = 0
        self.res_index_overide = np.asarray((), dtype=np.int16)
        self.overide_freq_index = np.asarray((), dtype=np.int16)
        self.shift_is_held = False
        self.update_min_index()
        if retune:
            self.combined_data_names = ['min index']

        if self.combined_data is not None:
            self.fig = plt.figure(1, figsize=(13, 10))
            self.ax = self.fig.add_subplot(221)
            self.ax.set_ylabel("Power (dB)")
            self.ax.set_xlabel("Frequecy (MHz)")
            self.ax2 = self.fig.add_subplot(222)
            self.ax2.set_ylabel("Q")
            self.ax2.set_xlabel("I")
            self.ax3 = self.fig.add_subplot(212)
            self.ax3.set_ylabel("")
            self.ax3.set_xlabel("Resonator index")
        else:
            self.fig = plt.figure(1, figsize=(13, 6))
            self.ax = self.fig.add_subplot(121)
            self.ax.set_ylabel("Power (dB)")
            self.ax.set_xlabel("Frequecy (MHz)")
            self.ax2 = self.fig.add_subplot(122)
            self.ax2.set_ylabel("Q")
            self.ax2.set_xlabel("I")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        if self.stream_data is not None:
            self.s2, = self.ax2.plot(np.real(self.stream_data[:, self.plot_index]),
                                     np.imag(self.stream_data[:, self.plot_index]), '.')

        if not sweep_line_styles:
            sweep_line_styles = ["-o"]
            for i in range(1, self.z.shape[2]):
                sweep_line_styles.append("-")
        if not sweep_labels:
            sweep_labels = ["Data 1"]
            for i in range(1, self.z.shape[2]):
                sweep_labels.append("Data " + str(i + 1))
        self._mag_lines = [self.ax.plot(self.chan_freqs[:, self.plot_index, i] / 10 ** 6, 10 * np.log10(
            self.Is[:, self.plot_index, i] ** 2 + self.Qs[:, self.plot_index, i] ** 2), sweep_line_styles[i], mec="k",
                                        label=sweep_labels[i])
                           for i in range(0, self.z.shape[2])]

        self.ax_leg = self.ax.legend()

        self._iq_lines = [self.ax2.plot(
            self.Is[:, self.plot_index, i], self.Qs[:, self.plot_index, i], sweep_line_styles[i], mec="k",
            label=sweep_labels[i])
            for i in range(0, self.z.shape[2])]

        if self.retune:
            self.p1, = self.ax.plot(self.chan_freqs[self.min_index[self.plot_index], self.plot_index, 0] / 10 ** 6,
                                    10 * np.log10(self.Is[self.min_index[self.plot_index], self.plot_index, 0] ** 2 + \
                                                  self.Qs[self.min_index[self.plot_index], self.plot_index, 0] ** 2),
                                    '*', markersize=15)
            self.p2, = self.ax2.plot(self.Is[self.min_index[self.plot_index], self.plot_index, 0],
                                     self.Qs[self.min_index[self.plot_index], self.plot_index, 0], '*', markersize=15)

        self.ax.set_title("Resonator Index " + str(self.plot_index))
        if self.retune:
            self.ax2.set_title("Look Around Points " + str(self.look_around))
        print("")
        print("Interactive Resonance Plotting Activated")
        print("Use left and right arrows to switch between resonators")
        if retune:
            if platform.system() == 'Darwin':
                print("Use the up and down arrows to change look around points")
                print("Hold letter a key and right click on the magnitude plot to overide tone position")
            else:
                print("Use the up and down arrows to change look around points")
                print("Hold shift and right click on the magnitude plot to overide tone position")

        if self.combined_data is not None:
            print("Use the up and down arrows to change data in bottom plot")
            print("double click on point in bottom plot to jump to that resonator index")
            if retune:
                print("Warning both look around points and combined data are mapped to " + \
                      "up and down arrows, consider not returning and ploting combined " + \
                      "data at the same time")
            self.combined_data = np.asarray(self.combined_data)
            if len(self.combined_data.shape) == 1:
                self.combined_data = np.expand_dims(self.combined_data, 1)

            if not combined_data_format:
                self.combined_data_format = []
                for i in range(0, self.combined_data.shape[1]):
                    self.combined_data_format.append(self.combined_data_names[i] + ': {:g}')
            else:
                self.combined_data_format = combined_data_format
            self.ax3.set_title(self.combined_data_names[self.combined_data_index])
            self.ax3.set_ylabel(self.combined_data_names[self.combined_data_index])
            self.combined_data_points, = self.ax3.plot(np.arange(0, self.combined_data.shape[0]),
                                                       self.combined_data[:, self.combined_data_index], 'o', mec="k")
            self.combined_data_highlight, = self.ax3.plot(self.plot_index,
                                                          self.combined_data[self.plot_index, self.combined_data_index],
                                                          'o', mec="k", label=self.combined_data_format[
                    self.combined_data_index].format(
                    self.combined_data[self.plot_index, self.combined_data_index]))
            self.ax3_legend = self.ax3.legend()
        plt.show(block=True)

    def update_min_index(self):
        if self.find_min:
            self.min_index = np.argmin(
                self.Is[self.targ_size // 2 - self.look_around:self.targ_size // 2 + self.look_around, :, 0] ** 2 +
                self.Qs[self.targ_size // 2 - self.look_around:self.targ_size // 2 + self.look_around, :, 0] ** 2,
                axis=0) + (self.targ_size // 2 - self.look_around)
        else:
            self.min_index = find_max_didq(self.z[:, :, 0], self.look_around)
        # handel overidden points
        for i, overide_index in enumerate(self.res_index_overide):
            self.min_index[overide_index] = self.overide_freq_index[i]

        if self.retune:
            self.combined_data = np.expand_dims(self.min_index, 1)

    def refresh_plot(self):
        for i, mag_line in enumerate(self._mag_lines):
            mag_line[0].set_data(self.chan_freqs[:, self.plot_index, i] / 10 ** 6, 10 * np.log10(
                self.Is[:, self.plot_index, i] ** 2 + self.Qs[:, self.plot_index, i] ** 2))
        if self.retune:
            self.p1.set_data(self.chan_freqs[self.min_index[self.plot_index], self.plot_index, 0] / 10 ** 6,
                             10 * np.log10(self.Is[self.min_index[self.plot_index], self.plot_index, 0] ** 2 +
                                           self.Qs[self.min_index[self.plot_index], self.plot_index, 0] ** 2))

        self.ax.relim()
        self.ax.autoscale()
        self.ax.set_title("Resonator Index " + str(self.plot_index))
        if self.retune:
            self.ax2.set_title("Look Around Points " + str(self.look_around))
        for i, iq_line in enumerate(self._iq_lines):
            iq_line[0].set_data((self.Is[:, self.plot_index, i],
                                 self.Qs[:, self.plot_index, i]))
        if self.retune:
            self.p2.set_data(self.Is[self.min_index[self.plot_index], self.plot_index, 0],
                             self.Qs[self.min_index[self.plot_index], self.plot_index, 0])

        if self.stream_data is not None:
            self.s2.set_data(np.real(self.stream_data[:, self.plot_index]),
                             np.imag(self.stream_data[:, self.plot_index]))
        self.ax2.relim()
        self.ax2.autoscale()
        if self.combined_data is not None:
            self.ax3.set_title(self.combined_data_names[self.combined_data_index])
            self.ax3.set_ylabel(self.combined_data_names[self.combined_data_index])
            self.combined_data_points.set_data(np.arange(0, self.combined_data.shape[0]),
                                               self.combined_data[:, self.combined_data_index])
            self.combined_data_highlight.set_data(self.plot_index,
                                                  self.combined_data[self.plot_index, self.combined_data_index])
            self.ax3_legend.texts[0].set_text(self.combined_data_format[self.combined_data_index].format(
                self.combined_data[self.plot_index, self.combined_data_index]))
            self.ax3.relim()
            self.ax3.autoscale()
        plt.draw()

    def on_key_press(self, event):
        # print event.key
        if event.key == 'right':
            if self.plot_index != self.chan_freqs.shape[1] - 1:
                self.plot_index = self.plot_index + 1
                self.refresh_plot()

        if event.key == 'left':
            if self.plot_index != 0:
                self.plot_index = self.plot_index - 1
                self.refresh_plot()

        if event.key == 'up':
            if self.look_around != self.chan_freqs.shape[0] // 2:
                self.look_around = self.look_around + 1
                self.update_min_index()
                if self.retune:
                    self.combined_data = np.expand_dims(self.min_index, 1)
            if self.combined_data is not None:
                if self.combined_data_index != self.combined_data.shape[1] - 1:
                    self.combined_data_index = self.combined_data_index + 1
            self.refresh_plot()

        if event.key == 'down':
            if self.look_around != 1:
                self.look_around = self.look_around - 1
                self.update_min_index()
                if self.retune:
                    self.combined_data = np.expand_dims(self.min_index, 1)
            if self.combined_data is not None:
                if self.combined_data_index != 0:
                    self.combined_data_index = self.combined_data_index + -1
            self.refresh_plot()

        if platform.system().lower() == 'darwin':
            if event.key == 'a':
                self.shift_is_held = True
        else:
            if event.key == 'shift':
                self.shift_is_held = True

    def on_key_release(self, event):
        # windows or mac
        if platform.system() == 'Darwin':
            if event.key == 'a':
                self.shift_is_held = False
        else:
            if event.key == 'shift':
                self.shift_is_held = False

    def onClick(self, event):
        if self.combined_data is not None:
            if event.dblclick:
                self.plot_index = np.argmin(np.abs(np.arange(0, self.combined_data.shape[0]) - event.xdata))
                self.refresh_plot()
                return
        if event.button == 3:
            if self.shift_is_held:
                print("overiding point selection", event.xdata)
                # print(self.chan_freqs[:,self.plot_index][50])
                # print((self.res_index_overide == self.plot_index).any())
                if (self.res_index_overide == self.plot_index).any():
                    replace_index = np.argwhere(
                        self.res_index_overide == self.plot_index)[0][0]
                    new_freq = np.argmin(
                        np.abs(event.xdata - self.chan_freqs[:, self.plot_index] / 10 ** 6))
                    self.overide_freq_index[replace_index] = np.int(new_freq)

                else:
                    self.res_index_overide = np.append(
                        self.res_index_overide, np.int(np.asarray(self.plot_index)))
                    # print(self.res_index_overide)
                    new_freq = np.argmin(
                        np.abs(event.xdata - self.chan_freqs[:, self.plot_index] / 10 ** 6))
                    # print("new index is ",new_freq)
                    self.overide_freq_index = np.append(
                        self.overide_freq_index, np.int(np.asarray(new_freq)))
                    # print(self.overide_freq_index)
                self.update_min_index()
                self.refresh_plot()


def tune_kids(f, z, find_min=True, interactive=True, **kwargs):
    # f and z should have shape (npts_sweep,n_res)
    # iq_dict = read_iq_sweep(filename)
    if "look_around" in kwargs:
        print("you are using " +
              str(kwargs['look_around']) + " look around points")
        look_around = kwargs['look_around']
    else:
        look_around = f.shape[0] // 2
    if find_min:  # fine the minimum
        print("centering on minimum")
        print(f.shape)
        if interactive:
            ip = InteractivePlot(f, z, look_around)
            print(ip.min_index.shape)
            print(ip.res_index_overide.shape)
            for i in range(0, len(ip.res_index_overide)):
                ip.min_index[ip.res_index_overide[i]
                ] = ip.overide_freq_index[i]
            new_freqs = f[(ip.min_index, np.arange(0, f.shape[1]))]
        else:
            min_index = np.argmin(np.abs(z) ** 2, axis=0)
            new_freqs = f[(min_index, np.arange(0, f.shape[1]))]
    else:  # find the max of dIdQ
        print("centering on max dIdQ")
        if interactive:
            ip = InteractivePlot(f, z, look_around, find_min=False)
            for i in range(0, len(ip.res_index_overide)):
                ip.min_index[ip.res_index_overide[i]
                ] = ip.overide_freq_index[i]
            new_freqs = f[(ip.min_index, np.arange(0, f.shape[1]))]
        else:
            min_index = find_max_didq(z, look_around)
            new_freqs = f[(min_index, np.arange(0, f.shape[1]))]
    return new_freqs


class interactive_power_tuning_plot(object):
    """
    special interactive plot for tune the readout power of resonators based on fist to thier non-linearity parameter
    f (frequencies of iq_sweep shape n_pts_iq_sweep x n_res x (optionally n_powers)
    z complex number where i is real and q is imaginary part shape n_pts_iq_sweep x n_res,  n_powers
    fitted_a_mag from magnitude fits
    fitted_a_iq from iq fits (one of two fitted as must be provided
    attn_levels list or array of power levels
    desired_a the desired non-linearity paramter to have powers set to
    Optional
    z_fit_mag data to display as the fit 
    z_fit_iq data to display as the fit
    """

    def __init__(self, f, z, attn_levels, fitted_a_mag=None, fitted_a_iq=None, desired_a=0.5, z_fit_mag=None,
                 z_fit_iq=None):
        self.attn_levels = np.asarray(attn_levels)
        self.bif_level = desired_a
        self.z = z
        self.Is = np.real(z)
        self.Qs = np.imag(z)
        self.z_fit_mag = z_fit_mag
        self.z_fit_iq = z_fit_iq
        if self.z_fit_iq is not None:
            self.Is_fit_iq = np.real(z_fit_iq)
            self.Qs_fit_iq = np.imag(z_fit_iq)
        self.fitted_a_mag = fitted_a_mag
        self.fitted_a_iq = fitted_a_iq
        if len(f.shape) > 2:
            self.f = f
        else:
            self.f = np.zeros(self.z.shape)
            for k in range(0, self.z.shape[1]):
                self.f[:, k, :] = self.f
        self.chan_freqs = f
        # self.targ_size = self.chan_freqs.shape[0]
        self.plot_index = 0
        self.power_index = 0
        self.res_index_overide = np.asarray((), dtype=np.int16)
        self.overide_freq_index = np.asarray((), dtype=np.int16)
        self.shift_is_held = False
        if fitted_a_mag is not None:
            self.bif_levels_mag = np.zeros(fitted_a_mag.shape[0])
        if fitted_a_iq is not None:
            self.bif_levels_iq = np.zeros(fitted_a_iq.shape[0])
        # here we compute our best guess for the proper power level from the fit data
        for i in range(0, self.f.shape[1]):
            if fitted_a_mag is not None:
                try:
                    first_bif_mag = np.where(fitted_a_mag[i, :] > bif_level)[0][0]
                except:
                    first_bif_mag = fitted_a_mag.shape[1] - 1
                if first_bif_mag == 0:
                    first_bif_mag = 1
                interp_mag = interpolate.interp1d(-self.attn_levels[0:first_bif_mag + 1],
                                                  fitted_a_mag[i, :][0:first_bif_mag + 1])
                powers_mag = np.linspace(-self.attn_levels[0] + .01, -self.attn_levels[first_bif_mag] - 0.01, 1000)
                bifurcation_mag = powers_mag[
                    np.argmin(np.abs(interp_mag(powers_mag) - np.ones(len(powers_mag)) * self.bif_level))]
                self.bif_levels_mag[i] = bifurcation_mag
            if fitted_a_iq is not None:
                try:
                    first_bif_iq = np.where(fitted_a_iq[i, :] > bif_level)[0][0]
                except:
                    first_bif_iq = self.fitted_a_iq.shape[1] - 1
                if first_bif_iq == 0:
                    first_bif_iq = 1
                interp_iq = interpolate.interp1d(-self.attn_levels[0:first_bif_iq + 1],
                                                 fitted_a_iq[i, :][0:first_bif_iq + 1])
                powers_iq = np.linspace(-self.attn_levels[0] + .01, -self.attn_levels[first_bif_iq] - 0.01, 1000)
                bifurcation_iq = powers_iq[
                    np.argmin(np.abs(interp_iq(powers_iq) - np.ones(len(powers_iq)) * self.bif_level))]
                self.bif_levels_iq[i] = bifurcation_iq

        if self.fitted_a_mag is not None:
            self.bif_levels = copy.deepcopy(self.bif_levels_mag)
        else:
            try:
                self.bif_levels = copy.deepcopy(self.bif_levels_iq)
            except:
                print("you must supply fits to the non-linearity paramter one of two fit variables")
                return
        self.power_index = np.argmin(np.abs(self.bif_levels[self.plot_index] + self.attn_levels))
        self.fig = plt.figure(1, figsize=(13, 10))
        self.ax = self.fig.add_subplot(221)
        self.ax.set_ylabel("Power (dB)")
        self.ax.set_xlabel("Frequecy (MHz)")
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_ylabel("Q")
        self.ax2.set_xlabel("I")
        self.ax3 = self.fig.add_subplot(212)
        self.ax3.set_xlabel("Power level")
        self.ax3.set_ylabel("Nolinearity parameter a")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        if z_fit_mag is not None:
            self.l1_fit, = self.ax.plot(self.chan_freqs[:, self.plot_index, self.power_index],
                                        10 * np.log10(self.z_fit_mag[:, self.plot_index, self.power_index]),
                                        label="Fit")
        self.l1, = self.ax.plot(self.chan_freqs[:, self.plot_index, self.power_index],
                                20 * np.log10(np.abs(self.z[:, self.plot_index, self.power_index])), 'o', mec="k",
                                label="Data")
        self.ax.legend()
        if self.z_fit_iq is not None:
            self.l2_fit, = self.ax2.plot(self.Is_fit_iq[:, self.plot_index, self.power_index],
                                         Qs_fit_iq[:, self.plot_index, self.power_index], label="Fit")
        self.l2, = self.ax2.plot(self.Is[:, self.plot_index, self.power_index],
                                 self.Qs[:, self.plot_index, self.power_index],
                                 'o', mec="k", label="Data")
        self.ax2.legend()
        if self.fitted_a_mag is not None:
            self.l3, = self.ax3.plot(-self.attn_levels, self.fitted_a_mag[self.plot_index, :], color='r',
                                     label="Mag fit")
        if self.fitted_a_iq is not None:
            self.l4, = self.ax3.plot(-self.attn_levels, self.fitted_a_iq[self.plot_index, :], color='g', label="IQ fit")
        self.l8, = self.ax3.plot((self.bif_levels[self.plot_index], self.bif_levels[self.plot_index]), (0, 1),
                                 "--", color='m', linewidth=3, label="Power pick")
        self.l5, = self.ax3.plot((-self.attn_levels[self.power_index], -self.attn_levels[self.power_index]),
                                 (0, 1), "--", color='k', label="Current power")
        if self.fitted_a_mag is not None:
            self.l6, = self.ax3.plot((self.bif_levels_mag[self.plot_index], self.bif_levels_mag[self.plot_index]),
                                     (0, self.bif_level), "--", color='r')
        if self.fitted_a_iq is not None:
            self.l7, = self.ax3.plot((self.bif_levels_iq[self.plot_index], self.bif_levels_iq[self.plot_index]),
                                     (0, self.bif_level), "--", color='g')
        self.ax3.legend()
        self.ax3.set_ylim(0, 1)
        self.ax.set_title("Resonator index " + str(self.plot_index))
        self.ax2.set_title("Power level " + str(-self.attn_levels[self.power_index]))
        print("")
        print("Interactive Power Tuning Activated")
        print("Use left and right arrows to switch between resonators")
        print("Use the up and down arrows to change between power levels")
        print("Hold shift and right click on the bottom plot to overide picked power level")
        print("or hold shift and press enter to overide picked power level to the current plotted power level")
        plt.show(block=True)

    def refresh_plot(self):
        self.l1.set_data(self.chan_freqs[:, self.plot_index, self.power_index],
                         10 * np.log10(self.Is[:, self.plot_index, self.power_index] ** 2 + self.Qs[:, self.plot_index,
                                                                                            self.power_index] ** 2))
        if self.z_fit_mag is not None:
            self.l1_fit.set_data(self.chan_freqs[:, self.plot_index, self.power_index],
                                 10 * np.log10(self.z_fit_mag[:, self.plot_index, self.power_index]))
        self.ax.relim()
        self.ax.autoscale()
        self.ax.set_title("Resonator index " + str(self.plot_index))
        self.ax2.set_title("Power level " + str(-self.attn_levels[self.power_index]))
        self.l2.set_data((self.Is[:, self.plot_index, self.power_index], self.Qs[:, self.plot_index, self.power_index]))
        if self.z_fit_iq is not None:
            self.l2_fit.set_data((self.Is_fit_iq[:, self.plot_index, self.power_index],
                                  self.Qs_fit_iq[:, self.plot_index, self.power_index]))
        if self.fitted_a_mag is not None:
            self.l3.set_data(-self.attn_levels, self.fitted_a_mag[self.plot_index, :])
        self.l4.set_data(-self.attn_levels, self.fitted_a_iq[self.plot_index, :])
        self.l5.set_data((-self.attn_levels[self.power_index], -self.attn_levels[self.power_index]), (0, 1))
        if self.fitted_a_mag is not None:
            self.l6.set_data((self.bif_levels_mag[self.plot_index], self.bif_levels_mag[self.plot_index]),
                             (0, self.bif_level))
        if self.fitted_a_iq is not None:
            self.l7.set_data((self.bif_levels_iq[self.plot_index], self.bif_levels_iq[self.plot_index]),
                             (0, self.bif_level))

        self.l8.set_data((self.bif_levels[self.plot_index], self.bif_levels[self.plot_index]), (0, 1))
        self.ax2.relim()
        self.ax2.autoscale()
        plt.draw()

    def on_key_press(self, event):
        # print(event.key) #for debugging
        if event.key == 'right':
            if self.plot_index != self.chan_freqs.shape[1] - 1:
                self.plot_index = self.plot_index + 1
                # snap to the automated choice in power level
                self.power_index = np.argmin(np.abs(self.bif_levels[self.plot_index] + self.attn_levels))
                self.refresh_plot()

        if event.key == 'left':
            if self.plot_index != 0:
                self.plot_index = self.plot_index - 1
                # snap to the automated choice in power level
                self.power_index = np.argmin(np.abs(self.bif_levels[self.plot_index] + self.attn_levels))
                self.refresh_plot()

        if event.key == 'up':
            if self.power_index != self.Is.shape[2] - 1:
                self.power_index = self.power_index + 1
            self.refresh_plot()

        if event.key == 'down':
            if self.power_index != 0:
                self.power_index = self.power_index - 1
            self.refresh_plot()

        if event.key == 'shift':
            self.shift_is_held = True
            # print("shift pressed") #for debugging

        if event.key == 'enter':
            if self.shift_is_held:
                self.bif_levels[self.plot_index] = -self.attn_levels[self.power_index]
                self.refresh_plot()

    def on_key_release(self, event):
        if event.key == "shift":
            self.shift_is_held = False
            # print("shift released") #for debugging

    def onClick(self, event):
        if event.button == 3:
            if self.shift_is_held:
                print("overiding point selection", event.xdata)
                self.bif_levels[self.plot_index] = event.xdata
                self.refresh_plot()


def tune_resonance_power(f, z, attn_levels, fitted_a_mag=None, fitted_a_iq=None, desired_a=0.5, z_fit_mag=None,
                         z_fit_iq=None):
    # Call the interactive plot
    ip = interactive_power_tuning_plot(f, z, attn_levels, fitted_a_mag=fitted_a_mag,
                                       fitted_a_iq=fitted_a_iq, desired_a=desired_a,
                                       z_fit_mag=z_fit_mag, z_fit_iq=z_fit_iq)

    picked_power_levels = ip.bif_levels + attn_levels[-1]
    # plot the resultant picked power levels and the corresponding transfer function
    plt.figure(1, figsize=(12, 10))
    plt.subplot(211)
    plt.title("Bifurcation power")
    plt.plot(picked_power_levels, 'o', color='g', label="Picked Valuse")
    plt.xlabel("Resonator index")
    plt.ylabel("Power at a = " + str(desired_a))
    plt.legend(loc=4)

    normalizing_amplitudes = np.sqrt(
        10 ** ((ip.bif_levels + attn_levels[-1]) / 10) / np.min(10 ** ((ip.bif_levels + attn_levels[-1]) / 10)))
    plt.subplot(212)
    plt.title("Transfer function")
    plt.plot(normalizing_amplitudes, 'o')
    plt.xlabel("Resonator index")
    plt.ylabel("Voltage factor")

    # plt.savefig("power_sweep_results_"+fine_names[0].split('/')[-1]+".pdf")

    # save the output
    # np.savetxt("bif_levels_mag_"+fine_names[0].split('/')[-1]+".csv",ip.bif_levels_mag+attn_levels[-1])
    # np.savetxt("bif_levels_iq_"+fine_names[0].split('/')[-1]+".csv",ip.bif_levels_iq+attn_levels[-1])
    # np.save("trans_func_"+fine_names[0].split('/')[-1],np.sqrt(10**((ip.bif_levels+attn_levels[-1])/10)/np.min(10**((ip.bif_levels+attn_levels[-1])/10))))

    plt.show()
    return picked_power_levels, normalizing_amplitudes

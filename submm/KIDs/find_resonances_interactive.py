import os
import platform
from typing import NamedTuple
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from scipy import signal, fftpack
from matplotlib.backends.backend_pdf import PdfPages
try:
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook() #input() breaks during interactive plot on linux without this if using Qt backend
except:
    pass
from submm.KIDs.res.fitting import fit_nonlinear_iq, fit_nonlinear_mag
from submm.KIDs.res.utils import colorize_text, text_color_matplotlib#, autoscale_from_data

"""
Standalone version of kidPy's find_KIDs_interactive
use fin

If you have already identified most of the resonators at indexes kid_idx
just call the interactive plot object like so
ip = InteractivePlot(f,20*np.log10(np.abs(z)),kid_idx)

if you want to use use the filtering and threshold resonator finder
call find_vna_sweep(f,z) like
ip = find_vna_sweep(f,z)
get the kid indexes out from ip.kid_idx
get the frequencies out from f[ip.kid_idx]

"""


def open_stored_sweep(savepath, load_std=False):
    """Opens sweep data
       inputs:
           char savepath: The absolute path where sweep data is saved
       ouputs:
           numpy array Is: The I values
           numpy array Qs: The Q values"""
    files = sorted(os.listdir(savepath))
    I_list, Q_list, stdI_list, stdQ_list = [], [], [], []
    for filename in files:
        if filename.startswith('I'):
            I_list.append(os.path.join(savepath, filename))
        if filename.startswith('Q'):
            Q_list.append(os.path.join(savepath, filename))
        if filename.startswith('stdI'):
            stdI_list.append(os.path.join(savepath, filename))
        if filename.startswith('stdQ'):
            stdQ_list.append(os.path.join(savepath, filename))
    Is = np.array([np.load(filename) for filename in I_list])
    Qs = np.array([np.load(filename) for filename in Q_list])
    if len(stdI_list) > 0:
        std_Is = np.array([np.load(filename) for filename in stdI_list])
        std_Qs = np.array([np.load(filename) for filename in stdQ_list])
    if load_std:
        return Is, Qs, std_Is, std_Qs
    else:
        return Is, Qs


class SingleWindow(NamedTuple):
    left_max: int
    left_fitter_pad: int
    left_pad: int
    left_window: int
    minima: int
    right_window: int
    right_pad: int
    right_fitter_pad: int
    right_max: int


class InteractivePlot(object):
    """
    Convention is to supply the data in magnitude units i.e. 20*np.log10(np.abs(z))
    frequencies should be supplied in Hz
    """

    def __init__(self, chan_freqs, data, kid_idx, flags=None, f_old=None, data_old=None, kid_idx_old=None):
        plt.rcParams['keymap.forward'] = ['v']
        plt.rcParams['keymap.back'] = ['c', 'backspace']  # remove arrows from back and forward on plot
        plt.rcParams['keymap.quit'] = ['k']  # remove q for quit make it k for kill
        plt.rcParams['keymap.home'] = ['h']  # remove r for home only make it h
        plt.rcParams['keymap.fullscreen'] = ['shift+=']  # remove ('f', 'ctrl+f'), make +

        # set up plot
        self.key_font_size = 9
        top = 0.90
        bottom = 0.1
        left = 0.08
        right = 0.99
        x_width = right - left
        y_height= top - bottom
        key_x_width = 0.2
        self.x_width_over_y_height = 1.0

        self.chan_freqs = chan_freqs
        self.data = data
        self.f_old = f_old
        self.data_old = data_old
        self.kid_idx_old = kid_idx_old
        print(kid_idx)
        self.kid_idx = kid_idx
        if flags is None:
            self.flags = []
            for i in range(len(self.kid_idx)): self.flags.append([])
        self.lim_shift_factor = 0.2
        self.zoom_factor = 0.1  # no greater than 0.5
        self.kid_idx_len = len(kid_idx)

        self.fig = plt.figure(2, figsize=(16, 6))
        #self.ax = self.fig.add_subplot(111)
        x_width_plot = x_width - key_x_width
        figure_coords = [left, bottom, x_width_plot, y_height]
        self.ax = self.fig.add_axes(figure_coords, frameon=False, autoscale_on=True)
        key_y_height = 0.8#y_height
        key_figure_coords = [left + x_width_plot, top-key_y_height, key_x_width, key_y_height]
        self.ax_key = self.fig.add_axes(key_figure_coords, frameon=False)
        self.ax_key.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        self.ax_key.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
     
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.l1, = self.ax.plot(self.chan_freqs / 10 ** 9, self.data)
        self.flagged_indexes = self.get_flagged_indexes()
        self.fp1, = self.ax.plot(self.chan_freqs[self.flagged_indexes] / 10 ** 9, self.data[self.flagged_indexes], "yo",
                                 markersize=9)
        self.p1, = self.ax.plot(self.chan_freqs[self.kid_idx] / 10 ** 9, self.data[self.kid_idx], "r*", markersize=8)
        self.text_dict = {}
        for i in range(0, len(self.kid_idx)):
            self.text_dict[i] = self.ax.text(self.chan_freqs[self.kid_idx][i] / 10 ** 9, self.data[self.kid_idx][i], str(i))

        if isinstance(self.f_old, np.ndarray):
            self.l2, = self.ax.plot(self.f_old / 10 ** 9, self.data_old, color="C0", alpha=0.25)
            self.p2, = self.ax.plot(self.f_old[self.kid_idx_old] / 10 ** 9, self.data_old[self.kid_idx_old], "r*",
                                    markersize=8,
                                    alpha=0.1)
            self.text_dict_old = {}
            for i in range(0, len(self.kid_idx_old)):
                self.text_dict_old[i] = self.ax.text(self.f_old[self.kid_idx_old][i] / 10 ** 9,
                                                 self.data_old[self.kid_idx_old][i],
                                                 str(i), color='Grey')

        self.shift_is_held = False
        self.control_is_held = False
        self.f_is_held = False
        self.add_list = []
        self.delete_list = []
        print("The controls are:")
        if platform.system() == 'Darwin':
            print("Hold the a key while right clicking to add points")
            print("Hold the d key while right clicking to delete points")
        else:
            print("Hold the shift key while right clicking to add points")
            print("Hold the control key while right clicking to delete points")
        print("Hold the f key while right clicking to flag points")
        print("Use the arrow keys to pan around plot")
        print("Use z to zoom")
        print("Use x to Xplode")
        print("Use q to zoom in x axis")
        print("Use w go Xplode in x axis")
        print("Use e to zoom in y axis")
        print("Use r go Xplode in y axis")
        print("Close all plots when finished")
        self.ax.set_xlabel('Frequency (GHz)')
        self.ax.set_ylabel('Power (dB)')
        self.plot_instructions()
        plt.show(block=True)

    def instructions(self):
        if platform.system() == 'Darwin':
            instructions = [("left-arrow", "move left", 'green'),
                        ("right-arrow", "move right", 'cyan'),
                        ("up-arrow", "move up", 'yellow'),
                        ("down-arrow", "move down", 'red'),
                        ("Z-key", "zoom", 'purple'),
                        ("X-key", "Xplode", 'blue'),
                        ("Q-key", "zoom x axis", 'black'),
                        ("W-key", "Xplode x axis", 'white'),
                        ("E-key", "Zoom y axis", 'green'),
                        ("R-key", "Xplode y axis", 'cyan'),
                        ("D+right-click", "delete points", 'yellow'),
                        ("A+right-click", "add points", 'red'),
                        ("double-click", "flag points", 'purple')]
        else:
            instructions = [("left-arrow", "move left", 'green'),
                        ("right-arrow", "move right", 'cyan'),
                        ("up-arrow", "move up", 'yellow'),
                        ("down-arrow", "move down", 'red'),
                        ("Z-key", "zoom", 'purple'),
                        ("X-key", "Xplode", 'blue'),
                        ("Q-key", "zoom x axis", 'black'),
                        ("W-key", "Xplode x axis", 'white'),
                        ("E-key", "Zoom y axis", 'green'),
                        ("R-key", "Xplode y axis", 'cyan'),
                        ("shift+right-click", "delete points", 'yellow'),
                        ("ctrl+right-click", "add points", 'red'),
                        ("double-click", "flag points", 'purple')]
        return instructions

    def plot_instructions(self):
        instructions = self.instructions()
        self.ax_key.clear()
        steps_per_item = 1.5
        y_step = 0.9 / (steps_per_item * float(len(instructions)))
        y_now = 0.9
        for key_press, description, color in instructions:
            self.ax_key.text(0.4, y_now, key_press.center(17), color=text_color_matplotlib[color],
                             ha='right', va='center', size=self.key_font_size, weight="bold",
                             family='monospace', bbox=dict(color=color, ls='-', lw=2.0, ec='black'))
            self.ax_key.text(0.45, y_now, description, color='black',
                             ha='left', va='center', size=self.key_font_size - 2)
            y_now -= steps_per_item * y_step
        #self.ax_key.set_xlim(0, 1)
        #self.ax_key.set_ylim(0, 1)
        self.ax_key.set_title('Main Menu')
        plt.draw()

    def on_key_press(self, event):
        # mac or windows
        if platform.system().lower() == 'darwin':
            if event.key == 'a':
                self.shift_is_held = True
            if event.key == 'd':
                self.control_is_held = True
        else:
            if event.key == 'shift':
                self.shift_is_held = True
            if event.key == 'control':
                self.control_is_held = True

        if event.key == 'f':
            self.f_is_held = True

        if event.key == 'right':  # pan right
            xlim_left, xlim_right = self.ax.get_xlim()
            xlim_size = xlim_right - xlim_left
            self.ax.set_xlim(xlim_left + self.lim_shift_factor * xlim_size,
                             xlim_right + self.lim_shift_factor * xlim_size)
            plt.draw()

        if event.key == 'left':  # pan left
            xlim_left, xlim_right = self.ax.get_xlim()
            xlim_size = xlim_right - xlim_left
            self.ax.set_xlim(xlim_left - self.lim_shift_factor * xlim_size,
                             xlim_right - self.lim_shift_factor * xlim_size)
            plt.draw()

        if event.key == 'up':  # pan up
            ylim_left, ylim_right = self.ax.get_ylim()
            ylim_size = ylim_right - ylim_left
            self.ax.set_ylim(ylim_left + self.lim_shift_factor * ylim_size,
                             ylim_right + self.lim_shift_factor * ylim_size)
            plt.draw()

        if event.key == 'down':  # pan down
            ylim_left, ylim_right = self.ax.get_ylim()
            ylim_size = ylim_right - ylim_left
            self.ax.set_ylim(ylim_left - self.lim_shift_factor * ylim_size,
                             ylim_right - self.lim_shift_factor * ylim_size)
            plt.draw()

        if event.key == 'z':  # zoom in
            xlim_left, xlim_right = self.ax.get_xlim()
            ylim_left, ylim_right = self.ax.get_ylim()
            xlim_size = xlim_right - xlim_left
            ylim_size = ylim_right - ylim_left
            self.ax.set_xlim(xlim_left + self.zoom_factor * xlim_size, xlim_right - self.zoom_factor * xlim_size)
            self.ax.set_ylim(ylim_left + self.zoom_factor * ylim_size, ylim_right - self.zoom_factor * ylim_size)
            plt.draw()

        if event.key == 'x':  # zoom out
            xlim_left, xlim_right = self.ax.get_xlim()
            ylim_left, ylim_right = self.ax.get_ylim()
            xlim_size = xlim_right - xlim_left
            ylim_size = ylim_right - ylim_left
            self.ax.set_xlim(xlim_left - self.zoom_factor * xlim_size, xlim_right + self.zoom_factor * xlim_size)
            self.ax.set_ylim(ylim_left - self.zoom_factor * ylim_size, ylim_right + self.zoom_factor * ylim_size)
            plt.draw()

        if event.key == 'q':  # zoom in x axis only
            xlim_left, xlim_right = self.ax.get_xlim()
            xlim_size = xlim_right - xlim_left
            self.ax.set_xlim(xlim_left + self.zoom_factor * xlim_size, xlim_right - self.zoom_factor * xlim_size)
            plt.draw()

        if event.key == 'w':  # zoom out x axis only
            xlim_left, xlim_right = self.ax.get_xlim()
            xlim_size = xlim_right - xlim_left
            self.ax.set_xlim(xlim_left - self.zoom_factor * xlim_size, xlim_right + self.zoom_factor * xlim_size)
            plt.draw()

        if event.key == 'e':  # zoom in y axis only
            ylim_left, ylim_right = self.ax.get_ylim()
            ylim_size = ylim_right - ylim_left
            self.ax.set_ylim(ylim_left + self.zoom_factor * ylim_size, ylim_right - self.zoom_factor * ylim_size)
            plt.draw()

        if event.key == 'r':  # zoom out y axis only
            ylim_left, ylim_right = self.ax.get_ylim()
            ylim_size = ylim_right - ylim_left
            self.ax.set_ylim(ylim_left - self.zoom_factor * ylim_size, ylim_right + self.zoom_factor * ylim_size)
            plt.draw()

    def on_key_release(self, event):
        # windows or mac
        if platform.system() == 'Darwin':
            if event.key == 'a':
                self.shift_is_held = False
            if event.key == 'd':
                self.control_is_held = False
        else:
            if event.key == 'shift':
                self.shift_is_held = False
            if event.key == 'control':
                self.control_is_held = False


    def onClick(self, event):
        if event.button == 3:
            if self.shift_is_held:  # add point
                print("adding point", event.xdata)
                self.kid_idx.append(np.argmin(np.abs(self.chan_freqs - event.xdata * 10 ** 9)))
                self.flags.append([])
                zipped_lists = zip(self.kid_idx, self.flags)
                sorted_pairs = sorted(zipped_lists)
                tuples = zip(*sorted_pairs)
                self.kid_idx, self.flags = [list(tuple) for tuple in tuples]
                # self.kid_idx.sort()
                self.refresh_plot()
            elif self.control_is_held:  # delete point
                print("removing point", event.xdata)
                delete_index = np.argmin(np.abs(self.chan_freqs[self.kid_idx] - event.xdata * 10 ** 9))
                self.kid_idx.pop(delete_index)
                self.flags.pop(delete_index)
                self.refresh_plot()
            else:
                print("please hold either the shift or control key while right clicking to add or remove points")
        elif event.dblclick:
            x_data_coords = self.chan_freqs[self.kid_idx]/10**9 - event.xdata
            x_data_min, x_data_max = self.ax.get_xlim()
            x_data_range = x_data_max - x_data_min
            x_norm_coords = x_data_coords / x_data_range
            x_yratio_coords = x_norm_coords * self.x_width_over_y_height
            y_data_coords = self.data[self.kid_idx] - event.ydata
            y_data_min, y_data_max = self.ax.get_ylim()
            y_data_range = y_data_max - y_data_min
            y_norm_coords = y_data_coords / y_data_range
            radius_array = np.sqrt(x_yratio_coords ** 2 + y_norm_coords ** 2)
            flag_index = np.argmin(radius_array)
            #print("flagging point", event.xdata)
            #flag_index = np.argmin(np.abs(self.chan_freqs[self.kid_idx] - event.xdata * 10 ** 9))
            print("current flags: ", self.flags[flag_index])
            pop_up = PopUpDataEntry("Enter flag string","collision")
            flag = pop_up.value.lower()
            if flag == "c":
                flag = "collision"
            elif flag == "s":
                flag = "shallow"
            else:
                pass
            self.flags[flag_index].append(flag)
            print("Flags are now: ", self.flags[flag_index])
            self.refresh_plot()


    def refresh_plot(self):
        self.flagged_indexes = self.get_flagged_indexes()
        self.fp1.set_data(self.chan_freqs[self.flagged_indexes] / 10 ** 9, self.data[self.flagged_indexes])
        self.p1.set_data(self.chan_freqs[self.kid_idx] / 10 ** 9, self.data[self.kid_idx])
        for i in range(0, self.kid_idx_len):
            self.text_dict[i].set_text("")  # clear all of the texts
        self.text_dict = {}
        for i in range(0, len(self.kid_idx)):
            self.text_dict[i] = self.ax.text(self.chan_freqs[self.kid_idx][i] / 10 ** 9, self.data[self.kid_idx][i], str(i))
        self.kid_idx_len = len(self.kid_idx)
        plt.draw()

    def get_flagged_indexes(self):
        flags_empty = True
        for flag_list in self.flags:
            if not flag_list:
                pass
            else:
                flags_empty = False
        if not flags_empty:
            zipped = zip(self.kid_idx, self.flags)
            pairs_with_flags = [pair for pair in zipped if len(pair[1]) > 0]
            tuples = zip(*pairs_with_flags)
            flagged_indexes, just_flags = [list(tuple) for tuple in tuples]
            return flagged_indexes
        else:
            return []






class InteractiveThresholdPlot(object):
    def __init__(self, f_Hz, s21_mag, peak_threshold_dB, spacing_threshold_Hz=1.0e5,
                 window_pad_factor=1.2, fitter_pad_factor=5.0, debug_mode=False):
        plt.rcParams['keymap.forward'] = ['v']
        plt.rcParams['keymap.back'] = ['c', 'backspace']  # remove arrows from back and forward on plot
        self.peak_threshold_dB = peak_threshold_dB
        self.spacing_threshold_Hz = spacing_threshold_Hz

        # set up plot
        self.key_font_size = 9
        top = 0.90
        bottom = 0.1
        left = 0.08
        right = 0.99
        x_width = right - left
        y_height= top - bottom
        key_x_width = 0.2

        self.window_pad_factor = window_pad_factor
        self.fitter_pad_factor = fitter_pad_factor
        self.f_Hz = f_Hz
        self.spacing_threshold_Q = np.mean(self.f_Hz) / self.spacing_threshold_Hz
        self.f_GHz = f_Hz * 1.0e-9
        self.s21_mag = s21_mag

        self.regions = None
        self.ilo = None
        self.local_minima = None
        self.minima_as_windows = None
        self.calc_regions()

        if not debug_mode:
            self.fig = plt.figure(2, figsize=(16, 6))

            x_width_plot = x_width - key_x_width
            figure_coords = [left, bottom, x_width_plot, y_height]
            self.ax = self.fig.add_axes(figure_coords, frameon=False, autoscale_on=True)
            key_y_height = 0.25#y_height
            key_figure_coords = [left + x_width_plot, top-key_y_height, key_x_width, key_y_height]
            self.ax_key = self.fig.add_axes(key_figure_coords, frameon=False)
            self.ax_key.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            self.ax_key.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.l1, = self.ax.plot(self.f_GHz, self.s21_mag)

            self.p1, = self.ax.plot(self.f_GHz[self.ilo], self.s21_mag[self.ilo], "r*")
            self.p2, = self.ax.plot(self.f_GHz[self.local_minima], self.s21_mag[self.local_minima], "b*")
            print("Press up or down to change the threshold by 0.1 dB or press t to enter a custom threshold value.")
            print("Press left or right to change the spacing threshold or y to enter a custom threshold value.")
            print("Close all plots when finished")
            self.ax.set_xlabel(
                F"Frequency (GHz)\n Spacing threshold {'%.2E' % self.spacing_threshold_Hz} Hz or Q~{'%.2E' % self.spacing_threshold_Q}")
            plt.ylabel('Power (dB)')
            self.ax.set_title(
                F"Threshold: 3 adjacent points under {'%2.2f' % self.peak_threshold_dB} dB.\n Found {'%.0f' % len(self.local_minima)} minima")

            self.plot_instructions()
            plt.show(block=True)

    def instructions(self):
        instructions = [("left-arrow", "decrease collision spacing", 'green'),
                        ("right-arrow", "increase collision spacing", 'cyan'),
                        ("up-arrow", "increase threshold by 0.1 dB", 'yellow'),
                        ("down-arrow", "decrease threshold by 0.1 dB", 'red'),
                        ("T-key", "custom dB threshold", 'purple'),
                        ("Y-key", "custom collision spacing", 'blue')]

        return instructions

    def plot_instructions(self):
        instructions = self.instructions()
        self.ax_key.clear()
        steps_per_item = 1.5
        y_step = 0.9 / (steps_per_item * float(len(instructions)))
        y_now = 0.9
        for key_press, description, color in instructions:
            self.ax_key.text(0.4, y_now, key_press.center(13), color=text_color_matplotlib[color],
                             ha='right', va='center', size=self.key_font_size, weight="bold",
                             family='monospace', bbox=dict(color=color, ls='-', lw=2.0, ec='black'))
            self.ax_key.text(0.45, y_now, description, color='black',
                             ha='left', va='center', size=self.key_font_size - 2)
            y_now -= steps_per_item * y_step
        #self.ax_key.set_xlim(0, 1)
        #self.ax_key.set_ylim(0, 1)
        self.ax_key.set_title('Main Menu')
        plt.draw()

    def on_key_press(self, event):
        # print event.key
        # has to be shifted and ctrl because remote viewers only forward
        # certain key combinations
        # print event.key == 'd'
        if event.key == 'up':
            self.peak_threshold_dB = self.peak_threshold_dB + 0.1
            self.refresh_plot()
        if event.key == 'down':
            self.peak_threshold_dB = self.peak_threshold_dB - 0.1
            self.refresh_plot()
        if event.key == 'left':
            self.spacing_threshold_Hz = self.spacing_threshold_Hz / 1.4
            self.spacing_threshold_Q = np.mean(self.f_Hz) / self.spacing_threshold_Hz
            self.refresh_plot()
        if event.key == 'right':
            self.spacing_threshold_Hz = self.spacing_threshold_Hz * 1.4
            self.spacing_threshold_Q = np.mean(self.f_Hz) / self.spacing_threshold_Hz
            self.refresh_plot()
        if event.key == 't':
            pop_up = PopUpDataEntry("Enter threshold in dB","3")
            self.peak_threshold_dB =  float(pop_up.value)
            self.refresh_plot()
        if event.key == 'y':
            pop_up = PopUpDataEntry("Enter Spacing threshold in Hz","100000")
            self.spacing_threshold_Hz = float(pop_up.value)
            self.refresh_plot()

    def refresh_plot(self):
        self.calc_regions()
        self.p1.set_data(self.f_GHz[self.ilo], self.s21_mag[self.ilo])
        self.p2.set_data(self.f_GHz[self.local_minima], self.s21_mag[self.local_minima])
        self.ax.set_title(
            F"Threshold: 3 adjacent points under {'%2.2f' % self.peak_threshold_dB} dB.\n Found {'%.0f' % len(self.local_minima)} minima")
        self.ax.set_xlabel(
            F"Frequency (GHz)\n Spacing threshold {'%.2E' % self.spacing_threshold_Hz} Hz or Q~{'%.2E' % self.spacing_threshold_Q}")
        plt.draw()

    def calc_regions(self):
        bool_threshhold = self.s21_mag < -1.0 * self.peak_threshold_dB
        # self.ilo = np.where(self.s21_mag < -1.0 * self.peak_threshold_dB)[0]
        self.ilo = []
        self.regions = []
        self.local_minima = []
        is_in_theshhold_last = False
        sub_region = []
        for test_index, is_in_theshhold in list(enumerate(bool_threshhold)):
            if is_in_theshhold:
                self.ilo.append(test_index)
                sub_region.append(test_index)
            else:
                if is_in_theshhold_last:
                    # when the last point was in, but not this point it is time to finish the old region
                    self.regions.append(sub_region)
                    sub_region = []
            is_in_theshhold_last = is_in_theshhold
        else:
            if sub_region:
                self.regions.append(sub_region)

        window_calc_data = []
        # calculate the local minima in a simple brute force method
        for region in self.regions:
            minima_this_region = []
            minima_this_region_index = []
            found_this_region = False
            if len(region) > 2:
                for region_index in range(len(region) - 2):
                    middle_region_index = region_index + 1
                    middle_data_index = region[middle_region_index]
                    left = self.s21_mag[region[region_index]]
                    middle = self.s21_mag[middle_data_index]
                    right = self.s21_mag[region[region_index + 2]]
                    if middle < left and middle <= right:
                        found_this_region = True
                        self.local_minima.append(middle_data_index)
                        minima_this_region.append(middle_data_index)
                        minima_this_region_index.append(middle_region_index)
            if found_this_region:
                window_calc_data.append((region, minima_this_region_index, minima_this_region))

        # calculate the resonator windows
        self.minima_as_windows = []
        data_index_minima_left = None
        single_window = None
        right_window_not_found = False
        data_index_bound = 0
        for region, minima_this_region_index, minima_this_region in window_calc_data:
            # deal with spacing conflicts in the same region
            minima_this_region, minima_this_region_index = \
                self.resolve_spacing_conflicts(minima_this_region=minima_this_region,
                                               minima_this_region_index=minima_this_region_index)
            data_index_region_bound_left = region[0]
            data_index_region_bound_right = region[-1]
            # combine minima in the same region with a spacing conflict
            for region_index, data_index_minima in zip(minima_this_region_index, minima_this_region):
                # halfway to the next resonator
                if single_window is not None:
                    data_index_bound = int(np.round((data_index_minima_left + data_index_minima) / 2))
                    if right_window_not_found:
                        single_window["right_max"] = single_window["right_pad"] = \
                            single_window["right_fitter_pad"] = single_window["right_window"] = data_index_bound
                    else:
                        single_window["right_max"] = data_index_bound
                        test_right_pad = single_window["minima"] \
                                         + int(np.round((single_window["right_window"] - single_window["minima"]) \
                                                        * self.window_pad_factor))
                        if single_window["right_max"] < test_right_pad:
                            single_window["right_pad"] = single_window["right_max"]
                        else:
                            single_window["right_pad"] = test_right_pad
                        test_right_fitter_pad = single_window["minima"] \
                                                + int(np.round((single_window["right_window"] - single_window["minima"]) \
                                                               * self.fitter_pad_factor))
                        if single_window["right_max"] < test_right_fitter_pad:
                            single_window["right_fitter_pad"] = single_window["right_max"]
                        else:
                            single_window["right_fitter_pad"] = test_right_fitter_pad

                    self.minima_as_windows.append(SingleWindow(**single_window))
                # the window where resonator is located
                if region_index == minima_this_region_index[0]:
                    data_index_boundary_left = data_index_region_bound_left
                else:
                    data_index_boundary_left = data_index_bound
                if region_index == minima_this_region_index[-1]:
                    data_index_boundary_right = data_index_region_bound_right
                    right_window_not_found = False
                else:
                    right_window_not_found = True
                if right_window_not_found:
                    single_window = {"left_max": data_index_bound, "left_window": data_index_boundary_left,
                                     "minima": data_index_minima}
                else:
                    single_window = {"left_max": data_index_bound, "left_window": data_index_boundary_left,
                                     "minima": data_index_minima, "right_window": data_index_boundary_right}
                # window padding
                test_left_pad = single_window["minima"] \
                                - int(np.round((single_window["minima"] - single_window["left_window"])
                                               * self.window_pad_factor))
                if test_left_pad < single_window["left_max"]:
                    single_window["left_pad"] = single_window["left_max"]
                else:
                    single_window["left_pad"] = test_left_pad
                test_left_fitter_pad = single_window["minima"] \
                                       - int(np.round((single_window["minima"] - single_window["left_window"])
                                                      * self.fitter_pad_factor))
                if test_left_fitter_pad < single_window["left_max"]:
                    single_window["left_fitter_pad"] = single_window["left_max"]
                else:
                    single_window["left_fitter_pad"] = test_left_fitter_pad

                data_index_minima_left = single_window["minima"]
        else:
            # finish the last step in the loop
            data_index_bound = len(self.s21_mag)
            if right_window_not_found:
                single_window["right_max"] = single_window["right_window"] = data_index_bound
            else:
                single_window["right_max"] = data_index_bound
                test_right_pad = single_window["minima"] + \
                                 int(np.round((single_window["right_window"] - single_window["minima"])
                                              * self.window_pad_factor))
                if single_window["right_max"] < test_right_pad:
                    single_window["right_pad"] = single_window["right_max"]
                else:
                    single_window["right_pad"] = test_right_pad
                test_right_fitter_pad = single_window["minima"] \
                                        + int(np.round((single_window["right_window"] - single_window["minima"])
                                                       * self.fitter_pad_factor))
                if single_window["right_max"] < test_right_fitter_pad:
                    single_window["right_fitter_pad"] = single_window["right_max"]
                else:
                    single_window["right_fitter_pad"] = test_right_fitter_pad
            self.minima_as_windows.append(SingleWindow(**single_window))
        self.local_minima = [single_window.minima for single_window in self.minima_as_windows]
        # spacing conflicts across all regions
        self.local_minima, self.minima_as_windows = \
            self.resolve_spacing_conflicts(minima_this_region=self.local_minima,
                                           minima_this_region_index=self.minima_as_windows)

    def resolve_spacing_conflicts(self, minima_this_region, minima_this_region_index):
        found_spacing_conflict = True
        while found_spacing_conflict:
            found_spacing_conflict = False
            number_of_minima_this_region = len(minima_this_region)
            if number_of_minima_this_region > 1:
                for counter in range(number_of_minima_this_region - 1):
                    data_index_minima_left_test = minima_this_region[counter]
                    data_index_minima_right_test = minima_this_region[counter + 1]
                    minima_spacing_Hz = abs(
                        self.f_Hz[data_index_minima_left_test] - self.f_Hz[data_index_minima_right_test])
                    if minima_spacing_Hz < self.spacing_threshold_Hz:
                        # minima are too close:
                        #print(F"Spacing Conflict in same threshold region.")
                        #print(F"   Allowed spacing (MHz): {'%3.3f' % (self.spacing_threshold_Hz * 1.0e-6)}")
                        #print(F"    Minima spacing (MHz): {'%3.3f' % (minima_spacing_Hz * 1.0e-6)}")
                        # keep the lowest of the minima
                        value_left_minima = self.s21_mag[data_index_minima_left_test]
                        value_right_minima = self.s21_mag[data_index_minima_right_test]
                        if value_left_minima < value_right_minima:
                            index_location_to_remove = counter + 1
                            index_location_to_keep = counter
                        else:
                            index_location_to_remove = counter
                            index_location_to_keep = counter + 1
                        # data for the print statement
                        data_index_kept = minima_this_region[index_location_to_keep]
                        data_index_removed = minima_this_region[index_location_to_remove]
                        value_kept_minima = self.s21_mag[data_index_kept]
                        f_MHz_kept_minima = self.f_GHz[data_index_kept] * 1.0e3
                        value_removed_minima = self.s21_mag[data_index_removed]
                        f_MHz_removed_minima = self.f_GHz[data_index_removed] * 1.0e3
                        # where the data is removed
                        minima_this_region_index.pop(index_location_to_remove)
                        minima_this_region.pop(index_location_to_remove)
                        # make the users see what decisions the code is making
                        #print(F"Minima Kept: {value_kept_minima} dbM at {'%3.3f' % f_MHz_kept_minima} MHz")
                        #print(F"Minima Removed: {value_removed_minima} dbM at {'%3.3f' % f_MHz_removed_minima} MHz\n")
                        # stop the loop here and restart from scratch with one less minima
                        found_spacing_conflict = True
                        break
        return minima_this_region, minima_this_region_index


class PopUpDataEntry(object):
    def __init__(self,label,inital_text):
        self.pop_up_fig = plt.figure(3,figsize = (3,1))
        self.value = None
        # text box axis
        text_box_height = 0.25
        text_box_figure_coords = [0.05, 0.05, 0.8, 0.05+text_box_height]
        self.pop_up_fig.text(0.5,0.65,label,ha = "center")
        self.axbox = self.pop_up_fig.add_axes(text_box_figure_coords)
        self.text_box = TextBox(self.axbox, "", textalignment="center",initial = inital_text)
        self.text_box.on_submit(self.submit)
        self.pop_up_fig.canvas.mpl_connect('key_press_event', self.update_text)
        self.pop_up_fig.canvas.mpl_connect('button_press_event', self.update_text)
        self.text_box.cursor_index = len(inital_text)
        self.text_box._rendercursor()
        plt.show(block = False)
        plt.pause(0.1)
        self.text_box.begin_typing(None)

        while self.value is None:  # can't use same plt.show(block = True) since already in use
            plt.pause(0.1)
        plt.close(self.pop_up_fig)


    def submit(self,expression):
        self.value = expression

    def update_text(self, event): #everytime you type redraw text box plot
        plt.draw()



def compute_dI_and_dQ(I, Q, freq=None, filterstr='SG', do_deriv=True):
    """
    Given I,Q,freq arrays
    input filterstr = 'SG' for sav-gol filter with builtin gradient, 'SGgrad' savgol then apply gradient to filtered
    do_deriv: if want to look at filtered non differentiated data.
    """
    if freq is None:
        df = 1.0
    else:
        df = freq[1] - freq[0]
    dI = filtered_differential(I, df, filtertype=filterstr, do_deriv=do_deriv)
    dQ = filtered_differential(Q, df, filtertype=filterstr, do_deriv=do_deriv)
    return dI, dQ


def filtered_differential(data, df, filtertype=None, do_deriv=True):
    """
    take 1d array data with spacing df. return filtered version of data depending on filterrype
    """
    window = 13
    n = 3
    if filtertype is None:
        out = np.gradient(data, df)
    elif filtertype.lower() == 'sg':
        if do_deriv == True:
            out = signal.savgol_filter(data, window, n, deriv=1, delta=df)
        else:
            out = signal.savgol_filter(data, window, n, deriv=0, delta=df)
    elif filtertype.lower() == 'sggrad':
        tobegrad = signal.savgol_filter(data, window, n)
        out = np.gradient(tobegrad, df)
    else:
        raise KeyError(F"filtertype: {filtertype} is not recognized.")
    return out


class InteractiveFilterPlot(object):
    def __init__(self, f_Hz, s21_mag, smoothing_scale_Hz=5.0e6):
        self.smoothing_scale_Hz = smoothing_scale_Hz
        self.f_Hz = f_Hz
        self.f_GHz = f_Hz * 1.0e-9
        self.s21_mag = s21_mag

        # set up plot
        self.key_font_size = 9
        top = 0.90
        bottom = 0.1
        left = 0.08
        right = 0.99
        x_width = right - left
        y_height= top - bottom
        key_x_width = 0.3
        
        self.fig = plt.figure(2, figsize=(16, 6))
        #self.ax = self.fig.add_subplot(111)
        x_width_plot = x_width - key_x_width
        figure_coords = [left, bottom, x_width_plot, y_height]
        self.ax = self.fig.add_axes(figure_coords, frameon=False, autoscale_on=True)
        key_y_height = 0.15#y_height
        key_figure_coords = [left + x_width_plot, top-key_y_height, key_x_width, key_y_height]
        self.ax_key = self.fig.add_axes(key_figure_coords, frameon=False)
        self.ax_key.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        self.ax_key.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
     
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # first plot data and filter function before removing filter function
        self.filtermags = lowpass_cosine(y=self.s21_mag,
                                         tau=self.f_Hz[1] - self.f_Hz[0],
                                         f_3db=1. / self.smoothing_scale_Hz,
                                         width=0.1 * (1.0 / self.smoothing_scale_Hz),
                                         padd_data=True)
        # the cosine filter drops the last point is the array has an pdd number of points
        self.len_filtered = len(self.filtermags)
        self.s21_mag = self.s21_mag[:self.len_filtered]
        self.f_Hz = self.f_Hz[:self.len_filtered]
        self.f_GHz = self.f_Hz * 1.0e-9
        # calculations for peak spacing (rejection based on threshold)
        self.highpass_mags = self.s21_mag - self.filtermags
        self.l1, = self.ax.plot(self.f_GHz, self.s21_mag, label='#nofilter')
        self.l2, = self.ax.plot(self.f_GHz, self.filtermags, label='#filter')
        self.ax.set_title(F"Close all plots when finished\nSmoothing Scale: {'%.2E' % self.smoothing_scale_Hz} Hz.")
        self.ax.legend(loc=1)

        print(
            "Press left or right to change the smothing scale by 40% or press t to enter a custom smoothing scale value.")
        print("Close all plots when finished")
        self.ax.set_xlabel('Frequency (GHz)')
        self.ax.set_ylabel('Power (dB)')
        self.plot_instructions()
        plt.show(block=True)


    def on_key_press(self, event):
        # print event.key
        # has to be shift and ctrl because remote viewers only forward
        # certain key combinations
        # print event.key == 'd'
        self.refresh_plot()
        if event.key == 'left':
            self.smoothing_scale_Hz = self.smoothing_scale_Hz / 1.4
            self.refresh_plot()
        if event.key == 'right':
            self.smoothing_scale_Hz = self.smoothing_scale_Hz * 1.4
            self.refresh_plot()
        if event.key == 't':
            pop_up = PopUpDataEntry("Enter smoothing scale value in Hz","6000000")
            self.smoothing_scale_Hz = float(pop_up.value)
            self.refresh_plot()

    def refresh_plot(self):
        # first plot data and filter function before removing filter function
        self.filtermags = lowpass_cosine(y=self.s21_mag,
                                         tau=self.f_Hz[1] - self.f_Hz[0],
                                         f_3db=1. / self.smoothing_scale_Hz,
                                         width=0.1 * (1.0 / self.smoothing_scale_Hz),
                                         padd_data=True)
        # the cosine filter drops the last point is the array has an odd number of points
        self.len_filtered = len(self.filtermags)
        self.s21_mag = self.s21_mag[:self.len_filtered]
        self.f_Hz = self.f_Hz[:self.len_filtered]
        self.f_GHz = self.f_Hz * 1.0e-9
        # calculations for peak spacing (rejection based on threshold)
        self.highpass_mags = self.s21_mag - self.filtermags
        self.l1.set_data(self.f_GHz, self.s21_mag)
        self.l2.set_data(self.f_GHz, self.filtermags)
        self.ax.set_title(F"Close all plots when finished\nSmoothing Scale: {'%.2E' % self.smoothing_scale_Hz} Hz.")

        plt.draw()

    def instructions(self):
        instructions = [("left-arrow", "decrease smoothing scale", 'green'),
                        ("right-arrow", "increase smoothing scale", 'cyan'),
                        ("T-key", "enter smoothing scale value", 'yellow')]

        return instructions

    def plot_instructions(self):
        instructions = self.instructions()
        self.ax_key.clear()
        steps_per_item = 1.5
        y_step = 0.9 / (steps_per_item * float(len(instructions)))
        y_now = 0.9
        for key_press, description, color in instructions:
            self.ax_key.text(0.4, y_now, key_press.center(13), color=text_color_matplotlib[color],
                             ha='right', va='center', size=self.key_font_size, weight="bold",
                             family='monospace', bbox=dict(color=color, ls='-', lw=2.0, ec='black'))
            self.ax_key.text(0.45, y_now, description, color='black',
                             ha='left', va='center', size=self.key_font_size - 2)
            y_now -= steps_per_item * y_step
        #self.ax_key.set_xlim(0, 1)
        #self.ax_key.set_ylim(0, 1)
        self.ax_key.set_title('Main Menu')
        plt.draw()


def filter_trace(path, bb_freqs, sweep_freqs):
    chan_I, chan_Q = open_stored_sweep(path)
    channels = np.arange(np.shape(chan_I)[1])
    mag = np.zeros((len(bb_freqs), len(sweep_freqs)))
    chan_freqs = np.zeros((len(bb_freqs), len(sweep_freqs)))
    for chan in channels:
        mag[chan] = (np.sqrt(chan_I[:, chan] ** 2 + chan_Q[:, chan] ** 2))
        chan_freqs[chan] = (sweep_freqs + bb_freqs[chan]) / 1.0e6
    # mag = np.concatenate((mag[len(mag)/2:], mag[0:len(mag)/2]))
    mags = 20 * np.log10(mag / np.max(mag))
    mags = np.hstack(mags)
    # chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[0:len(chan_freqs)/2]))
    chan_freqs = np.hstack(chan_freqs)
    return chan_freqs, mags


def lowpass_cosine(y, tau, f_3db, width, padd_data=True):
    # padd_data = True means we are going to symmetric copies of the data to the start and stop
    # to reduce/eliminate the discontinuities at the start and stop of a dataset due to filtering
    #
    # False means we're going to have transients at the start and stop of the data
    # kill the last data point if y has an odd length
    if np.mod(len(y), 2):
        y = y[0:-1]
    # add the weird padd
    # so, make a backwards copy of the data, then the data, then another backwards copy of the data
    if padd_data:
        y = np.append(np.append(np.flipud(y), y), np.flipud(y))
    # take the FFT
    ffty = fftpack.fft(y)
    ffty = fftpack.fftshift(ffty)
    # make the companion frequency array
    delta = 1.0 / (len(y) * tau)
    nyquist = 1.0 / (2.0 * tau)
    freq = np.arange(-nyquist, nyquist, delta)
    # turn this into a positive frequency array
    # print((len(ffty) // 2))
    pos_freq = freq[(len(ffty) // 2):]
    # make the transfer function for the first half of the data
    i_f_3db = min(np.where(pos_freq >= f_3db)[0])
    f_min = f_3db - (width / 2.0)
    i_f_min = min(np.where(pos_freq >= f_min)[0])
    f_max = f_3db + (width / 2.0)
    i_f_max = min(np.where(pos_freq >= f_max)[0])
    transfer_function = np.zeros(len(y) // 2)
    transfer_function[0:i_f_min] = 1
    transfer_function[i_f_min:i_f_max] = (1 + np.sin(-np.pi * ((freq[i_f_min:i_f_max] - freq[i_f_3db]) / width))) / 2.0
    transfer_function[i_f_max:(len(freq) // 2)] = 0
    # symmetrize this to be [0 0 0 ... .8 .9 1 1 1 1 1 1 1 1 .9 .8 ... 0 0 0] to match the FFT
    transfer_function = np.append(np.flipud(transfer_function), transfer_function)
    # apply the filter, undo the fft shift, and invert the fft
    filtered = np.real(fftpack.ifft(fftpack.ifftshift(ffty * transfer_function)))
    # remove the padd, if we applied it
    if padd_data:
        filtered = filtered[(len(y) // 3):(2 * (len(y) // 3))]
    # return the filtered data
    return filtered


def find_vna_sweep(f_Hz, z, smoothing_scale_Hz=5.0e6, spacing_threshold_Hz=1.0e5):
    """
    f is frequencies (Hz)
    z is complex S21
    Smoothing scale (Hz)
    spacing threshold (Hz)
    """
    # first plot data and filter function before removing filter function
    s21_mag = 20 * np.log10(np.abs(z))
    ipf = InteractiveFilterPlot(f_Hz, s21_mag, smoothing_scale_Hz=smoothing_scale_Hz)

    # identify peaks using the interactive threshold plot
    ipt = InteractiveThresholdPlot(f_Hz=ipf.f_Hz,
                                   s21_mag=ipf.highpass_mags,
                                   peak_threshold_dB=1.5,
                                   spacing_threshold_Hz=spacing_threshold_Hz)

    # Zero everything but the resonators
    # highpass_mags[highpass_mags > -1.0 * ipt.peak_threshold_dB] = 0

    # the spacing thresholding was move to be inside the interactive threshold class
    kid_idx = ipt.local_minima
    ip = InteractivePlot(ipf.f_Hz, ipf.highpass_mags, kid_idx)
    return ip


def slice_vna(f, z, kid_index, q_slice=2000, flag_collided=True):
    # make f in Hz for fitting
    # Q = f/(delta f) for fitting is determined by the lowest frequencies assumed to be at index 0
    # delta f = f/Q
    df = f[1] - f[0]
    n_iq_points = int(f[0] / q_slice // df)
    if np.mod(n_iq_points, 2) == 0:
        n_iq_points = n_iq_points + 1
    print(n_iq_points)
    res_freq_array = np.zeros((n_iq_points, len(kid_index)))
    res_array = np.zeros((n_iq_points, len(kid_index))).astype('complex')
    print(res_array.dtype)
    for i in range(0, len(kid_index)):
        a = kid_index[i] - n_iq_points // 2 - 1
        b = kid_index[i] + n_iq_points // 2

        res_freq_array[:, i] = f[a:b]
        res_array[:, i] = z[a:b]
        if flag_collided:
            if i < len(kid_index) - 1:  # dont check last res
                # print(i)
                if kid_index[i + 1] - kid_index[i] < n_iq_points:  # collision at higher frequency
                    high_cutoff = int((kid_index[i + 1] + kid_index[i]) / 2)
                    # print(i,a,high_cutoff,b)
                    res_freq_array[high_cutoff - a:,i] = np.nan
                    res_array[high_cutoff - a:,i] = np.nan * (1 + 1j)
            if i != 0:  # dont check first res
                # print(i)
                if kid_index[i] - kid_index[i - 1] < n_iq_points:
                    low_cutoff = int((kid_index[i] + kid_index[i - 1]) / 2)
                    # print(i,a,low_cutoff,b)
                    res_freq_array[:low_cutoff - a, i] = np.nan
                    res_array[:low_cutoff - a, i] = np.nan * (1 + 1j)

    return res_freq_array, res_array


def fit_slices(res_freq_array, res_array, do_plots=True, plot_filename='fits'):
    if do_plots:
        pdf_pages = PdfPages(plot_filename + ".pdf")
    fits_dict_mag = {}
    fits_dict_iq = {}
    for i in range(0, res_freq_array.shape[0]):
        if do_plots:
            fig = plt.figure(i, figsize=(12, 6))
        try:
            fit = fit_nonlinear_iq(res_freq_array[i, :][~np.isnan(res_freq_array[i, :])],
                                   res_array[i, :][~np.isnan(res_array[i, :])])
            fits_dict_iq[i] = fit
            if do_plots:
                plt.subplot(121)
                plt.plot(np.real(res_array[i, :]), np.imag(res_array[i, :]), 'o', label='data')
                plt.plot(np.real(fit['fit_result']), np.imag(fit['fit_result']), label='fit')
                plt.plot(np.real(fit['x0_result']), np.imag(fit['x0_result']), label='guess')
                plt.legend()
        except Exception as inst:
            print("could not fit")
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            # print(inst)          # __str__ allows args to be printed directly,
            fits_dict_iq[i] = 'bad fit'
        try:
            fit2 = fit_nonlinear_mag(res_freq_array[i, :][~np.isnan(res_freq_array[i, :])],
                                        res_array[i, :][~np.isnan(res_array[i, :])])
            fits_dict_mag[i] = fit2
            if do_plots:
                plt.subplot(122)
                plt.plot(res_freq_array[i, :], 20 * np.log10(np.abs(res_array[i, :])), label='data')
                plt.plot(res_freq_array[i, :][~np.isnan(res_freq_array[i, :])],
                         10 * np.log10(np.abs(fit2['fit_result'])), label='fit')
                plt.plot(res_freq_array[i, :][~np.isnan(res_freq_array[i, :])],
                         10 * np.log10(np.abs(fit2['x0_result'])), label='guess')
                plt.legend()
        except Exception as inst:
            print("could not fit")
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            # print(inst)          # __str__ allows args to be printed directly,
            fits_dict_mag[i] = 'bad fit'
        if do_plots:
            pdf_pages.savefig(fig)
            plt.close()
    if do_plots:
        pdf_pages.close()

    return fits_dict_iq, fits_dict_mag


def retune_vna(f, z, kid_index, n_points_look_around=0, look_low_high=[0, 0], f_old=None, z_old=None,
               kid_index_old=None):
    """
    This is a program for when the resonances move and you need to retune the indexes of the resonators
    use n_point_look_around = 10 to look to lower and higher frequencies within 10 data points to find a new min
    use look_left_right = [10,20] to look for a new min 10 points to the lower frequencies and 20 points to higher frequencies

    if you would like to have the old data and kid indexes displayed in the background suppy
    f_old, z_old, kid_index old
    """
    if n_points_look_around > 0:
        for i in range(0, len(kid_index)):
            new_index = np.argmin(
                20 * np.log10(np.abs(z[kid_index[i] - n_points_look_around:kid_index[i] + n_points_look_around]))) + \
                        kid_index[i] - n_points_look_around
            kid_index[i] = new_index

    ip = InteractivePlot(f, 20 * np.log10(np.abs(z)), kid_index, f_old=f_old, data_old=20 * np.log10(np.abs(z_old)),
                         kid_idx_old=kid_index_old)

    return ip



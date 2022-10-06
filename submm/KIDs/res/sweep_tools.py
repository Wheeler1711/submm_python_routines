import gc
import copy
import platform

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from matplotlib.backends.backend_pdf import PdfPages
import tqdm

from submm.KIDs.res.utils import colorize_text, text_color_matplotlib

'''
Tools for handling resonator iq sweeps 
i.e. finding minimum, maximum of iq sweep arrays
plotting iq sweep arrays
retuning resonators
finding correct readout power level
modified from https://github.com/sbg2133/kidPy/
'''


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


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


class InteractivePlot(object):
    """
    interactive plot for plot many resonators iq data
    chan freqs and z should have dimension n_iq points by n_res
    also can provide multiple sweeps for plotting by adding extra dimension
    i.e. chan freqs and z could have dimension n_iq points by n_res by n_sweeps
    combined data should have dimension n_res by n_different_types of data
    """

    log_y_data_types = {'chi_sq'}
    key_font_size = 9

    def __init__(self, chan_freqs, z, look_around=2, stream_data=None, retune=True, find_min=True,
                 combined_data=None, combined_data_names=None, sweep_labels=None, sweep_line_styles=None,
                 combined_data_format=None, flags=None, plot_title=None, plot_frames=True):
        if len(z.shape) < 3:  # only one sweep
            self.z = z.reshape((z.shape[0], z.shape[1], 1))
            self.chan_freqs = chan_freqs.reshape((chan_freqs.shape[0], chan_freqs.shape[1], 1))
        else:
            self.z = z
            self.chan_freqs = chan_freqs

        plt.rcParams['keymap.fullscreen'] = ['shift+=']  # remove ('f', 'ctrl+f'), make +

        self.Is = np.real(self.z)
        self.Qs = np.imag(self.z)
        self.find_min = find_min
        self.retune = retune
        self.combined_data = combined_data
        self.combined_data_format = combined_data_format
        self.combined_data_names = combined_data_names
        self.stream_data = stream_data
        self.targ_size = chan_freqs.shape[0]
        self.look_around = look_around
        self.plot_index = 0
        self.combined_data_index = 0
        self.res_index_override = np.asarray((), dtype=np.int16)
        self.override_freq_index = np.asarray((), dtype=np.int16)
        self.shift_is_held = False
        self.update_min_index()
        if flags is None:
            self.flags = []
            for i in range(chan_freqs.shape[1]):
                self.flags.append([])
        else:
            self.flags = flags
        if retune:
            self.combined_data_names = ['min index']
        # data remove variables for the interactive plot
        self.remove_mode = False
        self.res_indexes_removed = set()
        self.res_indexes_staged = set()

        # set up plot
        top = 0.94
        bottom = 0.05
        left = 0.08
        right = 0.99
        x_space = 0.075
        y_space = 0.09
        x_width_total = right - left - x_space
        y_height_total = top - bottom - y_space
        if self.combined_data is None:
            self.fig = plt.figure(1, figsize=(13, 6))
            # the magnitude plot - upper left quadrant
            mag_x_width = 0.5
            mag_y_height = y_height_total
            mag_figure_coords = [left, top - mag_y_height, mag_x_width, mag_y_height]
            # the IQ plot - upper right quadrant
            iq_x_width = x_width_total - mag_x_width
            iq_y_height = mag_y_height
            iq_figure_coords = [left + mag_x_width + x_space, top - iq_y_height, iq_x_width, iq_y_height]
            combined_figure_coords = None
            key_figure_coords = None
        else:
            self.fig = plt.figure(1, figsize=(13, 10))
            # the magnitude plot - upper left quadrant
            mag_x_width = 0.5
            mag_y_height = 0.45
            mag_figure_coords = [left, top - mag_y_height, mag_x_width, mag_y_height]
            # the IQ plot - upper right quadrant
            iq_x_width = x_width_total - mag_x_width
            iq_y_height = mag_y_height
            iq_figure_coords = [left + mag_x_width + x_space, top - iq_y_height, iq_x_width, iq_y_height]
            # the Key Figure area for the interactive plot instructions-key
            key_x_width = 0.2
            key_y_height = y_height_total - mag_y_height
            combined_x_width = right - left - key_x_width
            key_figure_coords = [left + combined_x_width, bottom, key_x_width, key_y_height]
            # the combined data plot - lower plane
            combined_y_height = key_y_height
            combined_figure_coords = [left, bottom, combined_x_width, combined_y_height]

        if plot_title is not None:
            self.fig.suptitle(plot_title, y=0.99)
        self.ax_mag = self.fig.add_axes(mag_figure_coords, frameon=plot_frames)
        self.ax_mag.set_ylabel("Power (dB)")
        self.ax_mag.set_xlabel("Frequency (MHz)")
        self.ax_iq = self.fig.add_axes(iq_figure_coords, frameon=plot_frames)
        self.ax_iq.set_ylabel("Q")
        self.ax_iq.set_xlabel("I")
        if combined_figure_coords is None:
            self.ax_combined = None
        else:
            self.ax_combined = self.fig.add_axes(combined_figure_coords, frameon=False, autoscale_on=False)
            self.ax_combined.set_ylabel("")
            self.ax_combined.set_xlabel("Resonator index")
            self.ax_key = self.fig.add_axes(key_figure_coords, frameon=False)
            self.ax_key.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            self.ax_key.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            self.plot_instructions()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        if self.stream_data is not None:
            self.s2, = self.ax_iq.plot(np.real(self.stream_data[:, self.plot_index]),
                                       np.imag(self.stream_data[:, self.plot_index]), '.')

        if not sweep_line_styles:
            sweep_line_styles = ["-o"]
            for i in range(1, self.z.shape[2]):
                sweep_line_styles.append("-")
        if not sweep_labels:
            sweep_labels = ["Data 1"]
            for i in range(1, self.z.shape[2]):
                sweep_labels.append("Data " + str(i + 1))
        self._mag_lines = [self.ax_mag.plot(self.chan_freqs[:, self.plot_index, i] / 10 ** 6, 10 * np.log10(
            self.Is[:, self.plot_index, i] ** 2 + self.Qs[:, self.plot_index, i] ** 2), sweep_line_styles[i], mec="k",
                                            label=sweep_labels[i])
                           for i in range(0, self.z.shape[2])]

        self.ax_leg = self.ax_mag.legend()

        self._iq_lines = [self.ax_iq.plot(
            self.Is[:, self.plot_index, i], self.Qs[:, self.plot_index, i], sweep_line_styles[i], mec="k",
            label=sweep_labels[i])
            for i in range(0, self.z.shape[2])]

        if self.retune:
            self.p1, = self.ax_mag.plot(self.chan_freqs[self.min_index[self.plot_index], self.plot_index, 0] / 10 ** 6,
                                        10 * np.log10(
                                            self.Is[self.min_index[self.plot_index], self.plot_index, 0] ** 2 +
                                            self.Qs[self.min_index[self.plot_index], self.plot_index, 0] ** 2),
                                        '*', markersize=15, color='darkorchid')
            self.p2, = self.ax_iq.plot(self.Is[self.min_index[self.plot_index], self.plot_index, 0],
                                       self.Qs[self.min_index[self.plot_index], self.plot_index, 0], '*', markersize=15)

        center_freq_MHz = self.chan_freqs[self.chan_freqs.shape[0] // 2, self.plot_index, 0] / 10 ** 6
        self.ax_mag.set_title('Resonator Index ' + str(self.plot_index) + '\n' + f'{"%3.3f" % center_freq_MHz} MHz')
        if self.retune:
            self.ax_iq.set_title("Look Around Points " + str(self.look_around))
        print("")
        print("Interactive Resonance Plotting Activated")
        self.print_instructions()

        # combined plot variables used in the first initialization
        self.combined_data_points = None
        self.combined_data_highlight = None
        self.combined_data_crosshair_x = None
        self.combined_data_crosshair_y = None
        self.combined_staged_for_removal = None
        self.combined_data_legend = None
        self.res_indexes = None
        self.combined_data_values = None
        if self.combined_data is None:
            self.res_indexes_original = np.arange(0, self.chan_freqs.shape[1])
            self.res_indexes = self.res_indexes_original
        else:
            # combined data initial formatting
            self.combined_data = np.asarray(self.combined_data)
            if len(self.combined_data.shape) == 1:
                self.combined_data = np.expand_dims(self.combined_data, 1)

            if not self.combined_data_format:
                self.combined_data_format = []
                for i in range(0, self.combined_data.shape[1]):
                    self.combined_data_format.append(self.combined_data_names[i] + ': {:g}')
            else:
                self.combined_data_format = self.combined_data_format
            self.res_indexes_original = np.arange(0, self.combined_data.shape[0])
            # run the initialization script
            self.combined_plot(ax_combined=self.ax_combined)

        plt.show(block=True)

    def combined_plot(self, ax_combined):
        if self.combined_data_points is not None:
            self.combined_data_points.remove()
            self.combined_data_points = None
        if self.combined_data_highlight is not None:
            self.combined_data_highlight.remove()
            self.combined_data_highlight = None
        if self.combined_data_crosshair_x is not None:
            self.combined_data_crosshair_x.remove()
            self.combined_data_crosshair_x = None
        if self.combined_data_crosshair_y is not None:
            self.combined_data_crosshair_y.remove()
            self.combined_data_crosshair_y = None
        gc.collect()
        # initialize or re-initialize the combined data for the plot
        if self.res_indexes_removed:
            removed_array = np.array(sorted(self.res_indexes_removed))
            self.res_indexes = np.delete(arr=self.res_indexes_original, obj=removed_array, axis=0)
            self.combined_data_values = np.delete(arr=self.combined_data, obj=removed_array, axis=0)
        else:
            self.res_indexes = self.res_indexes_original
            self.combined_data_values = self.combined_data
        # rest the plot index to the first resonator
        self.plot_index = self.res_indexes[0]
        data_name = self.combined_data_names[self.combined_data_index]
        # start setting the plot
        ax_combined.set_title(data_name)
        ax_combined.set_ylabel(data_name)
        if data_name in self.log_y_data_types:
            ax_combined.set_yscale('log')
        else:
            ax_combined.set_yscale('linear')
        # plot the points and define the curves handle to update the data later
        combined_values_this_index = self.combined_data_values[:, self.combined_data_index]
        self.combined_data_points, = ax_combined.plot(self.res_indexes, combined_values_this_index, '.',
                                                      markersize=12, color='darkorchid', markerfacecolor="black")
        # highlighting and cross-hairs for the selected data point
        highlighted_data_value = self.combined_data[self.plot_index, self.combined_data_index]
        label = self.combined_data_format[self.combined_data_index].format(highlighted_data_value)
        x_pos = self.plot_index
        y_pos = highlighted_data_value
        # the highlight symbol
        self.combined_data_highlight, = ax_combined.plot(x_pos, y_pos, 'o', markerfacecolor="None",
                                                         markeredgecolor='darkorange', markersize=14, label=label)
        # x crosshair
        self.combined_data_crosshair_x = ax_combined.axvline(x=x_pos, color='firebrick', ls='-', linewidth=1)
        # y crosshair
        self.combined_data_crosshair_y = ax_combined.axhline(y=y_pos, color='firebrick', ls='-', linewidth=1)
        self.combined_data_legend = ax_combined.legend()
        # Staged for Removal - In 'Remove Mode' these point have an 'X' to denote that they are staged for removal.
        self.plot_staged_for_removal(ax_combined=ax_combined)
        # we only rescale manually (intentionally) for this axis
        ax_combined.autoscale()

    def plot_staged_for_removal(self, ax_combined):
        if self.combined_staged_for_removal is not None:
            self.combined_staged_for_removal.remove()
            self.combined_staged_for_removal = None
        if self.res_indexes_staged:
            staged_indexes = sorted(self.res_indexes_staged)
            stage_values = [self.combined_data[staged_index, self.combined_data_index]
                            for staged_index in staged_indexes]
            self.combined_staged_for_removal, =  ax_combined.plot(staged_indexes, stage_values, 'x', ls='None',
                                                                  markersize=12, color='firebrick')

    def refresh_plot(self, autoscale=True):
        if len(self.flags[self.plot_index]) > 0:
            self.ax_mag.set_facecolor('lightyellow')
            self.ax_iq.set_facecolor('lightyellow')
        else:
            self.ax_mag.set_facecolor("None")
            self.ax_iq.set_facecolor("None")
        for i, mag_line in enumerate(self._mag_lines):
            mag_line[0].set_data(self.chan_freqs[:, self.plot_index, i] / 10 ** 6, 10 * np.log10(
                self.Is[:, self.plot_index, i] ** 2 + self.Qs[:, self.plot_index, i] ** 2))
        if self.retune:
            self.p1.set_data(self.chan_freqs[self.min_index[self.plot_index], self.plot_index, 0] / 10 ** 6,
                             10 * np.log10(self.Is[self.min_index[self.plot_index], self.plot_index, 0] ** 2 +
                                           self.Qs[self.min_index[self.plot_index], self.plot_index, 0] ** 2))

        self.ax_mag.relim()
        self.ax_mag.autoscale()
        center_freq_MHz = self.chan_freqs[self.chan_freqs.shape[0] // 2, self.plot_index, 0] / 10 ** 6
        self.ax_mag.set_title('Resonator Index ' + str(self.plot_index) + '\n' + f'{"%3.3f" % center_freq_MHz} MHz')
        if self.retune:
            self.ax_iq.set_title("Look Around Points " + str(self.look_around))
        for i, iq_line in enumerate(self._iq_lines):
            iq_line[0].set_data((self.Is[:, self.plot_index, i],
                                 self.Qs[:, self.plot_index, i]))
        if self.retune:
            self.p2.set_data(self.Is[self.min_index[self.plot_index], self.plot_index, 0],
                             self.Qs[self.min_index[self.plot_index], self.plot_index, 0])

        if self.stream_data is not None:
            self.s2.set_data(np.real(self.stream_data[:, self.plot_index]),
                             np.imag(self.stream_data[:, self.plot_index]))
        self.ax_iq.relim()
        self.ax_iq.autoscale()
        if self.combined_data is not None:
            data_type = self.combined_data_names[self.combined_data_index]
            self.ax_combined.set_title(data_type)
            self.ax_combined.set_ylabel(data_type)
            if data_type in self.log_y_data_types:
                self.ax_combined.set_yscale('log')
            else:
                self.ax_combined.set_yscale('linear')
            # reset the combined plot data
            self.combined_data_points.set_data(self.res_indexes, self.combined_data_values[:, self.combined_data_index])
            # label for the value of the highlighted data point
            highlighted_value = self.combined_data[self.plot_index, self.combined_data_index]
            label = self.combined_data_format[self.combined_data_index].format(highlighted_value)
            x_pos = self.plot_index
            y_pos = highlighted_value
            self.combined_data_highlight.set_data(x_pos, y_pos)
            self.combined_data_legend.texts[0].set_text(label)
            self.ax_combined.relim()
            if self.res_indexes_staged:
                staged_indexes = sorted(self.res_indexes_staged)
                stage_values = [self.combined_data[staged_index, self.combined_data_index]
                                for staged_index in staged_indexes]
                self.combined_staged_for_removal.set_data(staged_indexes, stage_values)
            if autoscale:
                self.ax_combined.autoscale()
            self.combined_data_crosshair_x.set_xdata(x_pos)
            self.combined_data_crosshair_y.set_ydata(y_pos)

        plt.draw()

    def update_min_index(self):
        if self.find_min:
            self.min_index = np.argmin(
                self.Is[self.targ_size // 2 - self.look_around:self.targ_size // 2 + self.look_around, :, 0] ** 2 +
                self.Qs[self.targ_size // 2 - self.look_around:self.targ_size // 2 + self.look_around, :, 0] ** 2,
                axis=0) + (self.targ_size // 2 - self.look_around)
        else:
            self.min_index = find_max_didq(self.z[:, :, 0], self.look_around)
        # handel overridden points
        for i, override_index in enumerate(self.res_index_override):
            self.min_index[override_index] = self.override_freq_index[i]

        if self.retune:
            self.combined_data = np.expand_dims(self.min_index, 1)

    def instructions(self):
        instructions = [("left-arrow", "change resonator left", 'green'),
                        ("right-arrow", "change resonator right", 'cyan')]
        if self.retune:
            instructions.extend([('down-arrow', 'change look around points down', 'yellow'),
                                 ('up-arrow', 'change look around points up', 'blue')])
            if platform.system() == 'Darwin':
                instructions.extend([("Hold any letter a key and right click on the magnitude plot",
                                      "to override tone position", 'clack')])
            else:
                instructions.extend([("Hold 'shift' and right click on the magnitude plot",
                                      "to override tone position", 'black')])
        if self.combined_data is not None:
            instructions.extend([('down-arrow', 'change y-data type', 'yellow'),
                                 ('up-arrow', 'change y-data type', 'blue'),
                                 ('double-click', 'go to the resonator index', 'black'),
                                 ("E-key", "to enter 'Remove-Mode'", 'red')])
            if self.retune:
                instructions.extend(["Warning both look around points and combined data are mapped to " +
                                     "up and down arrows, consider not returning and plotting combined " +
                                     "data at the same time"])
        instructions.append(("W-key", "write a pdf of all resonators", 'purple'))

        return instructions

    def instructions_remove(self):
        instructions = [("left-arrow", "change resonator left", 'green'),
                        ("right-arrow", "change resonator right", 'cyan'),
                        ('down-arrow', 'change y-data type', 'yellow'),
                        ('up-arrow', 'change y-data type', 'blue'),
                        ('double-click', 'go to the resonator index', 'black'),
                        ("T-key", "save and exit 'remove-mode'", 'purple'),
                        ("E-key", "exit 'remove-mode'", 'red'),
                        ("X-key", "stage for removal", 'yellow'),
                        ("Z-key", "un-stage (clear) for removal", 'white')]

        return instructions

    def plot_instructions(self):
        if self.remove_mode:
            instructions = self.instructions_remove()
        else:
            instructions = self.instructions()
        self.ax_key.clear()
        stemps_per_item = 1.5
        y_step = 0.9 / (stemps_per_item * float(len(instructions)))
        y_now = 0.95
        for key_press, description, color in instructions:
            self.ax_key.text(0.4, y_now, key_press.center(13), color=text_color_matplotlib[color],
                             ha='right', va='center', size=self.key_font_size, weight="bold",
                             family='monospace', bbox=dict(color=color, ls='-', lw=2.0, ec='black'))
            self.ax_key.text(0.45, y_now, description, color='black',
                             ha='left', va='center', size=self.key_font_size - 2)
            y_now -= stemps_per_item * y_step
        self.ax_key.set_xlim(0, 1)
        self.ax_key.set_ylim(0, 1)
        plt.draw()

    def print_instructions(self):
        if self.remove_mode:
            instructions = self.instructions_remove()
        else:
            instructions = self.instructions()
        for key_press, description, color in instructions:
            if color == 'black':
                text_color = 'white'
            else:
                text_color = 'black'
            color_text = colorize_text(text=key_press.capitalize().center(20), style_text='bold',
                                       color_text=text_color, color_background=color)
            print(f'{color_text} : {description}')

    def on_key_press(self, event):
        # items on all menus
        if event.key == 'right':
            if self.plot_index == self.res_indexes[-1]:
                self.plot_index = self.res_indexes[0]
            else:
                self.plot_index = self.res_indexes[np.where(self.res_indexes == self.plot_index)[-1] + 1][-1]
            self.refresh_plot()

        if event.key == 'left':
            if self.plot_index == self.res_indexes[0]:
                self.plot_index = self.res_indexes[-1]
            else:
                self.plot_index = self.res_indexes[np.where(self.res_indexes == self.plot_index)[-1] - 1][0]
            self.refresh_plot()

        if event.key == 'up':
            if self.look_around != self.chan_freqs.shape[0] // 2:
                self.look_around = self.look_around + 1
                self.update_min_index()
                if self.retune:
                    self.combined_data = np.expand_dims(self.min_index, 1)
            if self.combined_data is not None:
                if self.combined_data_index == self.combined_data.shape[1] - 1:
                    self.combined_data_index = 0
                else:
                    self.combined_data_index = self.combined_data_index + 1
            self.refresh_plot()

        if event.key == 'down':
            if self.look_around != 1:
                self.look_around = self.look_around - 1
                self.update_min_index()
                if self.retune:
                    self.combined_data = np.expand_dims(self.min_index, 1)
            if self.combined_data is not None:
                if self.combined_data_index == 0:
                    self.combined_data_index = self.combined_data.shape[1] - 1
                else:
                    self.combined_data_index = self.combined_data_index - 1
            self.refresh_plot()

        if platform.system().lower() == 'darwin':
            if event.key == 'a':
                self.shift_is_held = True
        else:
            if event.key == 'shift':
                self.shift_is_held = True

        # toggle different menus
        if self.remove_mode:
            self.on_key_press_remove_mode(event=event)
        else:
            # only on the main menu
            if event.key == 'e':
                self.remove_mode = True
                print('\nEntered "Remove Mode"')
                self.print_instructions()
                self.plot_instructions()

            if event.key == 'f':
                print("flagging point", event.xdata)
                print("current flags: ", self.flags[self.plot_index])
                flag = input("Enter flag string: ").lower()
                if flag == "c":
                    flag = "collision"
                elif flag == "s":
                    flag = "shallow"
                else:
                    pass

                if flag != '':
                    self.flags[self.plot_index].append(flag)
                    self.refresh_plot()
                print("Flags are now: ", self.flags[self.plot_index])

            if event.key == 'w':
                print("saving to pdf")
                filename = input("enter filename for pdf: ")
                if filename == '':
                    filename = "res_plots.pdf"
                elif filename[-4:] != '.pdf':
                    filename = filename + '.pdf'

                self.make_pdf(filename)

    def on_key_press_remove_mode(self, event):
        if event.key == 'e' or event.key == 't':
            # reset the plot instructions
            self.remove_mode = False
            self.plot_instructions()
            print('\nMain Menu')
            self.print_instructions()
            if event.key == 't':
                # save what is removed
                self.res_indexes_removed.update(self.res_indexes_staged)
                # reset the combined plot
                self.combined_plot(ax_combined=self.ax_combined)
            self.res_indexes_staged = set()
            # this removes the staged points (red "X"s) from the plot
            self.plot_staged_for_removal(ax_combined=self.ax_combined)
            plt.draw()
        elif event.key == 'x':
            # remove the selected point
            self.res_indexes_staged.add(self.plot_index)
            self.plot_staged_for_removal(ax_combined=self.ax_combined)
            plt.draw()
        elif event.key == 'z':
            # clear the staged points
            if self.plot_index in self.res_indexes_staged:
                self.res_indexes_staged.remove(self.plot_index)
                self.plot_staged_for_removal(ax_combined=self.ax_combined)
                plt.draw()
            else:
                print(f"Res-Index {self.plot_index} not staged for removal")

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
                self.refresh_plot(autoscale=False)
                return
        if event.button == 3:
            if self.shift_is_held:
                print("overriding point selection", event.xdata)
                # print(self.chan_freqs[:,self.plot_index][50])
                # print((self.res_index_override == self.plot_index).any())
                if (self.res_index_override == self.plot_index).any():
                    replace_index = np.argwhere(
                        self.res_index_override == self.plot_index)[0][0]
                    new_freq = np.argmin(
                        np.abs(event.xdata - self.chan_freqs[:, self.plot_index] / 10 ** 6))
                    self.override_freq_index[replace_index] = np.int(new_freq)

                else:
                    self.res_index_override = np.append(
                        self.res_index_override, np.int(np.asarray(self.plot_index)))
                    # print(self.res_index_override)
                    new_freq = np.argmin(
                        np.abs(event.xdata - self.chan_freqs[:, self.plot_index] / 10 ** 6))
                    # print("new index is ",new_freq)
                    self.override_freq_index = np.append(
                        self.override_freq_index, np.int(np.asarray(new_freq)))
                    # print(self.override_freq_index)
                self.update_min_index()
                self.refresh_plot()

    def make_pdf(self, filename):
        pdf_pages = PdfPages(filename)
        for i in tqdm.tqdm(range(0, self.chan_freqs.shape[1]), ascii=True):
            self.plot_index = i
            self.refresh_plot()
            pdf_pages.savefig(self.fig)
        pdf_pages.close()


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
            print(ip.res_index_override.shape)
            for i in range(0, len(ip.res_index_override)):
                ip.min_index[ip.res_index_override[i]
                ] = ip.override_freq_index[i]
            new_freqs = f[(ip.min_index, np.arange(0, f.shape[1]))]
        else:
            min_index = np.argmin(np.abs(z) ** 2, axis=0)
            new_freqs = f[(min_index, np.arange(0, f.shape[1]))]
    else:  # find the max of dIdQ
        print("centering on max dIdQ")
        if interactive:
            ip = InteractivePlot(f, z, look_around, find_min=False)
            for i in range(0, len(ip.res_index_override)):
                ip.min_index[ip.res_index_override[i]
                ] = ip.override_freq_index[i]
            new_freqs = f[(ip.min_index, np.arange(0, f.shape[1]))]
        else:
            min_index = find_max_didq(z, look_around)
            new_freqs = f[(min_index, np.arange(0, f.shape[1]))]
    return new_freqs


class InteractivePowerTuningPlot(object):
    """
    special interactive plot for tune the readout power of resonators based on fist to their non-linearity parameter
    f (frequencies of iq_sweep shape n_pts_iq_sweep x n_res x (optionally n_powers)
    z complex number where i is real and q is imaginary part shape n_pts_iq_sweep x n_res,  n_powers
    fitted_a_mag from magnitude fits
    fitted_a_iq from iq fits (one of two fitted as must be provided
    attn_levels list or array of power levels
    desired_a the desired non-linearity parameter to have powers set to
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
        self.res_index_override = np.asarray((), dtype=np.int16)
        self.override_freq_index = np.asarray((), dtype=np.int16)
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
                print("you must supply fits to the non-linearity parameter one of two fit variables")
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
                                         self.Qs_fit_iq[:, self.plot_index, self.power_index], label="Fit")
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
        print("Hold shift and right click on the bottom plot to override picked power level")
        print("or hold shift and press enter to override picked power level to the current plotted power level")
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

        if event.key == 'enter':  # is this still needed on any os?
            # print("enter pressed")
            if self.shift_is_held:
                self.bif_levels[self.plot_index] = -self.attn_levels[self.power_index]
                self.refresh_plot()

        if event.key == 'shift+enter':
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
    ip = InteractivePowerTuningPlot(f, z, attn_levels, fitted_a_mag=fitted_a_mag,
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

import gc
import time
import copy
import platform
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, TextBox
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import tqdm

from submm.KIDs.res.utils import colorize_text, text_color_matplotlib, autoscale_from_data

'''
Tools for identifiy what resonators are what from data 
where you turn on leds in front of resonators 
modified from res/sweep_tools.py
'''




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

        # set in self.update()
        self.xys = None
        self.update()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def update(self):
        self.xys = self.collection.get_offsets()

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def plot_reset(self):
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)

    def disconnect(self):
        self.lasso.disconnect_events()
        self.plot_reset()
        self.canvas.draw_idle()


class InteractivePlot(object):
    """
    interactive plot for plot many resonators iq data
    chan freqs and z should have dimension n_iq points by n_res
    also can provide multiple sweeps for plotting by adding extra dimension
    i.e. chan freqs and z could have dimension n_iq points by n_res by n_rows by n_cols
    combined data should have dimension n_res by n_different_types of data
    pixel_freqs should have shape n_resontors per pixel by m pixels
    """

    log_y_data_types = {'chi_sq'}
    key_font_size = 9

    flags_types_default = ["collision", "shallow", 'no-res', 'remove', 'other']

    def __init__(self, chan_freqs, z_baseline, z, pixel_locations_x = None,pixel_locations_y = None,
                 pixel_index = None,group_index = None,assigned_group_index = None,assigned_res_index = None,
                 assigned_pixel_index = None,pixel_freqs = None,look_around=2, stream_data=None, retune=True, find_min=True,
                 combined_data=None, combined_data_names=None, sweep_labels=None, sweep_line_styles=None,
                 combined_data_format=None, flags=None, flags_types=None, plot_title=None, plot_frames=True,
                 verbose=True,n_detectors_per_led = 1):
                

        self.z_baseline = z_baseline
        self.z = z
        self.chan_freqs = chan_freqs
        self.pixel_locations_x = pixel_locations_x
        self.pixel_locations_y = pixel_locations_y
        self.pixel_index = pixel_index
        self.group_index = group_index
        if assigned_group_index is None:
            self.assigned_group_index = -1*np.ones(self.chan_freqs.shape[1],dtype=int)
            self.assigned_res_index = -1*np.ones(self.chan_freqs.shape[1],dtype=int)
            self.assigned_pixel_index = -1*np.ones(self.chan_freqs.shape[1],dtype=int)
        else:
            self.assigned_group_index = assigned_group_index
            self.assigned_res_index = assigned_res_index
            self.assigned_pixel_index = assigned_pixel_index
        self.pixel_freqs = pixel_freqs
        self.designed_freqs = np.empty(chan_freqs.shape[1])
        self.designed_freqs[:] = np.nan
        self.assigned = np.zeros(combined_data.shape[1])
        # for reloaded data
        if self.pixel_freqs is not None and self.assigned_group_index.any()>-1: 
            for i in range(0,self.chan_freqs.shape[1]):
                self.plot_index = i
                if self.assigned_group_index[self.plot_index] >-1:
                    self.combined_data_index = np.where(np.logical_and(self.group_index == self.assigned_group_index[self.plot_index],
                                                                       self.pixel_index == self.assigned_pixel_index[self.plot_index]))[0][0]
                    self.assigned[self.combined_data_index] = 1
                    self.designed_freqs[self.plot_index] = self.pixel_freqs[self.assigned_res_index[self.plot_index],
                                                                    self.combined_data_index]
        self.plot_index = 0
        self.combined_data_index = 0
            #for i in range(0,chan_freqs.shape[1]):
            #    if self.assigned_group_index[i] > 0 and self.assigned_pixel_index[i] > 0 and self.assigned_res_index[i] >0:
            #        for k in range(0,self.pixel_freqs.shape[1]):
            #            if self.assigned_group_index[i] == self.group_index[k] and self.assigned_pixel_index[i] == pixel_index[k]:
            #                self.designed_freqs[i] = self.pixel_freqs[self.assigned_res_index[i],k]
            #                self.designed_freqs[self.plot_index] = self.pixel_freqs[self.assigned_res_index[self.plot_index],
            #                                                        self.combined_data_index]
        self.measured_freqs = self.chan_freqs[self.chan_freqs.shape[0]//2,:]
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
        self.lasso_mode = False
        self.lasso_start_time = 0.0
        self.selector = None
        self.update_min_index()
        self.cmap_name = 'cividis'
        self.cmap = plt.get_cmap(self.cmap_name)
        self.norm = mpl.colors.Normalize(vmin=np.min(self.combined_data[self.plot_index,:]),
                                         vmax=np.max(self.combined_data[self.plot_index,:]))
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        if flags is None:
            self.flags = []
            for i in range(chan_freqs.shape[1]):
                self.flags.append([])
        else:
            self.flags = flags
        if flags_types is None:
            self.flags_types = self.flags_types_default
        else:
            self.flags_types = flags_types
        self.verbose = verbose
        self.all_flags = set(self.flags_types)
        self.flag_type_index = 0
        if retune:
            self.combined_data_names = ['min index']
        # data remove variables for the interactive plot
        self.res_indexes_removed = set() 
        self.res_indexes_staged = {}
        self.combined_x_width_over_y_height = 1.0

        # set up plot
        top = 0.94
        bottom = 0.05
        left = 0.08
        right = 0.99
        x_space = 0.075
        y_space = 0.09
        x_width_total = right - left - x_space
        y_height_total = top - bottom - y_space
        
        
        
        self.fig = plt.figure(1, figsize=(12, 9))
        # the magnitude plot - upper left quadrant
        led_map_x_width = 0.3
        led_map_y_height = 0.25
        mag_x_width = 0.3
        mag_y_height = 0.25
        # [a,b,c,d] a,b is bottomleft cord c is width d is height
        led_map_figure_coords = [left, top -led_map_y_height, mag_x_width, mag_y_height]
        mag_figure_coords = [left, top -led_map_y_height-y_space-mag_y_height, mag_x_width, mag_y_height]
        led_heat_map_figure_coords = [left+led_map_x_width+x_space, top -led_map_y_height, mag_x_width, led_map_y_height]
        # the IQ plot - upper right quadrant
        iq_x_width = 0.3#x_width_total - mag_x_width
        iq_y_height = mag_y_height
        iq_figure_coords = [left + mag_x_width + x_space, top - iq_y_height-y_space-led_map_y_height, iq_x_width, iq_y_height]
        # the Key Figure area for the interactive plot instructions-key
        key_x_width = 0.2
        key_y_height = mag_y_height+y_space+led_map_y_height
        combined_x_width = 0.8
        key_figure_coords = [right - key_x_width, top - mag_y_height-y_space-led_map_y_height, key_x_width, key_y_height]
        # the combined data plot - lower plane
        combined_y_height = 0.2
        self.combined_x_width_over_y_height = combined_x_width / combined_y_height #for clicking
        self.led_map_x_width_over_y_height = led_map_x_width / led_map_y_height #for clicking
        self.iq_x_width_over_y_height = iq_x_width / iq_y_height #for clicking    
        combined_figure_coords = [left, bottom, combined_x_width, combined_y_height]

        if plot_title is not None:
            self.fig.suptitle(plot_title, y=0.99)
        self.ax_led_map = self.fig.add_axes(led_map_figure_coords, frameon=plot_frames)
        self.ax_led_map.axis('equal')
        self.ax_led_map.set_title('LED assignment map')
        self.ax_led_heat_map = self.fig.add_axes(led_heat_map_figure_coords, frameon=plot_frames)
        self.ax_led_heat_map.axis('equal')
        self.ax_led_heat_map.set_title("Resonator response to LEDs")
        self.ax_mag = self.fig.add_axes(mag_figure_coords, frameon=plot_frames)
        self.ax_mag.set_ylabel("Power (dB)")
        self.ax_mag.set_xlabel("Frequency (MHz)")
        self.ax_iq = self.fig.add_axes(iq_figure_coords, frameon=plot_frames)
        self.ax_iq.set_ylabel("As made frequency (GHz)")
        self.ax_iq.set_xlabel("Designed frequency (GHz)")
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
            sweep_labels = ["LED on"]
            for i in range(1, self.z.shape[2]):
                sweep_labels.append("Data " + str(i + 1))
        self.mag_line, = self.ax_mag.plot(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                          20 * np.log10(np.abs(self.z[:, self.plot_index, self.combined_data_index])), mec="k",
                                            label=sweep_labels[self.combined_data_index])
        self.baseline_mag, = self.ax_mag.plot(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                              20 * np.log10(np.abs(self.z_baseline[:, self.plot_index])),label = "LED off")

        self.ax_leg = self.ax_mag.legend()

        self.led_locs, = self.ax_led_map.plot(self.pixel_locations_x,self.pixel_locations_y,"h",color = "C1",label = "Unassigned pixels")
        self.assigned_pixel_locations_x = []
        self.assigned_pixel_locations_y = []
        for i in range(0,len(self.assigned)):
            if self.assigned[i]:
                self.assigned_pixel_locations_x.append(self.pixel_locations_x[i])
                self.assigned_pixel_locations_y.append(self.pixel_locations_y[i])
        self.led_locs_assigned, =  self.ax_led_map.plot(self.assigned_pixel_locations_x,self.assigned_pixel_locations_y,"h",color = "C0",label = "Assigned pixels")
        self.current_led_locs, = self.ax_led_map.plot(self.pixel_locations_x[0],self.pixel_locations_y[0],"h",color = "k",label = "Current pixel")
        self.ax_led_map_leg = self.ax_led_map.legend(ncol = 2)

        #self.led_locs_2, = self.ax_led_heat_map.plot(-1*np.asarray(self.pixel_locations_x),self.pixel_locations_y,"h")
        #self.led_locs_assigned_2, =  self.ax_led_heat_map.plot(-1*np.asarray(self.assigned_pixel_locations_x),self.assigned_pixel_locations_y,"h")
        #self.current_led_locs_2, = self.ax_led_heat_map.plot(-1*np.asarray(self.pixel_locations_x[0]),self.pixel_locations_y[0],"h")

        #self._iq_lines = [self.ax_iq.plot(
        #    self.Is[:, self.plot_index, i], self.Qs[:, self.plot_index, i], sweep_line_styles[i], mec="k",
        #    label=sweep_labels[i])
        #    for i in range(0, self.z.shape[2])]
        
        self.designed_v_measured, = self.ax_iq.plot(self.designed_freqs,self.measured_freqs/10**9,"."
                                                    ,label = "Assigned resonators")
        self.expected_frequencies, = self.ax_iq.plot(self.pixel_freqs[:,self.combined_data_index],
                                                     self.pixel_freqs[:,self.combined_data_index],"*",
                                                     label = "Expected frequencies\nfor current pixel")
        self.ax_iq_leg = self.ax_iq.legend()
        if self.retune:
            self.p1, = self.ax_mag.plot(self.chan_freqs[self.min_index[self.plot_index], self.plot_index] / 10 ** 6,
                                        20 * np.log10(np.abs(self.z[self.min_index[self.plot_index], self.plot_index, self.combined_data_index])),
                                        '*', markersize=15, color='darkorchid')
            #self.p2, = self.ax_iq.plot(self.Is[self.min_index[self.plot_index], self.plot_index, 0],
            #                           self.Qs[self.min_index[self.plot_index], self.plot_index, 0], '*', markersize=15)

        center_freq_MHz = self.chan_freqs[self.chan_freqs.shape[0] // 2, self.plot_index] / 10 ** 6
        self.ax_mag.set_title(f'{"%3.3f" % center_freq_MHz} MHz - Resonator Index: {self.plot_index:03}')
        if self.retune:
            self.ax_iq.set_title("Look Around Points " + str(self.look_around))
        # Say 'Hello' to the User
        if self.verbose:
            print("\nInteractive Resonance Plotting Activated")
        self.print_instructions()

        # combined plot variables used in the first initialization
        self.combined_data_points = None
        self.unassigned_combined_data_points = None
        self.combined_data_highlight = None
        self.combined_data_crosshair_x = None
        self.combined_data_crosshair_y = None
        self.combined_staged_for_removal = None
        self.combined_staged_flagged = None
        self.combined_data_legend = None
        self.res_indexes = None
        self.combined_data_values = None
        self.combined_values_this_index = None
        self.pop_up_text = None
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

        self.led_heat_map = None
        self.led_heat_map_plot(self.ax_led_heat_map)
        plt.show(block=True)

    def led_heat_map_plot(self,ax_led_heat_map):
        if self.led_heat_map is not None:
            self.led_heat_map.remove()
            self.led_heat_map = None
        gc.collect()
        color_array = self.scalarMap.to_rgba(self.combined_data[self.plot_index,:])
        #print(color_array)
        self.led_heat_map = ax_led_heat_map.scatter(x=self.pixel_locations_x, y=self.pixel_locations_y, s=50,
                                                        color=color_array, marker='h')
        self.led_heat_map_current, = self.ax_led_heat_map.plot(self.pixel_locations_x[self.combined_data_index],
                                                         self.pixel_locations_y[self.combined_data_index],'h',
                                                         fillstyle = 'none',mec = "black",markersize = 10,
                                                         markeredgewidth = 2)
        self.led_heat_map_current_2, = self.ax_led_heat_map.plot(self.pixel_locations_x[self.combined_data_index],
                                                         self.pixel_locations_y[self.combined_data_index],'h',
                                                         fillstyle = 'none',mec = "orange",markersize = 10,
                                                               markeredgewidth = 1)
        
    def combined_plot(self, ax_combined):
        if self.combined_data_points is not None:
            self.combined_data_points.remove()
            self.combined_data_points = None
        if self.unassigned_combined_data_points is not None:
            self.unassigned_combined_data_points.remove()
            self.unassigned_combined_data_points = None
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
        self.combined_values_this_index = self.combined_data_values[:, self.combined_data_index]
        self.unassigned_combined_values_this_index = []
        self.unassigned_res_indexes = []
        for res_index in range(0,self.chan_freqs.shape[1]):
            if self.assigned_pixel_index[res_index] == -1:
                self.unassigned_combined_values_this_index.append(self.combined_data_values[res_index,self.combined_data_index])
                self.unassigned_res_indexes.append(res_index)
        
        
        self.combined_data_points = ax_combined.scatter(x=self.res_indexes, y=self.combined_values_this_index, s=40,
                                                        color="C0", marker='o')         
        self.unassigned_combined_data_points = ax_combined.scatter(x=self.unassigned_res_indexes,
                                                                   y=self.unassigned_combined_values_this_index, s=40,
                                                                   color="C1", marker='o')
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

    def get_stage_plot_points(self):
        staged_indexes = sorted(self.res_indexes_staged.keys())
        staged_values = [self.combined_data[staged_index, self.combined_data_index]
                         for staged_index in staged_indexes]
        removed_indexes = []
        flag_indexes = []
        removed_values = []
        flag_values = []
        for staged_index, staged_value in zip(staged_indexes, staged_values):
            flags = self.res_indexes_staged[staged_index]
            if 'remove' in flags or 'no-res' in flags:
                removed_indexes.append(staged_index)
                removed_values.append(staged_value)
            else:
                flag_indexes.append(staged_index)
                flag_values.append(staged_value)
        return removed_indexes, removed_values, flag_indexes, flag_values

    def plot_staged_for_removal(self, ax_combined):
        if self.combined_staged_for_removal is not None:
            self.combined_staged_for_removal.remove()
            self.combined_staged_for_removal = None
        if self.combined_staged_flagged is not None:
            self.combined_staged_flagged.remove()
            self.combined_staged_flagged = None
        if self.res_indexes_staged:
            removed_indexes, removed_values, flag_indexes, flag_values = self.get_stage_plot_points()
            if removed_indexes:
                self.combined_staged_for_removal, = ax_combined.plot(removed_indexes, removed_values, 'x', ls='None',
                                                                     markersize=12, color='firebrick')
            if flag_indexes:
                self.combined_staged_flagged, = ax_combined.plot(flag_indexes, flag_values, '1', ls='None',
                                                                 markersize=12, color='deepskyblue')

    def refresh_plot(self, autoscale=True):
        if len(self.flags[self.plot_index]) > 0:
            self.ax_mag.set_facecolor('lightyellow')
            self.ax_iq.set_facecolor('lightyellow')
        else:
            self.ax_mag.set_facecolor("None")
            self.ax_iq.set_facecolor("None")
        
        self.mag_line.set_data(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                          20 * np.log10(np.abs(self.z[:, self.plot_index, self.combined_data_index])))
        self.baseline_mag.set_data(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                              20 * np.log10(np.abs(self.z_baseline[:, self.plot_index])))
        if self.retune:
            self.p1.set_data(self.chan_freqs[self.min_index[self.plot_index], self.plot_index] / 10 ** 6,
                             20 * np.log10(np.abs(self.z[self.min_index[self.plot_index], self.plot_index, self.combined_data_index])))


        self.ax_mag.relim()
        self.ax_mag.autoscale()
        center_freq_MHz = self.chan_freqs[self.chan_freqs.shape[0] // 2, self.plot_index] / 10 ** 6
        self.ax_mag.set_title(f'{"%3.3f" % center_freq_MHz} MHz - Resonator Index: {self.plot_index:03}')

        self.current_led_locs.set_data(self.pixel_locations_x[self.combined_data_index],
                                       self.pixel_locations_y[self.combined_data_index])
        self.assigned_pixel_locations_x = []
        self.assigned_pixel_locations_y = []
        for i in range(0,len(self.assigned)):
            if self.assigned[i]:
                self.assigned_pixel_locations_x.append(self.pixel_locations_x[i])
                self.assigned_pixel_locations_y.append(self.pixel_locations_y[i])
        self.led_locs_assigned.set_data(self.assigned_pixel_locations_x,self.assigned_pixel_locations_y)

        self.led_heat_map_current.set_data(self.pixel_locations_x[self.combined_data_index],
                                           self.pixel_locations_y[self.combined_data_index])
        self.led_heat_map_current_2.set_data(self.pixel_locations_x[self.combined_data_index],
                                           self.pixel_locations_y[self.combined_data_index])
        if self.retune:
            self.ax_iq.set_title("Look Around Points " + str(self.look_around))


        if self.stream_data is not None:
            self.s2.set_data(np.real(self.stream_data[:, self.plot_index]),
                             np.imag(self.stream_data[:, self.plot_index]))

        self.designed_v_measured.set_data(self.designed_freqs,self.measured_freqs/10**9)
        self.expected_frequencies.set_data(self.pixel_freqs[:,self.combined_data_index],
                                           self.pixel_freqs[:,self.combined_data_index])
        self.ax_iq.relim()
        self.ax_iq.autoscale()

        # reset the led_heat_map_plot
        self.norm = mpl.colors.Normalize(vmin=np.min(self.combined_data[self.plot_index,:]),
                                         vmax=np.max(self.combined_data[self.plot_index,:]))
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        #print(self.combined_data[self.plot_index,:]/self.chan_freqs.shape[1])
        color_array = self.scalarMap.to_rgba(self.combined_data[self.plot_index,:])

        self.led_heat_map.set_color(color_array)
                
        if self.combined_data is not None:
            data_type = self.combined_data_names[self.combined_data_index]
            self.ax_combined.set_title(data_type, color='black')
            self.ax_combined.set_ylabel(data_type)
            if data_type in self.log_y_data_types:
                self.ax_combined.set_yscale('log')
            else:
                self.ax_combined.set_yscale('linear')
            # reset the combined plot data
            self.combined_values_this_index = self.combined_data_values[:, self.combined_data_index]
            new_offsets = np.column_stack((self.res_indexes, self.combined_values_this_index))
            self.combined_data_points.set_offsets(new_offsets)
            self.unassigned_combined_values_this_index = []
            self.unassigned_res_indexes = []
            for res_index in range(0,self.chan_freqs.shape[1]):
                if self.assigned_pixel_index[res_index] == -1:
                    self.unassigned_combined_values_this_index.append(self.combined_data_values[res_index,self.combined_data_index])
                    self.unassigned_res_indexes.append(res_index)
            
            new_offsets = np.column_stack((self.unassigned_res_indexes, self.unassigned_combined_values_this_index))
            self.unassigned_combined_data_points.set_offsets(new_offsets)
            
            # label for the value of the highlighted data point
            highlighted_value = self.combined_data[self.plot_index, self.combined_data_index]
            label = self.combined_data_format[self.combined_data_index].format(highlighted_value)
            x_pos = self.plot_index
            y_pos = highlighted_value
            self.combined_data_highlight.set_data(x_pos, y_pos)
            self.combined_data_legend.texts[0].set_text(label)
            if self.res_indexes_staged:
                removed_indexes, removed_values, flag_indexes, flag_values = self.get_stage_plot_points()
                if removed_indexes:
                    self.combined_staged_for_removal.set_data(removed_indexes, removed_values)
                if flag_indexes:
                    self.combined_staged_flagged.set_data(flag_indexes, flag_values)
            if autoscale:
                self.ax_combined.set_xlim(autoscale_from_data(self.res_indexes))
                plot_min, plot_max = autoscale_from_data(self.combined_values_this_index[np.isfinite(self.combined_values_this_index)],
                                                         log_scale=data_type in self.log_y_data_types)
                self.ax_combined.set_ylim((plot_min, plot_max))
            self.combined_data_crosshair_x.set_xdata(x_pos)
            self.combined_data_crosshair_y.set_ydata(y_pos)
        # lasso tool
        if self.lasso_mode:
            self.selector.update()
        # Pop-up text window
        if self.pop_up_text is not None:
            self.pop_up_text.remove()
            self.pop_up_text = None
        if self.lasso_mode and time.time() < self.lasso_start_time + 15:
            self.popup_lasso()
        elif self.flags[self.plot_index]:
            pop_up_text = f"Res {self.plot_index} Flagged: {sorted(self.flags[self.plot_index])}"
            self.pop_up_text = self.ax_combined.text(0.5, 0.8, pop_up_text,
                                                     transform=self.ax_combined.transAxes, ha="center", va="center",
                                                     size=16, color="black", family="monospace",
                                                     bbox=dict(facecolor='Red', alpha=0.6))
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

    def popup_lasso(self):
        self.pop_up_text = self.ax_combined.text(0.5, 0.8, "Press enter to accept selection",
                                                 transform=self.ax_combined.transAxes, ha="center", va="center",
                                                 size=16, color="yellow", family="monospace",
                                                 bbox=dict(facecolor='black', alpha=0.6))

    def instructions(self):
        instructions = [("left-arrow", "change resonator left", 'green'),
                        ("right-arrow", "change resonator right", 'cyan')]
        if self.retune:
            instructions.extend([('down-arrow', 'change look around points down', 'yellow'),
                                 ('up-arrow', 'change look around points up', 'blue')])
            if platform.system() == 'Darwin':
                instructions.extend([("Hold any letter a key and right click on the magnitude plot",
                                      "to override tone position", 'black')])
            else:
                instructions.extend([("Hold 'shift' and right click on the magnitude plot",
                                      "to override tone position", 'black')])
        if self.combined_data is not None:
            flag_type = self.flags_types[self.flag_type_index]
            instructions.extend([('down-arrow', 'cycle through leds', 'yellow'),
                                 ('up-arrow', 'cycle through leds', 'blue'),
                                 ('double-click', 'go to the resonator/\nled position', 'black'),
                                 ('spacebar', 'assign current resonator', 'red'),
                                 ('D-key', "delete current resonator's\n assignments", 'white'),
                                 ('A-key', 'run auto assignment', 'purple')])
                                 #('Y-key', 'change y limits', 'green')   ])
            #if self.lasso_mode:
            #    instructions.extend([('Enter-Key', f'Stage Lassoed, flag: {flag_type}', 'red'),
            #                         ('B-Key', "Exit Lasso-selection", 'white'),
            #                         ('Click+Hold', "Draw to Lasso a group", 'black')])
            #else:
            #    instructions.extend([('F-key', f'stage for flag: {flag_type}', 'red'),
            #                         ('Z-Key', f'un-stage for flag: {flag_type}', 'cyan'),
            #                         ('B-Key', "Lasso-select to stage", 'white'),
            #                         ('D-key', 'change flag mode', 'yellow'),
            #                         ('T-Key', 'commit all staged flagging', 'blue'),
            #                         ('E-Key', 'clear all staged flagging', 'green')])
            #if self.retune:
            #    instructions.extend(["Warning both look around points and combined data are mapped to " +
            #                         "up and down arrows, consider not returning and plotting combined " +
            #                         "data at the same time"])
        #instructions.append(("W-key", "write a pdf of all resonators", 'cyan'))

        return instructions

    def plot_instructions(self):
        instructions = self.instructions()
        self.ax_key.clear()
        steps_per_item = 1.5
        y_step = 0.9 / (steps_per_item * float(len(instructions)))
        y_now = 0.95
        for key_press, description, color in instructions:
            self.ax_key.text(0.4, y_now, key_press.center(13), color=text_color_matplotlib[color],
                             ha='right', va='center', size=self.key_font_size, weight="bold",
                             family='monospace', bbox=dict(color=color, ls='-', lw=2.0, ec='black'))
            self.ax_key.text(0.45, y_now, description, color='black',
                             ha='left', va='center', size=self.key_font_size - 2)
            y_now -= steps_per_item * y_step
        self.ax_key.set_xlim(0, 1)
        self.ax_key.set_ylim(0, 1)
        if self.lasso_mode:
            self.ax_key.set_title('Lasso-Selection')
        else:
            self.ax_key.set_title('Main Menu')
        plt.draw()

    def print_instructions(self):
        instructions = self.instructions()
        for key_press, description, color in instructions:
            if color == 'black':
                text_color = 'white'
            else:
                text_color = 'black'
            color_text = colorize_text(text=key_press.capitalize().center(20), style_text='bold',
                                       color_text=text_color, color_background=color)
            if self.verbose:
                print(f'{color_text} : {description}')

    def set_flag_index(self, flag_index: int = None):
        if flag_index is None:
            # get the next flag index
            self.flag_type_index = (self.flag_type_index + 1) % len(self.flags_types)
        else:
            self.flag_type_index = flag_index
        if self.verbose:
            print(f'\n Flag mode is now: "{self.flags_types[self.flag_type_index]}"')
        self.print_instructions()
        self.plot_instructions()
        self.refresh_plot()

    def get_flag_type(self):
        flag_type = self.flags_types[self.flag_type_index]
        if flag_type == 'other':
            # allow for custom flag types to be added
            flag_type_new = input("Enter flag type: ").lower().strip()
            if flag_type_new and flag_type_new not in self.flags_types:
                # to new resonator type
                self.flags_types.append(flag_type_new)
                self.all_flags.add(flag_type_new)
                self.set_flag_index(flag_index=len(self.flags_types) - 1)
                flag_type = flag_type_new
        return flag_type

    def on_key_press(self, event):
        #print(event.key)
        if platform.system().lower() == 'darwin' and event.key == 'a':
            self.shift_is_held = True
        elif platform.system().lower() != 'darwin' and event.key == 'shift':
            self.shift_is_held = True
        # items on all menus
        if event.key == 'right':
            if self.plot_index == self.res_indexes[-1]:
                self.plot_index = self.res_indexes[0]
            else:
                self.plot_index = self.res_indexes[np.where(self.res_indexes == self.plot_index)[-1] + 1][-1]
            self.refresh_plot(autoscale = False)

        elif event.key == 'left':
            if self.plot_index == self.res_indexes[0]:
                self.plot_index = self.res_indexes[-1]
            else:
                self.plot_index = self.res_indexes[np.where(self.res_indexes == self.plot_index)[-1] - 1][0]
            self.refresh_plot(autoscale = False)

        elif event.key == 'up':
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

        elif event.key == 'down':
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
        # Writing an output file
        elif event.key == 'w':
            if self.verbose:
                print("saving to pdf")
            filename = input("enter filename for pdf: ")
            if filename == '':
                filename = "res_plots.pdf"
            elif filename[-4:] != '.pdf':
                filename = filename + '.pdf'
            self.make_pdf(filename)

        elif event.key == 'y':
            data_type = self.combined_data_names[self.combined_data_index]
            plot_min, plot_max = autoscale_from_data(self.combined_values_this_index[np.isfinite(self.combined_values_this_index)],
                                                         log_scale=data_type in self.log_y_data_types)
            pop_up = PopUpDataEntry("Enter upper y limit",str(plot_max))
            upper = float(pop_up.value)
            pop_up = PopUpDataEntry("Enter lower y limit",str(plot_min))
            lower = float(pop_up.value)
            self.ax_combined.set_ylim((lower, upper))
            self.ax_combined_hist.set_ylim((lower, upper))
            self.refresh_plot(autoscale = False)

        elif event.key == ' ':#spacebar
            pop_up = PopUpDataEntry("Enter Resonator Group index\n0 for PX1, 1 for PX2",str(self.group_index[self.combined_data_index]))
            self.assigned_group_index[self.plot_index] = int(pop_up.value)
            res_freq = self.chan_freqs[self.chan_freqs.shape[0] // 2,self.plot_index]
            if res_freq < np.mean(self.pixel_freqs[:,self.combined_data_index]*10**9):
                guess = 0
            else:
                guess = 1
            pop_up = PopUpDataEntry("Enter resonator index\n0 for left, 1 for right",str(guess))
            self.assigned_res_index[self.plot_index] = int(pop_up.value)
            if self.pixel_index is not None:
                pop_up = PopUpDataEntry("Enter Resonator index",str(self.pixel_index[self.combined_data_index]))
            else:
                pop_up = PopUpDataEntry("Enter Resonator index",str(0))
            self.assigned_pixel_index[self.plot_index] = int(pop_up.value)
            # check that combined_data index is correct for this pixel
            self.designed_freqs[self.plot_index] = self.pixel_freqs[self.assigned_res_index[self.plot_index],
                                                                    self.combined_data_index]
            self.assigned[self.combined_data_index] = 1
            self.refresh_plot()
            plt.figure(1) #grab attention back to the plot
            plt.get_current_fig_manager().show()

        elif event.key == 'd':#delete index idetificaitona information
            self.assigned_group_index[self.plot_index] = -1
            self.assigned_res_index[self.plot_index] = -1
            self.assigned_pixel_index[self.plot_index] = -1
            self.designed_freqs[self.plot_index] = np.nan
            self.refresh_plot()


        elif event.key == 'a':#auto assign
            pop_up = PopUpDataEntry("Enter threshold for min index",str(150))
            threshold = int(pop_up.value)
            for i in range(0,self.chan_freqs.shape[1]):
                self.plot_index = i
                self.refresh_plot()
                if (self.combined_data[self.plot_index,:] < threshold).any():
                    self.combined_data_index = np.argmin(self.combined_data[self.plot_index,:])
                    self.refresh_plot()
                    self.assigned[self.combined_data_index] = 1
                    self.assigned_group_index[self.plot_index] = self.group_index[self.combined_data_index]
                    res_freq = self.chan_freqs[self.chan_freqs.shape[0] // 2,self.plot_index]
                    if res_freq < np.mean(self.pixel_freqs[:,self.combined_data_index]*10**9):
                        guess = 0
                    else:
                        guess = 1
                    self.assigned_res_index[self.plot_index] = guess
                    self.assigned_pixel_index[self.plot_index] = self.pixel_index[self.combined_data_index]
                    self.designed_freqs[self.plot_index] = self.pixel_freqs[self.assigned_res_index[self.plot_index],
                                                                    self.combined_data_index]
                else:
                    print("skipping resonator ",i)
            self.refresh_plot()

        # Flagging and removing interactions
        elif not self.lasso_mode:
            if event.key == 'f':
                current_flags = self.flags[self.plot_index]
                if current_flags:
                    if self.verbose:
                        print(f"Res Index {self.plot_index} current flags: {current_flags}")
                # stage the selected point for flagging
                flag_type = self.get_flag_type()
                # stage the selected point for flagging
                if self.plot_index not in self.res_indexes_staged.keys():
                    self.res_indexes_staged[self.plot_index] = set()
                self.res_indexes_staged[self.plot_index].add(flag_type)
                # reset the plot
                self.plot_staged_for_removal(ax_combined=self.ax_combined)
                self.refresh_plot(autoscale=False)
                if self.verbose:
                    print(f'Res Index {self.plot_index} staged flags: {sorted(self.res_indexes_staged[self.plot_index])}')
                committed_flags = self.flags[self.plot_index]
            elif event.key == 'd':
                # cycle the flag mode by one type
                self.set_flag_index()
            elif event.key == 'e' or event.key == 't':
                # reset the plot instructions
                self.plot_instructions()
                if self.verbose:
                    print('\nMain Menu')
                self.print_instructions()
                if event.key == 't':
                    # save what is staged
                    for res_index in self.res_indexes_staged.keys():
                        for flag in sorted(self.res_indexes_staged[res_index]):
                            if flag == 'remove':
                                self.res_indexes_removed.add(res_index)
                            if flag == 'no-res':
                                self.res_indexes_removed.add(res_index)
                                self.flags[res_index].append(flag)
                            else:
                                self.flags[res_index].append(flag)
                    # reset the combined plot
                    self.combined_plot(ax_combined=self.ax_combined,ax_combined_hist=self.ax_combined_hist)
                self.res_indexes_staged = {}
                # this removes the staged points (red "X"s) from the plot
                self.plot_staged_for_removal(ax_combined=self.ax_combined)
                self.refresh_plot()
            elif event.key == 'z':
                flag_type = self.get_flag_type()
                if self.plot_index in self.res_indexes_staged:
                    flags_staged = self.res_indexes_staged[self.plot_index]
                    if flag_type in flags_staged:
                        # clear the staged point
                        flags_staged.remove(flag_type)
                        if not flags_staged:
                            del self.res_indexes_staged[self.plot_index]
                        self.plot_staged_for_removal(ax_combined=self.ax_combined)
                        self.refresh_plot()
                    else:
                        if self.verbose:
                            print(f"Res-Index {self.plot_index} not staged for flag type {flag_type}")
                else:
                    if self.verbose:
                        print(f"Res-Index {self.plot_index} not staged.")
            # Lasso selection
            elif event.key == 'b':
                self.lasso_mode = True
                self.plot_instructions()
                if self.verbose:
                    print('\nLasso Mode')
                self.print_instructions()
                self.ax_combined.set_title("Click and Drag to select points with a lasso (you click and draw on the plot).",
                                           size=16, color="firebrick", family="monospace")
                if self.pop_up_text is not None:
                    self.pop_up_text.remove()
                    self.pop_up_text = None
                self.popup_lasso()
                self.fig.canvas.draw()
                # lasso selector
                self.selector = SelectFromCollection(ax=self.ax_combined, collection=self.combined_data_points)
                self.fig.canvas.mpl_connect("key_press_event", self.accept_lasso)
                self.lasso_start_time = time.time()

    def accept_lasso(self, event_lasso):
        if self.lasso_mode:
            if event_lasso.key == "enter" or (event_lasso.key == "b" and time.time() > self.lasso_start_time + 1.0):
                if event_lasso.key == "enter":
                    selected_indexes = [int(index_as_float) for index_as_float
                                        in self.selector.xys[self.selector.ind][:, 0]]
                    flag_type = self.get_flag_type()
                    for index in selected_indexes:
                        if index not in self.res_indexes_staged:
                            self.res_indexes_staged[index] = set()
                        self.res_indexes_staged[index].add(flag_type)
                self.lasso_mode = False
                self.plot_staged_for_removal(ax_combined=self.ax_combined)
                self.selector.disconnect()
                self.selector = None
                self.refresh_plot()
                self.plot_instructions()
                if self.verbose:
                    print('\nMain Menu')
                self.print_instructions()

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
                # get the radius of each point from the click
                if event.inaxes==self.ax_led_map or event.inaxes==self.ax_led_heat_map:
                    x_data_coords = self.pixel_locations_x - event.xdata
                    x_data_min, x_dat_max = self.ax_led_map.get_xlim()
                    x_data_range = x_dat_max - x_data_min
                    x_norm_coords = x_data_coords / x_data_range
                    x_yratio_coords = x_norm_coords * self.led_map_x_width_over_y_height
                    y_data_coords = self.pixel_locations_y - event.ydata
                    y_data_min, y_dat_max = self.ax_led_map.get_ylim()
                    y_data_range = y_dat_max - y_data_min
                    y_norm_coords = y_data_coords / y_data_range
                    radius_array = np.sqrt(x_yratio_coords ** 2 + y_norm_coords ** 2)
                    self.combined_data_index = np.argmin(radius_array)
                    self.refresh_plot(autoscale = True)
                elif event.inaxes==self.ax_iq:
                    print("hello")
                    x_data_coords = self.designed_freqs - event.xdata
                    x_data_min, x_dat_max = self.ax_iq.get_xlim()
                    x_data_range = x_dat_max - x_data_min
                    x_norm_coords = x_data_coords / x_data_range
                    x_yratio_coords = x_norm_coords * self.iq_x_width_over_y_height
                    y_data_coords = self.measured_freqs/10**9 - event.ydata
                    y_data_min, y_dat_max = self.ax_iq.get_ylim()
                    y_data_range = y_dat_max - y_data_min
                    y_norm_coords = y_data_coords / y_data_range
                    radius_array = np.sqrt(x_yratio_coords ** 2 + y_norm_coords ** 2)
                    self.plot_index = np.nanargmin(radius_array)
                    self.refresh_plot(autoscale = True)
                else:
                    x_data_coords = self.res_indexes - event.xdata
                    x_data_min, x_dat_max = self.ax_combined.get_xlim()
                    x_data_range = x_dat_max - x_data_min
                    x_norm_coords = x_data_coords / x_data_range
                    x_yratio_coords = x_norm_coords * self.combined_x_width_over_y_height
                    y_data_coords = self.combined_values_this_index - event.ydata
                    y_data_min, y_dat_max = self.ax_combined.get_ylim()
                    y_data_range = y_dat_max - y_data_min
                    y_norm_coords = y_data_coords / y_data_range
                    radius_array = np.sqrt(x_yratio_coords ** 2 + y_norm_coords ** 2)
                    self.plot_index = self.res_indexes[np.argmin(radius_array)]
                    self.refresh_plot(autoscale=False)
                return
        elif event.button == 3:
            if self.shift_is_held:
                if self.verbose:
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
                    # if self.verbose:
                    #     print(self.res_index_override)
                    new_freq = np.argmin(
                        np.abs(event.xdata - self.chan_freqs[:, self.plot_index] / 10 ** 6))
                    # if self.verbose:
                    #     print("new index is ",new_freq)
                    self.override_freq_index = np.append(
                        self.override_freq_index, np.int(np.asarray(new_freq)))
                    # if self.verbose:
                    #     print(self.override_freq_index)
                self.update_min_index()
                self.refresh_plot()

    def make_pdf(self, filename):
        pdf_pages = PdfPages(filename)
        for i in tqdm.tqdm(range(0, self.chan_freqs.shape[1]), ascii=True):
            self.plot_index = i
            self.refresh_plot()
            pdf_pages.savefig(self.fig)
        pdf_pages.close()



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

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


class InteractivePlot(object):
    """
    interactive plot for plot many resonators iq data
    chan freqs and z should have dimension n_iq points by n_res
    also can provide multiple sweeps for plotting by adding extra dimension
    i.e. chan freqs and z could have dimension n_iq points by n_res by n_rows by n_cols
    combined data should have dimension n_res by n_different_types of data
    pixel_freqs should have shape n_resontors per pixel by m pixels
    group_index is 0 or 1 for PX1 or PX2
    res_index is 0 or 1 for the lower or higher frequency resonator
    pixel_index is the pixel index i.e 1 through 144
    """

    log_y_data_types = {'chi_sq'}
    key_font_size = 9

    flags_types_default = ["collision", "shallow", 'no-res', 'remove', 'other']

    def __init__(self, chan_freqs, z_baseline, z,z_baseline_unmasked= None,z_unmasked = None, pixel_locations_x = None,pixel_locations_y = None,
                 pixel_index = None,group_index = None,assigned_group_index = None,assigned_res_index = None,
                 assigned_pixel_index = None,pixel_freqs = None,look_around=2, stream_data=None, retune=True, find_min=True,
                 combined_data=None, combined_data_2 = None, combined_data_names=None, sweep_labels=None, sweep_line_styles=None,
                 combined_data_format=None, flags=None, flags_types=None, plot_title=None, plot_frames=True,
                 verbose=True,n_detectors_per_led = 1,pixel_dark = None):
                

        self.z_baseline = z_baseline
        self.z_baseline_unmasked = z_baseline_unmasked
        self.z = z
        self.z_unmasked = z_unmasked
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
        self.assigned_1 = np.zeros(combined_data.shape[1])
        self.assigned_2 = np.zeros(combined_data.shape[1])
        self.assigned = np.zeros(combined_data.shape[1])
        if pixel_dark is not None:
            self.pixel_dark = pixel_dark
        else:
            self.pixel_dark = np.zeros(combined_data.shape[1])
        
        self.plot_index = 0
        self.combined_data_index = 0
        self.measured_freqs = self.chan_freqs[self.chan_freqs.shape[0]//2,:]
        plt.rcParams['keymap.fullscreen'] = ['shift+=']  # remove ('f', 'ctrl+f'), make +

        self.Is = np.real(self.z)
        self.Qs = np.imag(self.z)
        self.find_min = find_min
        self.retune = retune
        self.combined_data = combined_data
        self.refresh_assigned_values_for_plotting()
        self.combined_data_2 = combined_data_2
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
        self.cmap_name = 'cividis'
        self.cmap = plt.get_cmap(self.cmap_name)
        self.norm = mpl.colors.Normalize(vmin=np.min(self.combined_data[self.plot_index,:]),
                                         vmax=np.max(self.combined_data[self.plot_index,:]))
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.norm_2 = mpl.colors.Normalize(vmin=np.min(self.combined_data_2[self.plot_index,:]),
                                         vmax=np.max(self.combined_data_2[self.plot_index,:]))
        self.scalarMap_2 = cm.ScalarMappable(norm=self.norm_2, cmap=self.cmap)
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
        x_space_top = 0.01
        y_space = 0.09
        x_width_total = right - left - x_space
        y_height_total = top - bottom - y_space
        
        
        
        self.fig = plt.figure(1, figsize=(12, 9))
        # the magnitude plot - upper left quadrant
        led_map_x_width = 0.29
        led_map_y_height = 0.25
        mag_x_width = 0.3
        mag_y_height = 0.25
        # [a,b,c,d] a,b is bottomleft cord c is width d is height
        led_map_figure_coords = [left, top -led_map_y_height, led_map_x_width, mag_y_height]
        mag_figure_coords = [left, top -led_map_y_height-y_space-mag_y_height, mag_x_width, mag_y_height]
        led_heat_map_figure_coords = [left+led_map_x_width+x_space_top, top -led_map_y_height, led_map_x_width, led_map_y_height]
        led_heat_map_2_figure_coords = [left+led_map_x_width*2+x_space_top*2, top -led_map_y_height, led_map_x_width, led_map_y_height]
        # the IQ plot - upper right quadrant
        iq_x_width = 0.3#x_width_total - mag_x_width
        iq_y_height = mag_y_height
        iq_figure_coords = [left + mag_x_width + x_space, top - iq_y_height-y_space-led_map_y_height, iq_x_width, iq_y_height]
        # the Key Figure area for the interactive plot instructions-key
        key_x_width = 0.2
        key_y_height = mag_y_height + y_space
        combined_x_width = 0.8
        key_figure_coords = [right - key_x_width, top - mag_y_height-y_space*2-led_map_y_height, key_x_width, key_y_height]
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
        self.ax_led_heat_map = self.fig.add_axes(led_heat_map_figure_coords, frameon=plot_frames)
        self.ax_led_heat_map.axis('equal')
        self.ax_led_heat_map.set_title("Min Index")
        self.ax_led_heat_map.tick_params(labelleft = False)
        self.ax_led_heat_map_2 = self.fig.add_axes(led_heat_map_2_figure_coords, frameon=plot_frames)
        self.ax_led_heat_map_2.axis('equal')
        self.ax_led_heat_map_2.set_title("Min Value")
        self.ax_led_heat_map_2.tick_params(labelleft = False)
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

        if self.z_baseline_unmasked is not None:
            self.baseline_mag_unmasked, = self.ax_mag.plot(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                                           20 * np.log10(np.abs(self.z_baseline_unmasked[:, self.plot_index])),
                                                           color = "deepskyblue",alpha = 1)
        self.baseline_mag, = self.ax_mag.plot(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                              20 * np.log10(np.abs(self.z_baseline[:, self.plot_index])),color = "C0",label = "LED off")
                
        if self.z_unmasked is not None:
            self.mag_line_unmasked, = self.ax_mag.plot(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                          20 * np.log10(np.abs(self.z_unmasked[:, self.plot_index, self.combined_data_index])),
                                                       color = "gold",alpha = 1)
        self.mag_line, = self.ax_mag.plot(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                          20 * np.log10(np.abs(self.z[:, self.plot_index, self.combined_data_index])), color = "C1",
                                            label=sweep_labels[self.combined_data_index])
        

        self.ax_leg = self.ax_mag.legend()

        self.led_locs, = self.ax_led_map.plot(self.pixel_locations_x,self.pixel_locations_y,"h",
                                              markeredgewidth = 0,markersize = 7.5,color = "C1",label = "Unassigned")
    
        self.led_locs_assigned_1, =  self.ax_led_map.plot(self.assigned_pixel_locations_x_1,self.assigned_pixel_locations_y_1,
                                                          "h",markerfacecolor = "C0",markerfacecoloralt = "C1",
                                                          markeredgewidth = 0,fillstyle = "left",markersize = 7.5,label = "Lower")
        self.led_locs_assigned_2, =  self.ax_led_map.plot(self.assigned_pixel_locations_x_2,self.assigned_pixel_locations_y_2,
                                                          "h",markerfacecolor = "C0",markerfacecoloralt = "C1",
                                                          markeredgewidth = 0,fillstyle = "right",markersize = 7.5,color = "C5",label = "Higher")
        self.led_locs_assigned, =  self.ax_led_map.plot(self.assigned_pixel_locations_x,self.assigned_pixel_locations_y,
                                                        "h",markeredgewidth = 0, markersize = 7.5,color = "C0",label = "Both")
        #dark pixels
        self.dark_pixel_locations_x = []
        self.dark_pixel_locations_y = []
        for i in range(0,len(self.pixel_dark)):
            if self.pixel_dark[i]:
                print("test")
                self.dark_pixel_locations_x.append(self.pixel_locations_x[i])
                self.dark_pixel_locations_y.append(self.pixel_locations_y[i])
        self.led_locs_dark, =  self.ax_led_map.plot(self.dark_pixel_locations_x,self.dark_pixel_locations_y,
                                                    "h",markerfacecolor="None",mec = "k",label = "Dark")
        #currently selected pixel
        self.current_led_locs, = self.ax_led_map.plot(self.pixel_locations_x[0],self.pixel_locations_y[0],
                                                      "h",color = "k",label = "Current")
        self.ax_led_map_leg = self.ax_led_map.legend(ncol = 3,fontsize = 9)

        
        self.designed_v_measured, = self.ax_iq.plot(self.designed_freqs,self.measured_freqs/10**9,"."
                                                    ,label = "Assigned",zorder = 100)
        y_data_min, y_data_max = np.min(self.measured_freqs/10**9),np.max(self.measured_freqs/10**9)#self.ax_iq.get_ylim()
        self.expected_frequencies, = self.ax_iq.plot([self.pixel_freqs[0,self.combined_data_index],self.pixel_freqs[0,self.combined_data_index]],
                                                     [y_data_min,y_data_max],
                                                     label = "Expected",color = "C1",zorder = 0)
        self.expected_frequencies_2, = self.ax_iq.plot([self.pixel_freqs[1,self.combined_data_index],self.pixel_freqs[1,self.combined_data_index]],
                                                       [y_data_min,y_data_max],color = "C1",zorder = 0)

        self.if_assigned, = self.ax_iq.plot(self.pixel_freqs[:,self.combined_data_index],
                                                     [self.measured_freqs[self.plot_index]/10**9,self.measured_freqs[self.plot_index]/10**9],
                                                     "o",label = "If Assigned",markerfacecolor = "None",mec = "k",markersize = 6)
        

        self.ax_iq_leg = self.ax_iq.legend(ncol = 3,fontsize = 9)

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
                    self.combined_data_format.append(': {:g}')
            else:
                self.combined_data_format = self.combined_data_format
            self.res_indexes_original = np.arange(0, self.combined_data.shape[0])
            # run the initialization script
            self.combined_plot(ax_combined=self.ax_combined)

        self.led_heat_map = None
        self.led_heat_map_plot(self.ax_led_heat_map)
        self.led_heat_map_2 = None
        self.led_heat_map_2_plot(self.ax_led_heat_map_2)
        self.refresh_plot()
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
        self.led_heat_map_current_2nd, = self.ax_led_heat_map.plot(self.pixel_locations_x[self.combined_data_index],
                                                         self.pixel_locations_y[self.combined_data_index],'h',
                                                         fillstyle = 'none',mec = "orange",markersize = 10,
                                                               markeredgewidth = 1)
    def led_heat_map_2_plot(self,ax_led_heat_map_2):
        if self.led_heat_map_2 is not None:
            self.led_heat_map_2.remove()
            self.led_heat_map_2 = None
        gc.collect()
        color_array = self.scalarMap_2.to_rgba(self.combined_data_2[self.plot_index,:])
        #print(color_array)                                                                                                                                          
        self.led_heat_map_2 = ax_led_heat_map_2.scatter(x=self.pixel_locations_x, y=self.pixel_locations_y, s=50,
                                                        color=color_array, marker='h')
        self.led_heat_map_current_2, = self.ax_led_heat_map_2.plot(self.pixel_locations_x[self.combined_data_index],
                                                         self.pixel_locations_y[self.combined_data_index],'h',
                                                         fillstyle = 'none',mec = "black",markersize = 10,
                                                         markeredgewidth = 2)
        self.led_heat_map_current_2nd_2, = self.ax_led_heat_map_2.plot(self.pixel_locations_x[self.combined_data_index],
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
        ax_combined.set_title(str(len(np.where(self.assigned_group_index>-1)[0]))+"/"+str(self.chan_freqs.shape[1])+" Resonators assigned", color='black')
        ax_combined.set_ylabel("min index")
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
        label = str(self.plot_index)+self.combined_data_format[self.combined_data_index].format(highlighted_data_value)
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
        # we only rescale manually (intentionally) for this axis
        ax_combined.autoscale()

 
    def refresh_plot(self, autoscale=True):
        self.ax_led_map.set_title('LED Assignment Map, current pos '+self.combined_data_names[self.combined_data_index] +
                                  ", PX"+str(self.group_index[self.combined_data_index]+1))
        if self.assigned_pixel_index[self.plot_index] > 0:
            self.ax_mag.set_facecolor('lightgreen')
        else:
            self.ax_mag.set_facecolor("None")
        
        self.mag_line.set_data(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                          20 * np.log10(np.abs(self.z[:, self.plot_index, self.combined_data_index])))
        if self.z_unmasked is not None:
            self.mag_line_unmasked.set_data(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                          20 * np.log10(np.abs(self.z_unmasked[:, self.plot_index, self.combined_data_index])))
        self.baseline_mag.set_data(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                              20 * np.log10(np.abs(self.z_baseline[:, self.plot_index])))
        if self.z_baseline_unmasked is not None:
            self.baseline_mag_unmasked.set_data(self.chan_freqs[:, self.plot_index] / 10 ** 6,
                                              20 * np.log10(np.abs(self.z_baseline_unmasked[:, self.plot_index])))

        self.ax_mag.relim()
        self.ax_mag.autoscale()
        center_freq_MHz = self.chan_freqs[self.chan_freqs.shape[0] // 2, self.plot_index] / 10 ** 6
        self.ax_mag.set_title(f'{"%3.3f" % center_freq_MHz} MHz - Resonator Index: {self.plot_index:03}')

        # led assignment map
        self.current_led_locs.set_data(self.pixel_locations_x[self.combined_data_index],
                                       self.pixel_locations_y[self.combined_data_index])

        self.refresh_assigned_values_for_plotting()
      
        self.led_locs_assigned.set_data(self.assigned_pixel_locations_x,self.assigned_pixel_locations_y)
        self.led_locs_assigned_1.set_data(self.assigned_pixel_locations_x_1,self.assigned_pixel_locations_y_1)
        self.led_locs_assigned_2.set_data(self.assigned_pixel_locations_x_2,self.assigned_pixel_locations_y_2)

        # led heat maps
        self.led_heat_map_current.set_data(self.pixel_locations_x[self.combined_data_index],
                                           self.pixel_locations_y[self.combined_data_index])
        self.led_heat_map_current_2nd.set_data(self.pixel_locations_x[self.combined_data_index],
                                           self.pixel_locations_y[self.combined_data_index])
        self.led_heat_map_current_2.set_data(self.pixel_locations_x[self.combined_data_index],
                                           self.pixel_locations_y[self.combined_data_index])
        self.led_heat_map_current_2nd_2.set_data(self.pixel_locations_x[self.combined_data_index],
                                           self.pixel_locations_y[self.combined_data_index])
        
      
        self.designed_v_measured.set_data(self.designed_freqs,self.measured_freqs/10**9)
        self.if_assigned.set_data(self.pixel_freqs[:,self.combined_data_index],
                                                     [self.measured_freqs[self.plot_index]/10**9,
                                                      self.measured_freqs[self.plot_index]/10**9])
        
        self.ax_iq.relim()
        self.ax_iq.autoscale()

        y_data_min, y_data_max = np.min(self.measured_freqs/10**9),np.max(self.measured_freqs/10**9)#self.ax_iq.get_ylim()  
        self.expected_frequencies.set_data([self.pixel_freqs[0,self.combined_data_index],self.pixel_freqs[0,self.combined_data_index]],
                                                     [y_data_min,y_data_max])
        self.expected_frequencies_2.set_data([self.pixel_freqs[1,self.combined_data_index],self.pixel_freqs[1,self.combined_data_index]],
                                                     [y_data_min,y_data_max])
        
        # reset the led_heat_map_plot
        self.norm = mpl.colors.Normalize(vmin=np.min(self.combined_data[self.plot_index,:]),
                                         vmax=np.max(self.combined_data[self.plot_index,:]))
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.norm_2 = mpl.colors.Normalize(vmin=np.min(self.combined_data_2[self.plot_index,:]),
                                         vmax=np.max(self.combined_data_2[self.plot_index,:]))
        self.scalarMap_2 = cm.ScalarMappable(norm=self.norm_2, cmap=self.cmap)
        #print(self.combined_data[self.plot_index,:]/self.chan_freqs.shape[1])
        color_array = self.scalarMap.to_rgba(self.combined_data[self.plot_index,:])

        self.led_heat_map.set_color(color_array)

        color_array = self.scalarMap_2.to_rgba(self.combined_data_2[self.plot_index,:])

        self.led_heat_map_2.set_color(color_array)
                
        if self.combined_data is not None:
            data_type = self.combined_data_names[self.combined_data_index]
            self.ax_combined.set_title(str(len(np.where(self.assigned_group_index>-1)[0]))+"/"+str(self.chan_freqs.shape[1])+" Resonators assigned", color='black')
            self.ax_combined.set_ylabel("min index")
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
            label = str(self.plot_index)+self.combined_data_format[self.combined_data_index].format(highlighted_value)
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


    def refresh_assigned_values_for_plotting(self):
        self.assigned_1 = np.zeros(self.combined_data.shape[1])
        self.assigned_2 = np.zeros(self.combined_data.shape[1])
        self.assigned = np.zeros(self.combined_data.shape[1])
        if self.pixel_freqs is not None and self.assigned_group_index.any()>-1:
            for i in range(0,self.chan_freqs.shape[1]):
                if self.assigned_group_index[i] >-1:
                    temp_index = np.where(np.logical_and(self.group_index == self.assigned_group_index[i],
                                                                       self.pixel_index == self.assigned_pixel_index[i]))[0][0]
                    if self.assigned_res_index[i] == 0:
                        self.assigned_1[temp_index] = 1
                    elif self.assigned_res_index[i] == 1:
                        self.assigned_2[temp_index] = 1
                    self.designed_freqs[i] = self.pixel_freqs[self.assigned_res_index[i],temp_index]

        for i in range(0,len(self.assigned)):
            if self.assigned_1[i] and self.assigned_2[i]:
                self.assigned[i] = 1

        self.assigned_pixel_locations_x = []
        self.assigned_pixel_locations_y = []
        self.assigned_pixel_locations_x_1 = []
        self.assigned_pixel_locations_y_1 = []
        self.assigned_pixel_locations_x_2 = []
        self.assigned_pixel_locations_y_2 = []
        for i in range(0,len(self.assigned)):
            if self.assigned_1[i]:
                self.assigned_pixel_locations_x_1.append(self.pixel_locations_x[i])
                self.assigned_pixel_locations_y_1.append(self.pixel_locations_y[i])
            if self.assigned_2[i]:
                self.assigned_pixel_locations_x_2.append(self.pixel_locations_x[i])
                self.assigned_pixel_locations_y_2.append(self.pixel_locations_y[i])
            if self.assigned[i]:
                self.assigned_pixel_locations_x.append(self.pixel_locations_x[i])
                self.assigned_pixel_locations_y.append(self.pixel_locations_y[i])

        
    def instructions(self):
        instructions = [("left-right", "change resonator", 'green')]
  
        if self.combined_data is not None:
            flag_type = self.flags_types[self.flag_type_index]
            instructions.extend([('up-down', 'cycle through leds', 'yellow'),
                                 ('double-click', 'go to the resonator/\nled position', 'black'),
                                 ('spacebar', 'assign current resonator', 'red'),
                                 ('D-key', "delete current resonator's\n assignments", 'white'),
                                 ('A-key', 'run auto assignment', 'purple'),
                                 ('W-key', 'Make PDF', 'cyan')])

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


    def on_key_press(self, event):
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
            if self.combined_data_index == self.combined_data.shape[1] - 1:
                self.combined_data_index = 0
            else:
                self.combined_data_index = self.combined_data_index + 1
            self.refresh_plot()

        elif event.key == 'down':
            if self.combined_data_index == 0:
                self.combined_data_index = self.combined_data.shape[1] - 1
            else:
                self.combined_data_index = self.combined_data_index - 1
            self.refresh_plot()
        # Writing an output file
        elif event.key == 'w':
            if self.verbose:
                print("saving to pdf")
            pop_up = PopUpDataEntry("Enter filename","network_")
            filename = pop_up.value
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
            
            self.check_for_conflicts(self.plot_index)
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
                    #self.assigned[self.combined_data_index] = 1
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
                    self.check_for_conflicts(self.plot_index)
                else:
                    print("skipping resonator ",i)
            self.refresh_plot()

        # Flagging and removing interactions
        elif event.key == 'f':
            current_flags = self.flags[self.plot_index]
            if current_flags:
                print(f"Res Index {self.plot_index} current flags: {current_flags}")
                
                

   
    def onClick(self, event):
        if self.combined_data is not None:
            if event.dblclick:
                # get the radius of each point from the click
                if event.inaxes==self.ax_led_map or event.inaxes==self.ax_led_heat_map or event.inaxes==self.ax_led_heat_map_2:
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
            

    def make_pdf(self, filename):
        pdf_pages = PdfPages(filename)
        for i in tqdm.tqdm(range(0, self.chan_freqs.shape[1]), ascii=True):
            self.plot_index = i
            if self.assigned_group_index[self.plot_index] > -1:
                self.combined_data_index = np.where(np.logical_and(self.group_index == self.assigned_group_index[self.plot_index],
                                                                       self.pixel_index == self.assigned_pixel_index[self.plot_index]))[0][0]
            else:
                self.combined_data_index = 0
            self.refresh_plot()
            pdf_pages.savefig(self.fig)
        pdf_pages.close()

    def check_for_conflicts(self,index):#assums index has been assigned
        group_index = self.assigned_group_index[index]
        res_index = self.assigned_res_index[index]
        pixel_index = self.assigned_pixel_index[index]
        for i in range(0,self.chan_freqs.shape[1]):
            if i != index:
                if group_index == self.assigned_group_index[i] and res_index == self.assigned_res_index[i] and pixel_index == self.assigned_pixel_index[i]:
                    print("Resonator " +str(i) +" was already assigned these parameters")
                    pop_up = PopUpDataEntry("Resonator " +str(i) +" was already\nassigned these parameters","")
                    self.assigned_group_index[index] = -1
                    self.assigned_res_index[index] = -1
                    self.assigned_pixel_index[index] = -1


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

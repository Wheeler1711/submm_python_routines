import numpy as np
import matplotlib.pyplot as plt


class Select_ZPD(object):
    """
    Convention is to supply the data in magnitude units i.e. 20*np.log10(np.abs(z))
    frequencies should be supplied in Hz
    """

    def __init__(self, y, multiple = False):
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
        x_space = 0.05
        self.x_width_over_y_height = 1.0

        self.y = y
        self.multiple = multiple

        # main figure
        self.fig = plt.figure(figsize=(16, 6))
        x_width_plot = x_width - key_x_width - x_space
        figure_coords = [left, bottom, x_width_plot, y_height]
        self.ax = self.fig.add_axes(figure_coords, frameon=False, autoscale_on=True)

        # instruction figure
        key_y_height = 0.8#y_height
        key_figure_coords = [left + x_width_plot, top-key_y_height, key_x_width, key_y_height]
        self.ax_key = self.fig.add_axes(key_figure_coords, frameon=False)
        self.ax_key.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        self.ax_key.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # connect to interactive stuff
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        #self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        
        self.l1, = self.ax.plot(y)
        self.ZPD1 = None
        self.start = None
        self.end = None
        self.ZPD2 = None

        #self.p1, = self.ax.plot(self.chan_freqs[self.kid_idx] / 10 ** 9, self.data[self.kid_idx], "r*", markersize=8)


        self.ax.set_xlabel('data index')
        self.ax.set_ylabel('y')
        self.plot_instructions()
        self.refresh_plot()
        plt.show(block=True)

    def instructions(self):
        instructions = [("H+key", "home", 'orange'),
                            ("O+key", "rectangle zoom", 'red'),
                            ("P+key", "pan", 'green'),
                            ("D+key", "start over", 'yellow'),
                        ("double-click", "select point", 'cyan')]
        return instructions


    def plot_instructions(self):
        instructions = self.instructions()
        self.ax_key.clear()
        steps_per_item = 1.5
        y_step = 0.9 / (steps_per_item * float(len(instructions)))
        y_now = 0.9
        for key_press, description, color in instructions:
            self.ax_key.text(0.4, y_now, key_press.center(17), color='k',
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

        if event.key == 'd':  # pan left
            print("starting over")
            if self.ZPD1 is not None:
                self.ZPD1.remove()
                self.ZPD1 = None
            if self.start is not None:
                self.start.remove()
                self.start = None
            if self.end is not None:
                self.end.remove()
                self.end = None
            if self.ZPD2 is not None:
                self.ZPD2.remove()
                self.ZPD2 = None
            if self.ax.legend is not None:
                self.ax.get_legend().remove()
            self.refresh_plot()
            plt.draw()

    def onClick(self, event):
        if event.dblclick:
            x_data_coords = np.arange(0,len(self.y)) - event.xdata
            x_data_min, x_data_max = self.ax.get_xlim()
            x_data_range = x_data_max - x_data_min
            x_norm_coords = x_data_coords / x_data_range
            x_yratio_coords = x_norm_coords * self.x_width_over_y_height
            y_data_coords = self.y - event.ydata
            y_data_min, y_data_max = self.ax.get_ylim()
            y_data_range = y_data_max - y_data_min
            y_norm_coords = y_data_coords / y_data_range
            radius_array = np.sqrt(x_yratio_coords ** 2 + y_norm_coords ** 2)
            index = np.argmin(radius_array)
            if self.ZPD1 is None:
                self.ZPD1, = self.ax.plot(index,self.y[index],'o',mec = 'k',label = "ZPD")
                self.ZPD1_index = index
            elif self.start is None:
                self.start, = self.ax.plot(index,self.y[index],'o',mec = 'k',label = "scan start")
                self.start_index = index
            elif self.end is None:
                self.end, = self.ax.plot(index,self.y[index],'o',mec = 'k',label = "scan end")
                self.end_index = index
            elif self.multiple and self.ZPD2 is None:
                self.ZPD2, = self.ax.plot(index,self.y[index],'o',mec = 'k',label = "ZPD 2")
                self.ZPD2_index = index
            

            self.refresh_plot()


    def refresh_plot(self):
        if self.ZPD1 is None:
            self.ax.set_title('Double click white light fringe')
        elif self.start is None:
            self.ax.set_title('Double click beginning of the scan')
        elif self.end is None:
            self.ax.set_title('Double click end of the scan')
        elif self.multiple and self.ZPD2 is None:
            self.ax.set_title('Double click end second white light fringe')
            
        self.ax.legend(loc=1)
        plt.draw()


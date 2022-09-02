import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# written on 10/16/2017 added a function for making prettier plots for presentations
# 4/20/2019 added in a function to add in more interactions for matplotlib plots


def presentation_plot():
    # function for setting matplotlib to makes plots that are actually visible for presentation and papers
    
    #set font size for plotting
    matplotlib.rcParams.update({'font.size': 16})
    #set axes linewidth for plotting
    matplotlib.rcParams['axes.linewidth'] = 2
    # set tick width
    matplotlib.rcParams['xtick.major.size'] = 10
    matplotlib.rcParams['xtick.direction'] = "in"
    matplotlib.rcParams['ytick.direction'] = "in"
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 5
    matplotlib.rcParams['xtick.minor.width'] = 1
    matplotlib.rcParams['ytick.major.size'] = 10
    matplotlib.rcParams['ytick.major.width'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 5
    matplotlib.rcParams['ytick.minor.width'] = 1



# function to add some functionality to matplotlib interactive plotting
# adds in the ability to pan around with the arrow keys
# adds in the ability to zoom in and out using the keyboard keys z and x
# adds in the ability to grab cursur info from screen
# adds in the ability to find mins and maximums 
def activate_interactive_plotting(fig,ax,zoom_factor = 0.1,lim_shift_factor = 0.1,show_pts = True):   
    class InteractivePlot(object): #must pass fig to this command
    
        def __init__(self,fig,ax,zoom_factor,lim_shift_factor,show_pts):
            #matplotlib.use("TkAgg")
            #print("hello world")
            self.fig = fig
            self.ax = ax
            self.show_pts = show_pts
            self.x_data = np.asarray(())
            self.y_data = np.asarray(())
            self.pts, = self.ax.plot(self.x_data,self.y_data,"+")
            self.text_dict = {}
            plt.rcParams['keymap.forward'] = ['v']
            plt.rcParams['keymap.back'] = ['c','backspace']# remove arrows from back and forward on plot
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
            self.fig.canvas.mpl_connect('button_press_event', self.onClick)
            self.shift_is_held = False
            self.control_is_held = False
            self.alt_is_held = False
            self.lim_shift_factor = lim_shift_factor
            self.zoom_factor = zoom_factor #no greater than 0.5
            print("")
            print("Interactive Plotting Tools Activated")
            print("Use arrow keys to pan.")
            print("Use z and x keys to zoom and Xplode.")
            print("Hold shift and right click to get cursur location data printed to screen")
            print("   and stored to x_data, y_data arrays.")
            print("Hold control and right click to remove a grabed data point from the plot")
            print("   and remove it from x_data, y_data arrays.")
            print("Hold alt and right click to get the nearest data point for plotted data")
            print("   and store it to x_data, y_data arrays")
            print("Press m to find the minimum point currently displayed")
            print("Press n to find the maximum point currently displayed")
            print("On mac shift and control key requires matplotlib backend TkAgg.")
            print("")

            #plt.show(block = True)

        def on_key_press(self, event):
            #print( event.key)
            if event.key == 'shift':
                self.shift_is_held = True
            if event.key == 'control':
                self.control_is_held = True
            if event.key == 'alt':
                self.alt_is_held = True
               
            if event.key == 'right': #pan right
                xlim_left, xlim_right = self.ax.get_xlim() 
                xlim_size = xlim_right-xlim_left
                self.ax.set_xlim(xlim_left+self.lim_shift_factor*xlim_size,xlim_right+self.lim_shift_factor*xlim_size)
                self.fig.canvas.draw()

            if event.key == 'left': #pan left
                xlim_left, xlim_right = self.ax.get_xlim() 
                xlim_size = xlim_right-xlim_left
                self.ax.set_xlim(xlim_left-self.lim_shift_factor*xlim_size,xlim_right-self.lim_shift_factor*xlim_size)
                self.fig.canvas.draw()

            if event.key == 'up': #pan up
                ylim_left, ylim_right = self.ax.get_ylim() 
                ylim_size = ylim_right-ylim_left
                self.ax.set_ylim(ylim_left+self.lim_shift_factor*ylim_size,ylim_right+self.lim_shift_factor*ylim_size)
                self.fig.canvas.draw()

            if event.key == 'down': #pan down
                ylim_left, ylim_right = self.ax.get_ylim() 
                ylim_size = ylim_right-ylim_left
                self.ax.set_ylim(ylim_left-self.lim_shift_factor*ylim_size,ylim_right-self.lim_shift_factor*ylim_size)
                self.fig.canvas.draw()

            if event.key == 'z': #zoom in
                xlim_left, xlim_right = self.ax.get_xlim() 
                ylim_left, ylim_right = self.ax.get_ylim() 
                xlim_size = xlim_right-xlim_left
                ylim_size = ylim_right-ylim_left
                self.ax.set_xlim(xlim_left+self.zoom_factor*xlim_size,xlim_right-self.zoom_factor*xlim_size)
                self.ax.set_ylim(ylim_left+self.zoom_factor*ylim_size,ylim_right-self.zoom_factor*ylim_size)
                self.fig.canvas.draw()
            elif event.key == 'Z':
                print("Is caps lock on?")

            if event.key == 'x': #zoom out
                xlim_left, xlim_right = self.ax.get_xlim() 
                ylim_left, ylim_right = self.ax.get_ylim() 
                xlim_size = xlim_right-xlim_left
                ylim_size = ylim_right-ylim_left
                self.ax.set_xlim(xlim_left-self.zoom_factor*xlim_size,xlim_right+self.zoom_factor*xlim_size)
                self.ax.set_ylim(ylim_left-self.zoom_factor*ylim_size,ylim_right+self.zoom_factor*ylim_size)
            elif event.key == 'X':
                print("Is caps lock on?")

            if event.key == 'm':
                self.find_min()
            elif event.key == 'M':
                print("Is caps lock on?")

            if event.key == 'n':
                self.find_max()
            elif event.key == 'N':
                print("Is caps lock on?")

        def on_key_release(self, event):
            if event.key == 'shift':
                self.shift_is_held = False
            if event.key == 'control':
                self.control_is_held = False
            if event.key == 'alt':
                self.alt_is_held = False

        def onClick(self, event):
            if event.button == 3:
                if self.alt_is_held: # add point on line
                    xlim_left, xlim_right = self.ax.get_xlim() # only look in current axis
                    index_low = np.argmin(np.abs(self.ax.lines[0].get_xdata()-xlim_left))
                    index_high  = np.argmin(np.abs(self.ax.lines[0].get_xdata()-xlim_right))+1
                    index = index_low + np.argmin(np.abs(self.ax.lines[0].get_xdata()[index_low:index_high]-event.xdata))
                    self.x_data = np.append(self.x_data,self.ax.lines[0].get_xdata()[index])
                    self.y_data = np.append(self.y_data,self.ax.lines[0].get_ydata()[index])
                    if self.show_pts == True:
                        self.refresh_plot()
                if self.shift_is_held: # add point
                    print("x, y = ", event.xdata,event.ydata)
                    self.x_data = np.append(self.x_data,event.xdata)
                    self.y_data = np.append(self.y_data,event.ydata)
                    if self.show_pts == True:
                        self.refresh_plot()
                if self.control_is_held: # delete point
                    if len(self.x_data)>0:
                        print("removing point")
                        delete_index = np.argmin(np.abs(self.x_data-event.xdata))
                        self.x_data = np.delete(self.x_data,delete_index)
                        self.y_data = np.delete(self.y_data,delete_index)
                        self.refresh_plot()



        def refresh_plot(self):
            self.pts.set_data(self.x_data,self.y_data)
            if len(self.text_dict)>0:
                for i in range(0,len(self.text_dict)):
                    self.text_dict[i].set_text("")# clear all of the texts
            self.text_dict = {} # empty the dictionary
            for i in range(0,len(self.x_data)): #rebuild the dictionary
                self.text_dict[i] = self.ax.text(self.x_data[i], self.y_data[i], str(i))
            self.fig.canvas.draw()

        def find_min(self):
            print("Finding minimum")
            xlim_left, xlim_right = self.ax.get_xlim() # only look in current axis
            index_low = np.argmin(np.abs(self.ax.lines[0].get_xdata()-xlim_left))
            index_high  = np.argmin(np.abs(self.ax.lines[0].get_xdata()-xlim_right))+1
            index = index_low + np.argmin(self.ax.lines[0].get_ydata()[index_low:index_high])
            self.x_data = np.append(self.x_data,self.ax.lines[0].get_xdata()[index])
            self.y_data = np.append(self.y_data,self.ax.lines[0].get_ydata()[index])
            self.refresh_plot()

        def find_max(self):
            print("Finding maximum")
            xlim_left, xlim_right = self.ax.get_xlim() # only look in current axis
            index_low = np.argmin(np.abs(self.ax.lines[0].get_xdata()-xlim_left))
            index_high  = np.argmin(np.abs(self.ax.lines[0].get_xdata()-xlim_right))+1
            index = index_low+np.argmax(self.ax.lines[0].get_ydata()[index_low:index_high])
            self.x_data = np.append(self.x_data,self.ax.lines[0].get_xdata()[index])
            self.y_data = np.append(self.y_data,self.ax.lines[0].get_ydata()[index])
            self.refresh_plot()

    
            

    ip = InteractivePlot(fig,ax,zoom_factor = zoom_factor,lim_shift_factor = lim_shift_factor,show_pts = show_pts)
    return ip



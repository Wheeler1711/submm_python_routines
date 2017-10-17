import matplotlib.pyplot as plt
import matplotlib

# written on 10/16/2017 added a function for making prettier plots for presentations


def presentation_plot():
    # function for setting matplotlib to makes plots that are actually visible for presentation and papers
    
    #set font size for plotting
    matplotlib.rcParams.update({'font.size': 16})
    #set axes linewidth for plotting
    matplotlib.rcParams['axes.linewidth'] = 2
    # set tick width
    matplotlib.rcParams['xtick.major.size'] = 10
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 5
    matplotlib.rcParams['xtick.minor.width'] = 1
    matplotlib.rcParams['ytick.major.size'] = 10
    matplotlib.rcParams['ytick.major.width'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 5
    matplotlib.rcParams['ytick.minor.width'] = 1

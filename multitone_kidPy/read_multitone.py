import numpy as np
import os

def openStoredSweep(savepath):
    """Opens sweep data
       inputs:
           char savepath: The absolute path where sweep data is saved
       ouputs:
           numpy array Is: The I values
           numpy array Qs: The Q values"""
    files = sorted(os.listdir(savepath))
    I_list, Q_list = [], []
    for filename in files:
        if filename.startswith('I'):
            I_list.append(os.path.join(savepath, filename))
        if filename.startswith('Q'):
            Q_list.append(os.path.join(savepath, filename))
    Is = np.array([np.load(filename) for filename in I_list])
    Qs = np.array([np.load(filename) for filename in Q_list])
    return Is, Qs


# reads in an iq sweep and stors i and q and the frequencies in a dictionary
def read_iq_sweep(filename):
	I, Q = openStoredSweep(filename)
	sweep_freqs = np.load(filename + '/sweep_freqs.npy')
	bb_freqs = np.load(filename + '/bb_freqs.npy')
	channels = len(bb_freqs)
	mags = np.zeros((channels, len(sweep_freqs))) 
	chan_freqs = np.zeros((len(sweep_freqs),channels))
	for chan in range(channels):
        	chan_freqs[:,chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
	dict = {'I': I, 'Q': Q, 'freqs': chan_freqs}
	return dict
	

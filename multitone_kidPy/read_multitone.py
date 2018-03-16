import numpy as np
from kidPy import openStoredSweep


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
	

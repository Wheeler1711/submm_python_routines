import numpy as np
import os
import pygetdata as gd
import struct
import matplotlib.pyplot as plt

def openStoredSweep(savepath,load_std = False):
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
    if len(stdI_list) >0:
            std_Is = np.array([np.load(filename) for filename in stdI_list])
            std_Qs = np.array([np.load(filename) for filename in stdQ_list])
    if load_std:
        return Is, Qs, std_Is, std_Qs
    else:
        return Is, Qs


# reads in an iq sweep and stors i and q and the frequencies in a dictionary
def read_iq_sweep(filename,load_std = False):
    if load_std:
        I, Q, I_std, Q_std = openStoredSweep(filename,load_std = True)
    else:
        I, Q, = openStoredSweep(filename)
    sweep_freqs = np.load(filename + '/sweep_freqs.npy')
    bb_freqs = np.load(filename + '/bb_freqs.npy')
    channels = len(bb_freqs)
    mags = np.zeros((channels, len(sweep_freqs))) 
    chan_freqs = np.zeros((len(sweep_freqs),channels))
    for chan in range(channels):
        chan_freqs[:,chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    if load_std:
        dict = {'I': I, 'Q': Q, 'freqs': chan_freqs,'I_std':I_std,'Q_std':Q_std}
    else:
        dict = {'I': I, 'Q': Q, 'freqs': chan_freqs}
    return dict

def read_stream(filename):
    firstframe = 0
    firstsample = 0
    d = gd.dirfile(filename, gd.RDWR|gd.UNENCODED)
    #print "Number of frames in dirfile =", d.nframes
    nframes = d.nframes
    
    vectors = d.field_list()
    ifiles = [i for i in vectors if i[0] == "I"]
    qfiles = [q for q in vectors if q[0] == "Q"]
    ifiles.remove("INDEX")
    ivals = d.getdata(ifiles[0], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
    qvals = d.getdata(qfiles[0], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
    ivals = ivals[~np.isnan(ivals)]
    qvals = qvals[~np.isnan(qvals)]
    i_stream = np.zeros((len(ivals),len(ifiles)))
    q_stream = np.zeros((len(qvals),len(qfiles)))
    
    for n in range(len(ifiles)):
        ivals = d.getdata(ifiles[n], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
        qvals = d.getdata(qfiles[n], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
        i_stream[:,n] = ivals[~np.isnan(ivals)]
        q_stream[:,n] = qvals[~np.isnan(qvals)]
    d.close()
    #read in the time file
    with open(filename+"/time",'rb') as content_file:
        content = content_file.read()
    time_val = []
    for i in range(0,len(content)/8):
        time_val.append(struct.unpack('d',content[0+8*i:8+8*i])[0])


    #read in the time file
    with open(filename+"/packet_count",'rb') as content_file:
        content = content_file.read()
    packet_val = []
    for i in range(0,len(content)/8):
        packet_val.append(struct.unpack('L',content[0+8*i:8+8*i])[0])
    packet = np.asarray(packet_val)
    if ((packet -np.roll(packet,1))[1:]!=1).any():#you dropped packet
        print("!!!!WARNING!!!!! you dropped some packets during your measurement consider increasing your system buffer size")
        plt.figure(1)
        plt.title("Delta t between packets")
        plt.plot((time_val-np.roll(time_val,1))[1:])
        plt.figure(2)
        plt.title("Delta packet")
        plt.plot((packet-np.roll(packet,1))[1:])
        plt.show()
    
    dictionary = {'I_stream':i_stream,'Q_stream':q_stream,'time':time_val,'packet_count':packet_val}
    return dictionary
	

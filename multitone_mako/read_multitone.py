import numpy as np
try:
    from astropy.io import fits
except:
    print("This function requires astropy to run") 

# this is code to read in the mako multitone data for fits file and store it in something useful for python


# Please not the date and details of any changes
# writen by Jordan Wheeler 8/25/2016
#Change log
# 3/17/2017 -Jordan added functionality to read stream fits files
# and added scaling the data by the scaling factor and averages



# read in calibration file i.e mako20160818_000147Cal.fits
def readcal(infile,hdu = False):
    #keyword blind
    # it would probably be good to just seperate the blind bins from the non blind bins
    # because I rarely actually care about the blind bins and it could simplfy my overarching codes
    
    hdulist = fits.open(infile)
    freqs = hdulist[1].data['freqs'][::1] # ther first data point is useless and dumb
    bins = hdulist[1].data['bins'][1::] #first data point is useless and dumb
    blind_bins = hdulist[1].data['Blindbin'][0:-1] #the last data point is useless and dumb
    I = hdulist[2].data['I']*hdulist[2].header['scaling']/hdulist[2].header['averages']
    Q = hdulist[2].data['Q']*hdulist[2].header['scaling']/hdulist[2].header['averages']
    I_Q_freqs = hdulist[2].header['bininHz']*hdulist[2].data['tones']

    # we want to know which bins are blind bins so that we can appropriatly ignore them
    # nested for loop is bad (slow in python) see if we can do better when bored
    isblind = np.ones(bins.shape[0])
    for k in range(0,blind_bins.size):
        for j in range(0,blind_bins.size):
            if bins[k] == blind_bins[j]:
                isblind[k] = -1

    #lets seperate the blind bins from the real bins
    blind_index = np.where(isblind<0)[0]
    non_blind_index = np.where(isblind>0)[0]

    I_blind = I[:,blind_index]
    Q_blind = Q[:,blind_index]
    freqs_blind = freqs[blind_index]
    I_Q_freqs_blind = I_Q_freqs[:,blind_index]

    I_res = I[:,non_blind_index]
    Q_res = Q[:,non_blind_index]
    freqs_res = freqs[non_blind_index]
    I_Q_freqs_res = I_Q_freqs[:,non_blind_index]

    #make a dicitonary for it
    cal_dict = {'I': I_res, 'Q': Q_res, 'freqs': I_Q_freqs_res, 'center_freqs': freqs_res,'I_blind':I_blind,'Q_blind':Q_blind,'freqs_blind':I_Q_freqs_blind,'center_freqs_blind':freqs_blind}
    
    hdulist.close()
    
    # we want to reformat some of the data so that it is more accessable
    if hdu == False:
        return cal_dict
    else:
         return hdulist  


def readstream(infile,hdu = False):

    hdulist = fits.open(infile)
    
    cal_file = hdulist[0].header['calfile']
    hdulist_cal = fits.open(cal_file)
    scaling = hdulist_cal[2].header['scaling']
    averages = hdulist_cal[2].header['averages']
    freqs = hdulist_cal[1].data['freqs'][::1] # ther first data point is useless and dumb
    bins = hdulist_cal[1].data['bins'][1::] #first data point is useless and dumb
    blind_bins = hdulist_cal[1].data['Blindbin'][0:-1] #the last data point is useless and dumb
    hdulist_cal.close()

    # we want to know which bins are blind bins so that we can appropriatly ignore them
    # nested for loop is bad (slow in python) see if we can do better when bored
    isblind = np.ones(bins.shape[0])
    for k in range(0,blind_bins.size):
        for j in range(0,blind_bins.size):
            if bins[k] == blind_bins[j]:
                isblind[k] = -1

    blind_index = np.where(isblind<0)[0]
    non_blind_index = np.where(isblind>0)[0]

    #we have to find out how many hdulist entrys there are
    i = -1
    test = False
    while test == False:
        try:
            i = i+1
            k = hdulist[i]
        except:
            test = True
    hdulist_length = i

    tones = hdulist[1].data['tones'][0,:]
    
    I = hdulist[1].data['I']*scaling/averages
    Q = hdulist[1].data['Q']*scaling/averages
    for i in range(2,hdulist_length):
        I = np.vstack((I,hdulist[i].data['I']*scaling/averages))
        Q = np.vstack((Q,hdulist[i].data['Q']*scaling/averages))

    I_blind = I[:,blind_index]
    Q_blind = Q[:,blind_index]

    I_res = I[:,non_blind_index]
    Q_res = Q[:,non_blind_index]

    stream_dict = {'I': I_res, 'Q': Q_res, 'I_blind':I_blind,'Q_blind':Q_blind}

    hdulist.close()

    # we want to reformat some of the data so that it is more accessable
    if hdu == False:
        return stream_dict
    else:
         return hdulist 


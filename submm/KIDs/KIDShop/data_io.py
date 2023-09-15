import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import glob
import pickle
import os
from submm.KIDs import analyze_single_tone as ast

def convert_noise_mat_to_dict(filename,verbose = False):
    '''
    function for taking the noise files saved by KIDShop and turning
    it into a python dictionary
    everything as numpy arrays
    Not sure if it works for a single channel yet
    '''
    
    data = loadmat(filename)
    dp = data['dp']
    traj =  dp['traj'][0][0][0]
    if 'noise' in dp.dtype.fields:
        noise = dp['noise'][0][0][0]
    else:
        noise = None
    if 'noised' in dp.dtype.fields:
        noised = dp['noised'][0][0][0]
    else:
        noised = None

    if verbose:
        print("---------\ntraj\n---------")
        for field in traj.dtype.fields:
            #traj_dict[field] = traj[field]
            print(field)
        if noise is not None:
            print("---------\nnoise\n---------")
            for field in noise.dtype.fields:
                #noise_dict[field] = noise[field]
                print(field)
        else:
            print("no noise in data product")
                
        if noised is not None:
            print("---------\nnoised\n---------")
            for field in noised.dtype.fields:
                #noised_dict[field] = noised[field]
                print(field)
        else:
            print("no noised in data product")

            
        

    n_res = len(traj['tau'])
    
    #######################################
    #  traj
    #######################################
    traj_dict = {}
    
    tau = np.asarray(())
    for i in range(0,n_res):
        tau = np.append(tau,traj['tau'][i][0][0])
    traj_dict['tau'] = tau

    farray = np.zeros((traj['farray'][0].shape[0],n_res))
    for i in range(0,n_res):
        farray[:,i] = traj['farray'][i][:,0]
    traj_dict['farray'] = farray

    zarray = np.zeros((traj['zarray'][0].shape[0],n_res),dtype = np.complex128)
    for i in range(0,n_res):
        zarray[:,i] = traj['zarray'][i][:,0]
    traj_dict['zarray'] = zarray

    dzdf = np.zeros((traj['dzdf'][0].shape[0],n_res),dtype = np.complex128)
    for i in range(0,n_res):
        dzdf[:,i] = traj['dzdf'][i][:,0]
    traj_dict['dzdf'] = dzdf

    fnid = np.asarray((),dtype = np.uint8)
    for i in range(0,n_res):
        fnid = np.append(fnid,traj['fnid'][i][0][0])
    traj_dict['fnid'] = fnid

    #######################################
    #  noise
    #######################################
    noise_dict = {}

    if noise is not None:
    
        fn = np.asarray(())
        for i in range(0,n_res):
            fn = np.append(fn,noise['fn'][i][0][0])
        noise_dict['fn'] = fn

        fs = np.asarray((),dtype = np.int32)
        for i in range(0,n_res):
            fs = np.append(fs,noise['fs'][i][0][0])
        noise_dict['fs'] = fs

        adtime = np.asarray((),dtype = np.uint8)
        for i in range(0,n_res):
            adtime = np.append(adtime,noise['adtime'][i][0][0])
        noise_dict['adtime'] = adtime

        nshot = np.asarray((),dtype = np.uint8)
        for i in range(0,n_res):
            nshot = np.append(nshot,noise['nshot'][i][0][0])
        noise_dict['nshot'] = nshot

        r = np.asarray((),dtype = np.uint8)
        for i in range(0,n_res):
            r = np.append(r,noise['r'][i][0][0])
        noise_dict['r'] = r

        ch = np.zeros((noise['ch'][0][0].shape[0],n_res),dtype = np.uint8)
        for i in range(0,n_res):
            ch[:,i] = noise['ch'][i][0]
        noise_dict['ch'] = ch

        vin = np.zeros((noise['vin'][0][0].shape[0],n_res))
        for i in range(0,n_res):
            vin[:,i] = noise['vin'][i][0]
        noise_dict['vin'] = vin

        tt = np.zeros((noise['tt'][0].shape[0],noise['tt'][0].shape[1],n_res))
        for i in range(0,n_res):
            tt[:,:,i] = noise['tt'][i][0]
        noise_dict['tt'] = tt

        zfn = np.asarray((),dtype = np.complex128)
        for i in range(0,n_res):
            zfn = np.append(zfn,noise['zfn'][i][0][0])
        noise_dict['zfn'] = zfn

        zn = np.zeros((noise['zn'][0].shape[0],n_res),dtype = np.complex128)
        for i in range(0,n_res):
            zn[:,i] = noise['zn'][i][:,0]+zfn[i]
        noise_dict['zn'] = zn

        #d0 = np.asarray((),dtype = np.uint8)
        #for i in range(0,n_res):
        #    d0 = np.append(d0,noise['d0'][i][0][0])
        #noise_dict['d0'] = d0

        #filename = np.asarray((),dtype = np.uint8)
        #for i in range(0,n_res):
        #    filename = np.append(filename,noise['filename'][i][0][0])
        #noise_dict['filename'] = filename

    #######################################
    #  noised
    #######################################
    noised_dict = {}
    if noised is not None:

        fn = np.asarray(())
        for i in range(0,n_res):
            fn = np.append(fn,noised['fn'][i][0][0])
        noised_dict['fn'] = fn

        fs = np.asarray((),dtype = np.int32)
        for i in range(0,n_res):
            fs = np.append(fs,noised['fs'][i][0][0])
        noised_dict['fs'] = fs

        adtime = np.asarray((),dtype = np.uint8)
        for i in range(0,n_res):
            adtime = np.append(adtime,noised['adtime'][i][0][0])
        noised_dict['adtime'] = adtime

        nshot = np.asarray((),dtype = np.uint8)
        for i in range(0,n_res):
            nshot = np.append(nshot,noised['nshot'][i][0][0])
        noised_dict['nshot'] = nshot

        r = np.asarray((),dtype = np.uint8)
        for i in range(0,n_res):
            r = np.append(r,noised['r'][i][0][0])
        noised_dict['r'] = r

        ch = np.zeros((noised['ch'][0][0].shape[0],n_res),dtype = np.uint8)
        for i in range(0,n_res):
            ch[:,i] = noised['ch'][i][0]
        noised_dict['ch'] = ch

        vin = np.zeros((noised['vin'][0][0].shape[0],n_res))
        for i in range(0,n_res):
            vin[:,i] = noised['vin'][i][0]
        noised_dict['vin'] = vin

        tt = np.zeros((noised['tt'][0].shape[0],noised['tt'][0].shape[1],n_res))
        for i in range(0,n_res):
            tt[:,:,i] = noised['tt'][i][0]
        noised_dict['tt'] = tt

        zfn = np.asarray((),dtype = np.complex128)
        for i in range(0,n_res):
            zfn = np.append(zfn,noised['zfn'][i][0][0])
        noised_dict['zfn'] = zfn

        zn = np.zeros((noised['zn'][0].shape[0],n_res),dtype = np.complex128)
        for i in range(0,n_res):
            zn[:,i] = noised['zn'][i][:,0] +zfn[i]
        noised_dict['zn'] = zn

    dp = {'traj': traj_dict, 'noise': noise_dict, 'noised': noised_dict}
    
    return dp

def save_dp(filename,dp):
    fileObj = open(filename, 'wb')
    pickle.dump(dp,fileObj)
    fileObj.close()

def load_dp(filename):
    fileObj = open(filename, 'rb')
    dp = pickle.load(fileObj)
    fileObj.close()
    return dp

def convert_all():
    '''
    convert all noise dp in directory to python pickled dictionaries
    '''
    filenames = glob.glob("Res*.mat")
    for filename in filenames:
        print(filename)
        filename_p = filename.removesuffix(".mat")+".p"
        if os.path.isfile("./"+filename_p):
            print("already converted")
        else:
            try:
                dp = convert_noise_mat_to_dict(filename)
                save_dp(filename_p,dp)
            except Exception as e:
                print("Could not convert ",filename)
                print(e)

def calibrate_all(tau = None):
    if not os.path.isdir("processed_python"):
        print("creating processed_python folder")
        os.makedirs("processed_python")
    filenames = glob.glob("Res*.p")
    for filename in filenames:
        filename_check = "processed_python/"+filename.removesuffix(".p")+"_"+str(0)+".p"
        if os.path.isfile(filename_check):
            print(filename)
            print("already calibrated")
        else:
            dp = load_dp(filename)
            if 'fn' in dp['noised'].keys():
                for i in range(0,len(dp['traj']['tau'])):
                    filename_save = "processed_python/"+filename.removesuffix(".p")+"_"+str(i)+".p"
                    if tau is None:
                        cal_dict = ast.calibrate_single_tone(dp['traj']['farray'][:,i]*10**9,dp['traj']['zarray'][:,i],
                                                             dp['noised']['fn'][i]*10**9,dp['noised']['zn'][:,i],
                                                                 tau = -dp['traj']['tau'][i]*10**-9,filename = filename_save)
                    else: # you forgot to enter tau into the gui
                        print("using specified tau")
                        cal_dict = ast.calibrate_single_tone(dp['traj']['farray'][:,i]*10**9,dp['traj']['zarray'][:,i],
                                                             dp['noised']['fn'][i]*10**9,dp['noised']['zn'][:,i],
                                                                 tau = tau,filename = filename_save)
            else:
                print("skiping")
                print(filename)
                print("no noised in data")
            
    
            
        
    
    

    
    

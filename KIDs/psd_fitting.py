import numpy as np
from scipy.stats import binned_statistic
import scipy.optimize as optimization
import matplotlib.pyplot as plt

#set of modules for fitting psd of kinetic inductance detectors
#Written by Jordan 1/5/2017

#To Do
#add verbose = true keyword

#Change Log
#1/9/2017 Added sigma_increase_cutoff and sigma_increase_factor to fit_psd modules


# noise profiles
def noise_profile(y,a,b,c):
    return a+b*y**-c
def noise_slope(y,b,c):
    return b*y**-c
def noise_white(y,a):
    return np.ones(y.shape[0])*a
def noise_profile_lor(y,a,b,c,d):
    return (a+b*y**-c)/(1+(2*np.pi*y*d)**2.)
def noise_profile_lor_vec(y,a,b,c,d):
    y = np.reshape(y,(y.shape[0],1,1,1,1))
    return (a+b*y**-c)/(1+(2*np.pi*y*d)**2.)

def std_of_mean(x):
    if np.size(x) == 1:
        return x
    else:
        return np.std(x)/np.sqrt(np.size(x))

    index_for_fitting = np.where(((x>=freq_range[0]) & (x <=freq_range[1])))
        
    if ('ranges' in keywords):
        ranges = keywords['ranges']
    else:
        if ('white_freq' in keywords):
            white_freq = keywords['white_freq']
            white_index = np.argmin(np.abs(x[index_for_fitting]-white_freq))
        else:
            white_index = len(y[index_for_fitting])//2 # noise is white in the middle of psd

        white_guess = y[index_for_fitting][white_index]
        x0_guess = np.array([ white_guess,
                                 (y[index_for_fitting][0]-white_guess)/x[index_for_fitting][0]**(-1.), # assume 1/f dominates at lowest frequency but still subtract off white noise
                                 1,# guess 1/f is index is 1
                                 1./2/np.pi/x[index_for_fitting][np.argmin(np.abs(white_guess/2.-y[index_for_fitting]))]]) # look for 3dB decrease from white noise guess
        print("guess values are")
        print(x0_guess)
        ranges = np.asarray(([x0_guess[0]/2,x0_guess[1]/20,0.5,x0_guess[3]/2],[ x0_guess[0]*2,x0_guess[1]*10,2,x0_guess[3]*2]))
                

    if error is None:
        error = np.ones(len(x[index_for_fitting]))

    a_values = np.linspace(ranges[0][0],ranges[1][0],n_grid_points)
    b_values = np.linspace(ranges[0][1],ranges[1][1],n_grid_points)
    c_values = np.linspace(ranges[0][2],ranges[1][2],n_grid_points)
    d_values = np.linspace(ranges[0][3],ranges[1][3],n_grid_points)
    evaluated_ranges = np.vstack((a_values,b_values,c_values,d_values))

    a,b,c,d = np.meshgrid(a_values,b_values,c_values,d_values,indexing = "ij") #always index ij

    evaluated = noise_profile_lor_vec(x[index_for_fitting],a,b,c,d)
    data_values = np.reshape(y[index_for_fitting],(y[index_for_fitting].shape[0],1,1,1,1))
    error = np.reshape(error,(y[index_for_fitting].shape[0],1,1,1,1))
    #print(evaluated.shape)
    #print(data_values.shape)
    #print(error.shape)
    sum_dev = np.sum(((evaluated-data_values)**2/error**2),axis = 0) # comparing in magnitude space rather than magnitude squared
    #print(sum_dev.shape)

    min_index = np.where(sum_dev == np.min(sum_dev))
    print("grid values at minimum are")
    print(min_index)
    index1 = min_index[0][0]
    index2 = min_index[1][0]
    index3 = min_index[2][0]
    index4 = min_index[3][0]
    fit_values = np.asarray((a_values[index1],b_values[index2],c_values[index3],d_values[index4]))
    fit_values_names = ('a (white)','b (1/f)','c (1/f exponent)','d (tau)')
    fit_result = noise_profile_lor(x,a_values[index1],b_values[index2],c_values[index3],d_values[index4])
    x0_guess_result = noise_profile_lor(x,x0_guess[0],x0_guess[1],x0_guess[2],x0_guess[3])
    noise_slope_result = noise_slope(x,fit_values[1],fit_values[2])
    fine_freqs = np.logspace(np.log10(freq_range[0]),np.log10(freq_range[1]),10000)
    print(fine_freqs)
    knee = fine_freqs[np.argmin(np.abs( fit_values[1]*fine_freqs**-fit_values[2]-fit_values[0] ))]

    fit_dict = {'fit_values': fit_values,'fit_values_names':fit_values_names, 'sum_dev': sum_dev, 'fit_result': fit_result,'x0_guess_result':x0_guess_result,'evaluated_ranges':evaluated_ranges,'knee':knee,'noise_slope_result':noise_slope_result}#, 'x0':x0, 'z':z},'marginalized_2d':marginalized_2d,'marginalized_1d':marginalized_1d,
    return fit_dict   

    return fit_dict

def fit_psd_lor(x,y,**keywords):
    '''
    # keywards are
    # use_range ---is an n length tuple of frequencies to use while fitting
    # Example: [[1,57],[63,117],[123,177]] here we fit from 1 to 57Hz and 63 to 117 Hz and 123 to 177Hz avoid 60 Hz and harmonics
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # there are two way the function if fit one is it is fit without binning. In this case a error for the fit is calculated
    # by calculating the standard deviation of the surronding 50 or so points. Then the data below 10Hz has the error artificially
    # lowered so that the fitter doesn't ignore in when comparing it to the many more points at higher frequencies. The other way is to use
    # the log keyword which then log bins the data and calculates the error for each bin. This way there are around the same number of points
    # at low frequency as compared to high frequency
    # since there are less points at low frequencies than high frequencies I artificially increase the accuracy of the low frequency points
    # below sigma_increase_cutoff by scaling the error that the fitter uses by sigma_increase_factor
    '''
    if ('sigma_increase_cutoff' in keywords):
        sigma_increase_cutoff = keywords['sigma_increase_cutoff']
    else:
        #define default bounds
        sigma_increase_cutoff = 2. #(Hz)
    if ('sigma_increase_cutoff' in keywords):
        sigma_increase_factor = keywords['sigma_increase_factor']
    else:
        #define default bounds
        sigma_increase_factor = 5. 

    # bounds with out these some paramter might converge to non physical values
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        bounds = ([10**-20,0.0,0.0,9e-7],[10**-8,10**-8,3,0.001]) 
    if ('use_range' in keywords):
        use_range = keywords['use_range']
        # create an index of the values you want to fit
        index = np.where((x>use_range[0][0]) & (x<use_range[0][1]) )[0]
        for i in range(1,len(use_range)):
            index2 = np.where((x>use_range[i][0]) & (x<use_range[i][1]) )
            index = np.hstack((index,index2[0]))  
    else:
        index = range(0,x.shape[0])

    # initial conditions    
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess
        print("default initial guess used")        
        x0  = np.array([1.*10**(-15.75), 1.*10**(-17),1,0.00001]) # default values that work OK for superspec

    # log bin the data first or no    
    if ('log' in keywords):
        print("psd will be log binned before fitting")
        log = 1
        bins = np.logspace(np.log10(x[0]),np.log10(x[x.shape[0]-1]),100) #100 logspaced bins 
    else:
        log = 0

    if log == 1:
        binnedfreq_temp =  binned_statistic(x[index], x[index], bins=bins)[0]
        binnedvals_temp = binned_statistic(x[index], y[index], bins=bins)[0]
        binnedvals_std = binned_statistic(x[index], y[index], bins=bins, statistic = std_of_mean)[0]
        binnedfreq = binnedfreq_temp[~np.isnan(binnedfreq_temp)]
        binnedvals = binnedvals_temp[~np.isnan(binnedfreq_temp)]
        binnedstd = binnedvals_std[~np.isnan(binnedfreq_temp)]

    freqs = x[index]
    vals = y[index]

    if log ==0: #when fitting there are so many points at high frequencies compared to at low frequecies a fitting will almost ingnore the low frequency end
        # I get an extimate fo the noise by taking the standard deviation of each 10 consective points (will be some error for th last 10 points)
        std_pts = 100 # if this number is to low it seems to bias the fits to the lower side
        low_freq_index = np.where(freqs<sigma_increase_cutoff)
        temp = np.zeros((vals.shape[0],std_pts))
        # here I estimate the error by looking at the 100 surronding points and calculated the std      
        for i in range(0,std_pts):
            temp[:,i] = np.roll(vals,-i)
        sigma = np.std(temp,axis = 1)
        sigma[low_freq_index] = sigma[low_freq_index]/sigma_increase_factor # artificial pretend the noise at low frequcies is 5 time lower than every where else
        fit = optimization.curve_fit(noise_profile_lor, freqs, vals, x0 , sigma,bounds = bounds)
    else:
        sigma = binnedstd
        fit = optimization.curve_fit(noise_profile_lor, binnedfreq, binnedvals, x0 ,sigma,bounds = bounds)

    return fit


def fit_psd_lor_brute(x, y, n_grid_points=20, error=None, **keywords):
    """
    brute force fitting (the only way to fit with 4 or less variables)
    x is the psd frequency
    y is the psd magnitude
    n_grid_points is the deminsion of the 4 dimensional cube over which function will be evaluated
    keywords are
    ranges is the ranges for each parameter i.e. np.asarray(([a_low,b_low,c_low,d_low],[a_high,b_high,c_high,d_high]))
    freq_range = (f_low,f_high) bounds over which the psd should be fit
    white_freq = 50 (i.e) Hz if using the automated range it is very useful to specify where the noise psd is white (frequency independant)

    due to the vector nature of the calculations used by this brute for fitter
    n_grid_points will be limited by your computers ram and it grows fast
    for example 300points 50^4*(2bytes per float*(2arrays) = 3.5GB of ram
    just watch your resources when you fit if you exceed your ram you will write to disk and the fit will never finish

    To Do add in marginilaztions for error bars like in the brute force fitter in resonance fitting
    also add in the corner plot for marginalized values
    would be good to add in a nested version of this where it fits again over a smaller paramter space
    """

    if ('freq_range' in keywords):
        freq_range = keywords['freq_range']
    else:
        freq_range = (x[0], x[-1])

    index_for_fitting = np.where(((x >= freq_range[0]) & (x <= freq_range[1])))

    if ('ranges' in keywords):
        ranges = keywords['ranges']
    else:
        if ('white_freq' in keywords):
            white_freq = keywords['white_freq']
            white_index = np.argmin(np.abs(x[index_for_fitting] - white_freq))
        else:
            white_index = len(y[index_for_fitting]) // 2  # noise is white in the middle of psd

        white_guess = y[index_for_fitting][white_index]
        x0_guess = np.array([white_guess,
                             (y[index_for_fitting][0] - white_guess) / x[index_for_fitting][0] ** (-1.),
                             # assume 1/f dominates at lowest frequency but still subtract off white noise
                             1,  # guess 1/f is index is 1
                             1. / 2 / np.pi / x[index_for_fitting][np.argmin(np.abs(white_guess / 2. - y[
                                 index_for_fitting]))]])  # look for 3dB decrease from white noise guess
        print("guess values are")
        print(x0_guess)
        ranges = np.asarray(([x0_guess[0] / 2, x0_guess[1] / 20, 0.5, x0_guess[3] / 2],
                             [x0_guess[0] * 2, x0_guess[1] * 10, 2, x0_guess[3] * 2]))

    if error is None:
        error = np.ones(len(x[index_for_fitting]))

    a_values = np.linspace(ranges[0][0], ranges[1][0], n_grid_points)
    b_values = np.linspace(ranges[0][1], ranges[1][1], n_grid_points)
    c_values = np.linspace(ranges[0][2], ranges[1][2], n_grid_points)
    d_values = np.linspace(ranges[0][3], ranges[1][3], n_grid_points)
    evaluated_ranges = np.vstack((a_values, b_values, c_values, d_values))

    a, b, c, d = np.meshgrid(a_values, b_values, c_values, d_values, indexing="ij")  # always index ij

    evaluated = noise_profile_lor_vec(x[index_for_fitting], a, b, c, d)
    data_values = np.reshape(y[index_for_fitting], (y[index_for_fitting].shape[0], 1, 1, 1, 1))
    error = np.reshape(error, (y[index_for_fitting].shape[0], 1, 1, 1, 1))
    # print(evaluated.shape)
    # print(data_values.shape)
    # print(error.shape)
    sum_dev = np.sum(((evaluated - data_values) ** 2 / error ** 2),
                     axis=0)  # comparing in magnitude space rather than magnitude squared
    # print(sum_dev.shape)

    min_index = np.where(sum_dev == np.min(sum_dev))
    print("grid values at minimum are")
    print(min_index)
    index1 = min_index[0][0]
    index2 = min_index[1][0]
    index3 = min_index[2][0]
    index4 = min_index[3][0]
    fit_values = np.asarray((a_values[index1], b_values[index2], c_values[index3], d_values[index4]))
    fit_values_names = ('a (white)', 'b (1/f)', 'c (1/f exponent)', 'd (tau)')
    fit_result = noise_profile_lor(x, a_values[index1], b_values[index2], c_values[index3], d_values[index4])
    x0_guess_result = noise_profile_lor(x, x0_guess[0], x0_guess[1], x0_guess[2], x0_guess[3])
    noise_slope_result = noise_slope(x, fit_values[1], fit_values[2])
    fine_freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 10000)
    print(fine_freqs)
    knee = fine_freqs[np.argmin(np.abs(fit_values[1] * fine_freqs ** -fit_values[2] - fit_values[0]))]

    fit_dict = {'fit_values': fit_values, 'fit_values_names': fit_values_names, 'sum_dev': sum_dev,
                'fit_result': fit_result, 'x0_guess_result': x0_guess_result, 'evaluated_ranges': evaluated_ranges,
                'knee': knee,
                'noise_slope_result': noise_slope_result}  # , 'x0':x0, 'z':z},'marginalized_2d':marginalized_2d,'marginalized_1d':marginalized_1d,
    return fit_dict


def fit_psd(x,y,plot = False,**keywords):
    """
    # keywards are
    # use_range ---is an n length tuple of frequencies to use while fitting
    # Example: [[1,57],[63,117],[123,177]] here we fit from 1 to 57Hz and 63 to 117 Hz and 123 to 177Hz avoid 60 Hz and harmonics
    # x0    --- intial guess for the fit this can be very important becuase because least square space over all the parameter is comple
    # there are two way the function if fit one is it is fit without binning. In this case a error for the fit is calculated
    # by calculating the standard deviation of the surronding 50 or so points. Then the data below 10Hz has the error artificially
    # lowered so that the fitter doesn't ignore in when comparing it to the many more points at higher frequencies. The other way is to use
    # the log keyword which then log bins the data and calculates the error for each bin. This way there are around the same number of points
    # at low frequency as compared to high frequency
    # since there are less points at low frequencies than high frequencies I artificially increase the accuracy of the low frequency points
    # below sigma_increase_cutoff by scaling the error that the fitter uses by sigma_increase_factor
    """
    if ('sigma_increase_cutoff' in keywords):
        sigma_increase_cutoff = keywords['sigma_increase_cutoff']
    else:
        #define default bounds
        sigma_increase_cutoff = 2. #(Hz)
    if ('sigma_increase_cutoff' in keywords):
        sigma_increase_factor = keywords['sigma_increase_factor']
    else:
        #define default bounds
        sigma_increase_factor = 5. 

    index_for_fitting = np.where(((x>=freq_range[0]) & (x <=freq_range[1])))
        
    if ('ranges' in keywords):
        ranges = keywords['ranges']
    else:
        if ('white_freq' in keywords):
            white_freq = keywords['white_freq']
            white_index = np.argmin(np.abs(x[index_for_fitting]-white_freq))
        else:
            white_index = len(y[index_for_fitting])//2 # noise is white in the middle of psd

        white_guess = y[index_for_fitting][white_index]
        x0_guess = np.array([ white_guess,
                                 (y[index_for_fitting][0]-white_guess)/x[index_for_fitting][0]**(-1.), # assume 1/f dominates at lowest frequency but still subtract off white noise
                                 1,# guess 1/f is index is 1
                                 1./2/np.pi/x[index_for_fitting][np.argmin(np.abs(white_guess/2.-y[index_for_fitting]))]]) # look for 3dB decrease from white noise guess
        print("guess values are")
        print(x0_guess)
        ranges = np.asarray(([x0_guess[0]/2,x0_guess[1]/20,0.5,x0_guess[3]/2],[ x0_guess[0]*2,x0_guess[1]*10,2,x0_guess[3]*2]))
                

    if error is None:
        error = np.ones(len(x[index_for_fitting]))

    a_values = np.linspace(ranges[0][0],ranges[1][0],n_grid_points)
    b_values = np.linspace(ranges[0][1],ranges[1][1],n_grid_points)
    c_values = np.linspace(ranges[0][2],ranges[1][2],n_grid_points)
    d_values = np.linspace(ranges[0][3],ranges[1][3],n_grid_points)
    evaluated_ranges = np.vstack((a_values,b_values,c_values,d_values))

    a,b,c,d = np.meshgrid(a_values,b_values,c_values,d_values,indexing = "ij") #always index ij

    evaluated = noise_profile_lor_vec(x[index_for_fitting],a,b,c,d)
    data_values = np.reshape(y[index_for_fitting],(y[index_for_fitting].shape[0],1,1,1,1))
    error = np.reshape(error,(y[index_for_fitting].shape[0],1,1,1,1))
    #print(evaluated.shape)
    #print(data_values.shape)
    #print(error.shape)
    sum_dev = np.sum(((evaluated-data_values)**2/error**2),axis = 0) # comparing in magnitude space rather than magnitude squared
    #print(sum_dev.shape)

    min_index = np.where(sum_dev == np.min(sum_dev))
    print("grid values at minimum are")
    print(min_index)
    index1 = min_index[0][0]
    index2 = min_index[1][0]
    index3 = min_index[2][0]
    index4 = min_index[3][0]
    fit_values = np.asarray((a_values[index1],b_values[index2],c_values[index3],d_values[index4]))
    fit_values_names = ('a (white)','b (1/f)','c (1/f exponent)','d (tau)')
    fit_result = noise_profile_lor(x,a_values[index1],b_values[index2],c_values[index3],d_values[index4])
    x0_guess_result = noise_profile_lor(x,x0_guess[0],x0_guess[1],x0_guess[2],x0_guess[3])
    noise_slope_result = noise_slope(x,fit_values[1],fit_values[2])
    fine_freqs = np.logspace(np.log10(freq_range[0]),np.log10(freq_range[1]),10000)
    print(fine_freqs)
    knee = fine_freqs[np.argmin(np.abs( fit_values[1]*fine_freqs**-fit_values[2]-fit_values[0] ))]

    fit_dict = {'fit_values': fit_values,'fit_values_names':fit_values_names, 'sum_dev': sum_dev, 'fit_result': fit_result,'x0_guess_result':x0_guess_result,'evaluated_ranges':evaluated_ranges,'knee':knee,'noise_slope_result':noise_slope_result}#, 'x0':x0, 'z':z},'marginalized_2d':marginalized_2d,'marginalized_1d':marginalized_1d,
    return fit_dict   

    return fit_dict

    if ('use_range' in keywords):
        use_range = keywords['use_range']
        # create an index of the values you want to fit
        index = np.where((x>use_range[0][0]) & (x<use_range[0][1]) )[0]
        for i in range(1,len(use_range)):
            index2 = np.where((x>use_range[i][0]) & (x<use_range[i][1]) )
            index = np.hstack((index,index2[0]))  
    else:
        index = range(0,x.shape[0])

    if ('uniform_weight' in keywords):
        uniform_weight = keywords['uniform_weight']
    else:
        uniform_weight = False

    # initial conditions    
    if ('x0' in keywords):
        x0 = keywords['x0']
    else:
        #define default intial guess # guess for b: b*freqs**-c = value => b = value/freqs**-c
        print("default initial guess used")        
        x0  = np.array([y[index][-1], y[index][0]/x[index][0]**(-1.) ,1]) # default values that work OK for superspec
    # bounds with out these some paramter might converge to non physical values
    if ('bounds' in keywords):
        bounds = keywords['bounds']
    else:
        #define default bounds
        print("default bounds used")
        #bounds = ([10**-20,10**-20,0],[10**-10,10**-10,3])
        bounds = ([x0[0]/10.,x0[1]/10.,0],[x0[0]*10,x0[1]*10.,3])

    # log bin the data first or no    
    if ('log' in keywords):
        print("psd will be log binned before fitting")
        log = 1
        bins = np.logspace(np.log10(x[0]),np.log10(x[x.shape[0]-1]),100) #100 logspaced bins 
    else:
        log = 0
        bins = x.shape[0] # doesn't bin at all

 
    binnedfreq_temp =  binned_statistic(x[index], x[index], bins=bins)[0]
    binnedvals_temp = binned_statistic(x[index], y[index], bins=bins)[0]
    binnedvals_std = binned_statistic(x[index], y[index], bins=bins, statistic = std_of_mean)[0]
    binnedfreq = binnedfreq_temp[~np.isnan(binnedfreq_temp)]
    binnedvals = binnedvals_temp[~np.isnan(binnedfreq_temp)]
    binnedstd = binnedvals_std[~np.isnan(binnedfreq_temp)]

    freqs = x[index]
    vals = y[index] 

    if log ==0: #when fitting there are so many points at high frequencies compared to at low frequecies a fitting will almost ingnore the low frequency end
        # I get an extimate fo the noise by taking the standard deviation of each 10 consective points (will be some error for th last 10 points)
        std_pts = 100 # if this number is to low it seems to bias the fits to the lower side
        low_freq_index = np.where(freqs<sigma_increase_cutoff)
        temp = np.zeros((vals.shape[0],std_pts))       
        for i in range(0,std_pts):
            temp[:,i] = np.roll(vals,-i)
        sigma = np.std(temp,axis = 1)
        sigma[low_freq_index] = sigma[low_freq_index]/sigma_increase_factor # artificial pretend the noise at low frequcies is 10 time lower than every where else
        fit = optimization.curve_fit(noise_profile, freqs, vals, x0 , sigma,bounds = bounds)
        print("hello")
    else:
        if uniform_weight == True:
            sigma = np.ones(len(binnedstd))*np.mean(binnedstd)

        else:
            sigma = binnedstd
        fit = optimization.curve_fit(noise_profile, binnedfreq, binnedvals, x0 ,sigma,bounds = bounds)

    if plot == True:
        plt.loglog(binnedfreq,binnedvals,label = "Data",linewidth = 2)
        #plt.errorbar(binnedfreq,binned_psd_log,binned_std_log, fmt='o')
        plt.loglog(binnedfreq,noise_profile(binnedfreq,x0[0],x0[1],x0[2]),linewidth = 2,label = "Initial Guess")
        plt.loglog(binnedfreq,noise_profile(binnedfreq,fit[0][0],fit[0][1],fit[0][2]),linewidth = 2,label = "Fit")
        plt.loglog(binnedfreq,noise_slope(binnedfreq,fit[0][1],fit[0][2]),linewidth = 2, label = "1/f^" + str(fit[0][2])[0:4])
        plt.loglog(binnedfreq,noise_white(binnedfreq,fit[0][0]),linewidth = 2,label = "White"+ " " +  str(fit[0][0]*10**16)[0:4] +" x10^-16")
        plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Sxx (1/Hz)")
        plt.ylim(np.min(y),np.max(y))
        plt.show(block = False)
        

    return fit


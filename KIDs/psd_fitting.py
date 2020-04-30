import numpy as np
from scipy.stats import binned_statistic
import scipy.optimize as optimization

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



def fit_psd(x,y,**keywords):
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
        bounds = ([10**-20,10**-20,0],[10**-10,10**-10,3])  
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
        #define default intial guess
        print("default initial guess used")        
        x0  = np.array([1.*10**(-15.75), 1.*10**(-17),1]) # default values that work OK for superspec

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

    return fit


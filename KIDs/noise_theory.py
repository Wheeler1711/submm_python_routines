import numpy as np
import scipy.special as special

# this is a list of definitions that can be used to predict noise in KIDS
# right now it just contains the nessasary requirements for perdicting G-R noise in TiN
# I should expand it to include some of the definitions in Jason code
# this is adapted from some IDL code I got from Matt Bradford

#Written by Jordan on 12/12/2016

#Change log
# 1/6/2017 added nqp_min as a specified parameter for grnoise 



def nqp(t,tc,v,nqp_min):

    #if ~keyword_set(nqp_min) then nqp_min=400. ; per cubic micron: zero-Temp residual QP density
    #V is in cubic microns
    #N0=1.72e10 for aluminum
    N0=4.e10 # for TiN
    N0 = N0/1.6e-19 # now microns^3 / Joule

    #Delta = double (3.5 * 1.381e-23 * tc)
    Delta = 1.74 * 1.381e-23 * tc
    Nqp = v*2*N0*np.sqrt(2*np.pi*Delta*1.381e-23*t)*np.exp(-1*Delta/(1.381e-23*t))+v*nqp_min

    return Nqp

def deltaf_f(t, tc, nu, alpha, gamma):
    #From Steve Hailey-Dunsheath, 2 Februay 2016
    #Calculate a model for the fractional frequency shift due to change in bath temperature
    #This is (alpha)*(gamma/2)*[sigma2(T)/sigma2(T=0)-1], where sigma2(T) is
    #from equation 2.92 in Gao thesis, and equation 20 in Gao+08 JLTP

    #nu is in MHz
    #if ~keyword_set(alpha) then alpha=1
    #if ~keyword_set(gamma) then gamma=1
    d_0 = 1.762*tc
    xi = 6.626e-34*nu*1.e6/(2.*1.38e-23*t)
    model = -1.*alpha*gamma/2.*np.exp(-1.*d_0/t)*((2.*np.pi*t/d_0)**0.5 + 2.*np.exp(-1.*xi)*special.iv(0,xi))
    return model

def df_response(t,tc,f):
    #calculate d (df/f) / dT via finite difference
    #f is in MHz
    #calls deltaf_f which computes frequency shift
    delta_t = t/20.
    dff_dt = (deltaf_f(t+delta_t,tc,f,1,1) - deltaf_f(t-delta_t,tc,f,1,1))/(2*delta_t)
    return dff_dt


def grnoise(t,tc,V,tau_qp,N0,f,nqp_min):
# V in cubic microns
#if ~keyword_set(N0) then N0=4.e10 ; microns^3 / eV
#if ~keyword_set(tau_qp) then tau_qp = 5e-6 ; sec
#if ~keyword_set(nqp_min) then nqp_min=400. ; QP per cubic micron at zero Temp
    N0 = N0/1.6e-19 # now microns^3 / Joule

    #ef^2 = 4 beta^2 Nqp tau_qp
    #beta = df_0 / d Nqp , so use (df_0/ dT) (dT / dNqp)

    #Delta = double (3.5 * 1.381e-23 * tc)
    Delta = 1.74*1.381e-23*tc
    #Nqp = v * 2 * N0 * sqrt(2*!pi*delta * 1.381e-23 * T) *exp(-1*Delta / (1.381e-23 * T)) + v * nqp_min
    dNqp_dt = V*2.*N0*np.sqrt(2*np.pi*Delta*1.318e-23)*np.exp(-1.*Delta/(1.381e-23*t))*(1./(2.*np.sqrt(t)) + np.sqrt(t)*Delta/1.381e-23/t**2)
    delta_t = t/30.
    dNqp_dt = (nqp(t+delta_t,tc,V,nqp_min)-nqp(t-delta_t,tc,V,nqp_min))/(2*delta_t)

    beta = df_response(t,tc,f)/dNqp_dt
    #assume a frequency of 100 MHz here, shouldn't matter.

    ef2 = 4. * beta**2 * nqp(t,tc,V,nqp_min) * tau_qp
    return ef2

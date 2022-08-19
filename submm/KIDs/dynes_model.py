#!/usr/bin/env python

"""This is an implementation to the model for disordered superconductivity using
Dynes model of elastic scattering DOS broadening. This is mostly following the
Zemlicka et al. 2015 derivation of a Dynes type model with modified propagators
in the Nam equations, making the calculations very similar to the AG model in
the 'disordered_model.py' file

Joseph Redford 10/21/2021
"""

import multiprocessing as mp

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy import interpolate

#constants
k_B = 1.380649e-23 #J/K
h = 6.62607e-34 #J*s

def calc_delta_int(Delta0, gamma, kT):
    def integrand(E, Delta0, gamma, delta, kT):
        term1 = np.real(np.power(E * E - (Delta0 - 1.j * gamma)**2, -0.5))
        term2 = np.tanh(0.5 * E / kT) * np.real(np.power(E * E - (delta
                - 1.j * gamma)**2, -0.5))
        return term1 - term2

    def prepare_integrand(delta):
        int_fn = lambda E: integrand(E, Delta0, gamma, delta, kT)
        return int_fn
    root_fn = lambda delta: quad(prepare_integrand(delta), 0., np.inf)[0]
    delta_sol = root_scalar(root_fn, x0=Delta0, x1 = 0.)

    return delta_sol.root


def propagator_n(E, delta, gamma):
    denom = np.power(((E + 1.j * gamma)**2.) - delta * delta, -0.5) 
    num = np.sign(E) * (E + 1.j * gamma)

    return num * denom


def propagator_p(E, delta, gamma):
    denom = np.power(((E + 1.j * gamma)**2.) - delta * delta, -0.5) 
    num = np.sign(E) * delta

    return num * denom


def calc_sigma2_thermal(kT, delta, h_nu, gamma):
    def integrand(E):
        if kT > 0.:
            beta = 1. / kT
        else:
            beta = np.inf
        occupation = np.tanh(0.5 * beta * (E + h_nu))
        prop_term = np.imag(propagator_n(E, delta, gamma)) * np.real(
                propagator_n(E + h_nu, delta, gamma)) + np.imag(
                propagator_p(E, delta, gamma)) * np.real(
                propagator_p(E + h_nu, delta, gamma))
        return occupation * prop_term

    int_val, _ = quad(integrand, -np.inf, np.inf)
    return  int_val/ h_nu


def calc_sigma1_thermal(kT, delta, h_nu, gamma):
    def integrand(E):
        if kT > 0.:
            beta = 1. / kT
        else:
            beta = np.inf
        occupation = 0.5 * (np.tanh(0.5 * beta * (E + h_nu))
                - np.tanh(0.5 * beta * E))
        prop_term = np.real(propagator_n(E, delta, gamma)) * np.real(
                propagator_n(E + h_nu, delta, gamma)) + np.real(
                propagator_p(E, delta, gamma)) * np.real(
                propagator_p(E + h_nu, delta, gamma))
        return occupation * prop_term

    int_val, _ = quad(integrand, -np.inf, np.inf)
    return  int_val/ h_nu



def sigma2_fit_fn(kT, Delta0, h_nu, gamma):
    current_delta = calc_delta_int(Delta0, gamma, kT)
    sigma2_ratio = calc_sigma2_thermal(kT, current_delta, h_nu, gamma)

    return sigma2_ratio


def thermal_Qi(kT, Delta0, h_nu, gamma):
    current_delta = calc_delta_int(Delta0, gamma, kT)
    sigma2_ratio = calc_sigma2_thermal(kT, current_delta, h_nu, gamma)
    sigma1_ratio = calc_sigma1_thermal(kT, current_delta, h_nu, gamma)
    
    return sigma2_ratio / sigma1_ratio


def n_qp(kT, delta, gamma, N_0):
    def integrand(E):
        pop_level = 1. - np.tanh(0.5 * E / kT)
        DOS = np.real((E + 1.j * gamma) * np.power(
                (E + 1.j * gamma)**2. - delta * delta, -0.5))

        return pop_level * DOS

    N_pop, _ = quad(integrand, 0., np.inf)

    return 2. * N_0 * N_pop


def S2_calc(kT, delta, gamma, N_0, h_nu):
    n_thermal = n_qp(T, delta, gamma, N_0)
    sigma2_0 = calc_sigma2_thermal(0., delta, h_nu, gamma)
    S2_prefix = 0.5 * (calc_sigma2_thermal(kT, delta, h_nu, gamma)
            - sigma2_0)/ (sigma2_0 * n_thermal)

    return S2_prefix


def GR_Sxx(kT, delta, gamma, N_0, h_nu, tau, tau_max, V):
    #Delta = calc_delta_int(T_c, alpha, T, N_V)
    #Delta_0 = calc_delta_int(T_c, alpha, 0., N_V)
    n_thermal = n_qp(kT, delta, gamma, N_0)
    if n_thermal == 0.:
        return 0.
    sigma2_0 = calc_sigma2_thermal(0., delta, h_nu, gamma)
    S2_prefix = (sigma2_0 - calc_sigma2_thermal(kT, delta, h_nu, gamma)) \
            / (sigma2_0 * n_thermal)
    fluctuation_term = 2. * tau * (1. + tau / tau_max) * n_thermal / V

    return S2_prefix * S2_prefix * fluctuation_term


#!/usr/bin/env python

"""This is an implementation to the model for disordered TiN mostly following
the approach laid out in Coumou's thesis, but broken up and reparameterized in
ways I suspect will be more conducive to numerical calculation and integration.
It ends up being an implementation of the Abrikosov-Gor'kov (AG) model with the
admittance calculations of the film calculated with Nam generalizations of the
Matthis-Bardeen integrals.

Joseph Redford 5/31/2021
"""

import multiprocessing as mp

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy import interpolate

#constants
k_B = 1.380649e-23 #J/K
h = 6.62607e-34 #J*s


def get_usadel_x(E, Delta, eta):
    """This calculates the value of e^(i \\theta) = x from the Usadel equation.
    I think this is better than fully backing out theta or cos/sin theta because
    this then becomes a problem of finding polynomial roots and all later
    quantities can be put into terms of x with algebra and the Euler identity

    E is the energy level being calculated
    Delta is the average gap energy of the disordered superconductor
    eta is the depairing energy/gap broadening parameter

    In terms of units, E and Delta should just have to share units but could
    be put in any energy unit or even a temperature using a kT scaling
    eta is dimensionless
    """
    poly_coeff = [0.5j * Delta * eta, Delta + E, 0., Delta - E,
            -0.5j * Delta * eta]
    roots = np.roots(poly_coeff)
    roots = roots[np.argsort(np.imag(roots))]
    #the two roots that remain purely imaginary and are not physical are
    #maximum and minimum in the imaginary component for positive energy values
    #once the real component is non-zero the two correct roots should only
    #differ by sign of the real component, so taking positive real part to
    #consistently be on the same branch
    roots = roots[1:3]
    delta = roots[1] - roots[0]
    if np.imag(delta) > np.real(delta):
        return roots[0]
    else:
        return roots[1]


def get_usadel_x2(E, Delta, alpha):
    """This calculates the value of e^(i \\theta) = x from the Usadel equation.
    I think this is better than fully backing out theta or cos/sin theta because
    this then becomes a problem of finding polynomial roots and all later
    quantities can be put into terms of x with algebra and the Euler identity

    E is the energy level being calculated
    Delta is the average gap energy of the disordered superconductor
    alpha is the gap broadening parameter

    In terms of units, E and Delta should just have to share units but could
    be put in any energy unit or even a temperature using a kT scaling
    eta is dimensionless
    """
    poly_coeff = [1.j * alpha, 2. * (Delta + E), 0., 2. * (Delta - E),
            -1.j * alpha]
    roots = np.roots(poly_coeff)
    roots = roots[np.argsort(np.imag(roots))]
    #the two roots that remain purely imaginary and are not physical are
    #maximum and minimum in the imaginary component for positive energy values
    #once the real component is non-zero the two correct roots should only
    #differ by sign of the real component, so taking positive real part to
    #consistently be on the same branch
    roots = roots[1:3]
    delta = roots[1] - roots[0]
    if np.imag(delta) > np.real(delta):
        return roots[0]
    else:
        return roots[1]


def g1(E1, E2, Delta, alpha):
    """This is the g1 greens function used in the dissipation calculations 

    E1 and E2 are the two energy arguments of the greens function
    Delta is the average gap energy
    alpha is the the depairing energy/gap broadening parameter

    just like the usadel function, E1, E2, and Delta could be any units as
    long as they are the same
    """
    x1 = get_usadel_x2(np.abs(E1), Delta, alpha)
    x2 = get_usadel_x2(np.abs(E2), Delta, alpha)
    x1_sq_mag = np.real(x1 * np.conj(x1)) #I am not sure the real is necessary,
            #but ensures that this is not complex data type
    x2_sq_mag = np.real(x2 * np.conj(x2))
    
    g1_val = np.real(x1 * x1_sq_mag + x1) * np.real(x2 * x2_sq_mag + x2)
    g1_val += np.real(x1 * x1_sq_mag - x1) * np.real(x2 * x2_sq_mag - x2)
    g1_val /= (x1_sq_mag * x2_sq_mag)

    return 0.25 * g1_val


def g2(E1, E2, Delta, alpha):
    """This is the g2 greens function used in the inductance calculations 

    E1 and E2 are the two energy arguments of the greens function
    Delta is the average gap energy
    eta is the the depairing energy/gap broadening parameter

    just like the usadel function, E1, E2, and Delta could be any units as
    long as they are the same
    """
    x1 = get_usadel_x2(np.abs(E1), Delta, alpha)
    x2 = get_usadel_x2(np.abs(E2), Delta, alpha)
    x1_sq_mag = np.real(x1 * np.conj(x1)) #I am not sure the real is necessary,
            #but ensures that this is not complex
    x2_sq_mag = np.real(x2 * np.conj(x2))

    g2_val = 0.5 * (x1_sq_mag * x2_sq_mag - 1) / (x1_sq_mag * x2_sq_mag)
    g2_val *= np.imag(x1) * np.real(x2)

    return 0.25 * g2_val #double check the 0.25 prefix


def get_Eg_fraction(eta, cutoff, precision):
    """Since only relative values of E and Delta to each other matter,
    I am just going to find the fraction of Delta Eg is at (the effective
    fractional reduction of the gap due to eta). So eta is and cutoff is the
    only parameters

    This search will be one sided as the real value of x is 0 (to within
        precision of the root finder routine) until suddenly you cross Eg

    I will use cutoff to set a threshold above the noise of the root finder
        imprecision, that generates some small real parts to some of the sub-gap
        roots and then a "binary search" type stepping and testing. If the step
        is above the Eg step again, else halve the step. It is primitive, but
        should work just fine.
    """
    current = 1.
    step = 0.5
    while step >= precision:
        if current - step < 0.:
            step *= 0.5 #shouldn't go negative, not sure the Usadel solver
                    #will work, and the physical system should be symmetric
                    #around 0 anyway
            continue
        temp_x = get_usadel_x(current-step, 1., eta)
        if np.real(temp_x) > cutoff:
            current -= step
        else:
            step *= 0.5

    return current


def analytical_Eg(Delta, alpha):
    if Delta < alpha:
        return 0.
    else:
        return (Delta**(2./3.) - alpha**(2./3.))**1.5 / Delta


def thermal_gap_suppression(T, Delta_0):
    Tc = Delta_0 / 1.76 #assuming delta is really k_B Delta, so as always, can
            #keep this unit agnostic
    eff_delta = Delta_0 * np.sqrt(1. - T / Tc) #this is just a first test to see
        #if the gap suppression is necessary to get my results to match
        #the plots of Coumou, I remember this being a simple approximation
        #of the gap suppression with temperature

    #return eff_delta
    return Delta_0


def fermi(E, mu, beta):
    """I use $\\beta = 1 / (k T)$ just because then I can have E, mu, and beta
    in similar units and this can accept anything that is consistent in the unit
    system.

    E is the energy level
    mu is the chemical potential
    beta is inverse kT so should be in inverse units of E and mu
    """
    ex_term = np.exp(beta * (E - mu))

    return 1. / (1. + ex_term)


def calc_sigma1_thermal(T, Delta, h_nu, alpha):
    """This calculates the $\\frac{\\sigma_1}{\\sigma_n}$ with numerical
    integration of the greens function formalism from Nam.
    As with everything I will assume the proper factors of k and h were applied
    such that T, Delta, and h_nu are all in same units of energy/effective
    temperature.

    T is the temperature
    Delta is the average gap energy
    h_nu is the microwave photon energy
    eta is the depairing energy/gap broadening parameter
    """
    E_g = analytical_Eg(Delta, alpha)
    beta = 1. / T
    integrand1 = lambda E: g1(E, E + h_nu, Delta, alpha) * np.tanh(
            0.5 * beta * (E + h_nu))
    integrand2 = lambda E: g1(E, E + h_nu, Delta, alpha) * (
            np.tanh(0.5 * beta * (E + h_nu)) - np.tanh(0.5 * E * beta))
    term1, _ = quad(integrand1, E_g - h_nu, -E_g) #fine tune integration
            #parameters later
    term2, _ = quad(integrand2, E_g, np.inf)

    return (term1 + term2) / h_nu


def calc_sigma2_thermal(T, Delta, h_nu, alpha):
    """This calculates the $\\frac{\\sigma_1}{\\sigma_n}$ with numerical
    integration of the greens function formalism from Nam.
    As with everything I will assume the proper factors of k and h were applied
    such that T, Delta, and h_nu are all in same units of energy/effective
    temperature.

    T is the temperature
    Delta is the average gap energy
    h_nu is the microwave photon energy
    eta is the depairing energy/gap broadening parameter
    """
    #eff_delta = thermal_gap_suppression(T, Delta)
    E_g = analytical_Eg(Delta, alpha)
    #E_g *= eff_delta #convert in our delta for limits of integration
    if T <= 0.:
        fermi_val = lambda E: 0.
    else:
        beta = 1. / T
        fermi_val = lambda E : fermi(E, 0., beta)
    integrand1 = lambda E: g2(E, E + h_nu, Delta, alpha) * (1 - 2 * fermi_val(
                E + h_nu))
    integrand2 = lambda E: g2(E + h_nu, E, Delta, alpha) * (1 - 2 * fermi_val(
                E))
    limit = np.max([E_g - h_nu, -E_g]) #for KIDS I don't think we will ever
        #end up with -E_g being the greater
    term1, _ = quad(integrand1, limit, np.inf) #fine tune integration
            #parameters later
    term2, _ = quad(integrand2, E_g, np.inf)

    return (term1 + term2) / h_nu


def iterative_gap(T, T_c, alpha):
    def f(series_cutoff, delta, alpha, T, T_c):
        series_diff = np.inf
        series_sum = 0.
        i = 0
        while series_diff > series_cutoff:
            omega = (2. * i + 1.) * T * np.pi
            #x = matsubara_usadel_x(omega, curr_delta, alpha)
            #sin_term = -0.5j * (x + 1. / x)
            #term = (curr_delta / omega - np.real(sin_term))
            term = iterative_gap_matsubara_arg(omega, delta, alpha)
            series_sum += term
            series_diff = np.abs(term)
            i += 1
        new_delta = np.real(series_sum * 2. * np.pi * T)
        return new_delta

    #alpha = T_c * eta / 1.76
    cutoff = 1e-6
    min_T = 0.002
    if T < min_T:
        T = min_T
        #return Delta_0
    series_cutoff = 5e-8
    #I will assume T is actually kT in same units as Delta_0
    #using secant method for root finding
    curr_delta = T_c * 3.
    #if alpha / T_c > 0.1:
        #last_delta = (1. - (alpha / T_c)**0.6) * 1.76 * T_c
    #else:
    last_delta = 0.9 * 1.76 * T_c
    curr_f =  f(series_cutoff, curr_delta, alpha, T, T_c)
    last_f =  f(series_cutoff, last_delta, alpha, T, T_c)
    diff = np.abs(curr_delta - last_delta)
    a = np.log(T_c / T)
    while diff > cutoff:
        #secant iteration
        new_delta = (last_delta * (curr_f - a * curr_delta) - curr_delta * (
                last_f - a * last_delta)) / (curr_f - a * curr_delta - last_f
                + a * last_delta)
        diff = np.abs(curr_delta - new_delta)
        #print(diff)
        last_delta = curr_delta
        curr_delta = new_delta
        last_f = curr_f
        curr_f = f(series_cutoff, curr_delta, alpha, T, T_c)

    return curr_delta


def matsubara_usadel_x(omega, delta, alpha):
    poly_coeff = [alpha, 2. * (omega - 1.j * delta), 0.,
            -2. * (omega + 1.j * delta), -alpha]
    roots = np.roots(poly_coeff)
    roots = sorted(roots)

    return roots[3]


def iterative_gap_matsubara_arg(omega, delta, alpha):
    y = delta / omega
    poly_coeff = [alpha * alpha,
            2. * alpha * delta - 4. * alpha * alpha * y,
            6. * alpha * alpha * y * y - 6. * alpha * delta * y + omega * omega
            - alpha * alpha + delta * delta,
            6. * alpha * delta * y * y - 4. * alpha * alpha * y * y * y
            - 2. * (omega * omega - alpha * alpha + delta * delta) * y
            - 2. * alpha * delta,
            alpha * alpha * (y**4.) - 2. * alpha * delta * (y**3.)
            + (omega * omega - alpha * alpha + delta * delta) * y * y
            + (2. * alpha / omega - 1.) * delta * delta]
    roots = np.roots(poly_coeff)
    #roots = roots[np.argsort(-1.j * roots)] #sort by imaginary
    #roots = np.sort(roots)

    return np.real(roots[3]) #should be real root, making sure return data type
            #is not complex, but float

def calc_sigma2_Tc(T, T_c, h_nu, alpha):
    delta = iterative_gap(T, T_c, alpha)
    if delta < 1e-8:
        return 0.

    return calc_sigma2_thermal(T, delta, h_nu, alpha)


def calc_sigma1_Tc(T, T_c, h_nu, alpha):
    delta = iterative_gap(T, T_c, alpha)
    if delta < 1e-7:
        return 1.

    return calc_sigma1_thermal(T, delta, h_nu, alpha)


def calc_eff_Tc(T_c0, alpha):
    delta_thres = 1e-5
    zero_thres = 1e-8
    if alpha == 0.:
        return T_c0
    #right now, brute force binary search
    low_delta = iterative_gap(0., T_c0, alpha)
    low_Tc = 0.
    high_Tc = T_c0
    high_delta = iterative_gap(high_Tc, T_c0, alpha)
    while (high_Tc - low_Tc) / high_Tc >= 1e-5:
        next_Tc = 0.5 * (high_Tc + low_Tc)
        print(next_Tc)
        next_delta = iterative_gap(next_Tc, T_c0, alpha)
        if next_delta > zero_thres:
            low_Tc = next_Tc
            low_delta = next_delta
        else:
            high_Tc = next_Tc
            high_delta = next_delta

    return new_Tc


def map_fn(vals):
    ret_val = calc_eff_Tc(vals[0], vals[1])
    print(vals[1])

    return ret_val


def calc_Tc_var(Tc, delta_alpha):
    alpha_vals = np.linspace(0.01, Tc, int(np.ceil(Tc /delta_alpha)))
    #eff_Tc = np.ndarray(alpha_vals.shape)
    pool = mp.Pool()
    vals = [(Tc, alpha) for alpha in alpha_vals]
    eff_Tc = map(map_fn, vals)
    eff_Tc = [val for val in eff_Tc]
    print(eff_Tc)
    eff_Tc = np.array(eff_Tc)
    #for i, alpha in enumerate(alpha_vals):
    #    print(i)
    #    eff_Tc[i] = calc_eff_Tc(Tc, alpha)

    return eff_Tc, alpha_vals



def calc_delta_temp(delta, T_c, alpha):
    def f(series_cutoff, delta, alpha, T, T_c):
        #if T > T_c:
        #    return T
        #if T <= 0.:
        #    return T_c - T
        series_diff = np.inf
        series_sum = 0.
        i = 0
        while series_diff > series_cutoff:
            omega = (2. * i + 1.) * T * np.pi
            term = iterative_gap_matsubara_arg(omega, delta, alpha)
            series_sum += term
            series_diff = np.abs(term)
            i += 1
        new_delta = np.real(series_sum * 2. * np.pi * T)
        return new_delta

    #alpha = T_c * eta / 1.76
    cutoff = 1e-6
        #return Delta_0
    series_cutoff = 5e-8
    #I will assume T/T_c is actually kT in same units as Delta_0
    #using secant method for root finding
    current_T = T_c
    last_T = 1.1 * T_c
    root_func = lambda T_val: f(series_cutoff, delta, alpha, T_val, T_c) \
            - np.log(T_c / T_val) * delta
    current_T = root_scalar(root_func, x0=last_T, x1=current_T, maxiter=int(1e3))

    return current_T.root


def iterative_gap2(T, T_c, alpha):
    def f(series_cutoff, delta, alpha, T, T_c):
        series_diff = np.inf
        series_sum = 0.
        i = 0
        while series_diff > series_cutoff:
            omega = (2. * i + 1.) * T * np.pi
            #x = matsubara_usadel_x(omega, curr_delta, alpha)
            #sin_term = -0.5j * (x + 1. / x)
            #term = (curr_delta / omega - np.real(sin_term))
            term = iterative_gap_matsubara_arg(omega, delta, alpha)
            series_sum += term
            series_diff = np.abs(term)
            i += 1
        new_delta = np.real(series_sum * 2. * np.pi * T)
        return new_delta

    #alpha = T_c * eta / 1.76
    cutoff = 1e-6
    series_cutoff = 5e-8
    min_T = 0.02 * T_c
    if T < min_T:
        T = min_T
        #return Delta_0
    curr_delta = T_c * 1.76
    last_delta = 2.1 * T_c
    a = np.log(T / T_c)
    root_fn = lambda delta: f(series_cutoff, delta, alpha, T, T_c) \
            + a * delta
    curr_delta = root_scalar(root_fn, x0=last_delta, x1=curr_delta, bracket=(1e-3, 2 * 1.76 * T_c), maxiter=int(1e3))

    return curr_delta.root


def calc_delta_int(T_c, alpha, T, N_V):
    def sine_theta(E, delta):
        x = get_usadel_x2(E, delta, alpha)
        return (x * x - 1 ) / (2.j * x)
    debeye_temp = 0.5 * 1.76 * T_c * np.exp(1. / N_V)
    if T / T_c < 1e-2:
        temp_depend = lambda E: 0.
    else:
        beta = 1. / T
        temp_depend = lambda E: fermi(E, 0., beta)
    def disorder_root(delta):
        integrand = lambda E: np.imag(sine_theta(E, delta)) * (1. -
                2. * temp_depend(E))
        integral_val, _ = quad(integrand, 0., debeye_temp)
        return delta - N_V * integral_val
    delta_0 = root_scalar(disorder_root, x0=1.76*T_c, x1=1.6 * T_c, maxiter=int(1e4))

    return delta_0.root


def calc_NV(T_c, alpha):
    def root_fn(NV):
        if NV < 1e-4:
            debeye_temp = np.inf
        else:
            debeye_temp = 0.5 * 1.76 * T_c * np.exp(1. / NV)
        delta_0 = 1.76 * T_c
        f = lambda E: E / np.sqrt(E * E + delta_0 * delta_0)
        term1, _ = quad(f, 0., debeye_temp, ) #fine tune integration
        return delta_0 - NV * term1

    N_V = root_scalar(root_fn, x0=0.1, x1=0.2, maxiter=int(1e4))
    N_V = N_V.root

    return N_V


def calc_T_from_delta_int(T_c, alpha, delta, N_V):
    def sine_theta(E, delta):
        x = get_usadel_x2(E, delta, alpha)
        return (x * x - 1 ) / (2.j * x)
    debeye_temp = 0.5 * 1.76 * T_c * np.exp(1. / N_V)
    def disorder_root(T):
        if T / T_c < 1e-2:
            temp_depend = lambda E: 0.
        else:
            beta = 1. / T
            temp_depend = lambda E: fermi(E, 0., beta)
        integrand = lambda E: np.imag(sine_theta(E, delta)) * (1. -
                2. * temp_depend(E))
        integral_val, _ = quad(integrand, 0., debeye_temp)
        return delta - N_V * integral_val
    T_sol = root_scalar(disorder_root, x0=0., x1=0.1 * T_c, maxiter=int(1e4))

    return T_sol.root


def calc_real_Tc(T_c_orig, alpha, res):
    NV = calc_NV(T_c_orig, alpha)
    delta_0 = calc_delta_int(T_c_orig, alpha, 0., NV)
    last_d = 0.1 * delta_0
    last_T = calc_T_from_delta_int(T_c_orig, alpha, last_d, NV)
    curr_d = 0.05 * delta_0
    curr_T = calc_T_from_delta_int(T_c_orig, alpha, curr_d, NV)
    estimate = curr_T + curr_d * (curr_T - last_T) / (last_d - curr_d)
    while estimate - curr_T > res:
        print(estimate)
        last_d = curr_d
        last_T = curr_T
        curr_d = 0.2 * last_d
        curr_T = calc_T_from_delta_int(T_c_orig, alpha, curr_d, NV)
        estimate = curr_T + curr_d * (curr_T - last_T) / (last_d - curr_d)
    return estimate


def invert_eff_Tc(alpha_start, alpha_stop, delta):
    N_steps = int(np.ceil((alpha_stop - alpha_start) / delta))
    alpha_vals = np.linspace(alpha_start, alpha_stop, N_steps)
    results = np.ndarray(alpha_vals.shape)
    delta_0 = np.ndarray(alpha_vals.shape)
    for i, a in enumerate(alpha_vals):
        NV = calc_NV(1., a)
        delta_0[i] = calc_delta_int(1., a, 0., NV)
        results[i] = calc_real_Tc(1., a, 1e-5)
    print(delta_0)
    invert = interpolate.interp1d(alpha_vals / delta_0, results)
    alpha_revert = interpolate.interp1d(alpha_vals / delta_0, alpha_vals)
    def inversion(T_c_eff, alpha_eff):
        alpha_ratio = alpha_eff / T_c_eff
        T_c = T_c_eff / invert(alpha_ratio)
        alpha = alpha_revert(alpha_ratio) * T_c
        return T_c, alpha

    return inversion


def map_fn(args):
    return args[0], args[1], calc_delta_int(1., args[2], args[3], args[4])


def precalc_temp_delta(T_start, T_stop, alpha_start, alpha_stop, step_size):
    N_steps_a = int(np.ceil((alpha_stop -  alpha_start) / step_size))
    N_steps_T = int(np.ceil((T_stop - T_start) / step_size))
    T_vals = np.linspace(T_start, T_stop, N_steps_T)
    alpha_vals = np.linspace(alpha_start, alpha_stop, N_steps_a)
    results = np.ndarray((N_steps_a, N_steps_T))
    args = []
    for i, alpha in enumerate(alpha_vals):
        NV = calc_NV(1., alpha)
        print(i)
        for j, T in enumerate(T_vals):
            args.append((i, j, alpha, T, NV))
            #results[i][j] = calc_delta_int(1., alpha, T, NV)
    pool = mp.Pool()
    map_results = pool.map(map_fn, args)
    for data in map_results:
        results[data[0]][data[1]] = data[2]
    return interpolate.interp2d(T_vals, alpha_vals, results)


def generate_sigma2_fn(T_start, T_stop, alpha_start, alpha_stop, step_size):
    inverter = invert_eff_Tc(alpha_start, alpha_stop, step_size)
    delta_fn = precalc_temp_delta(T_start, T_stop, alpha_start, alpha_stop,
            step_size)
    def ret_fn(T, T_c, h_nu, alpha):
        real_Tc, real_alpha = inverter(T_c, alpha)
        delta = real_Tc * delta_fn(T / real_Tc, real_alpha / real_Tc)[0]
        return calc_sigma2_thermal(T, delta, h_nu, real_alpha)

    return ret_fn


def calc_Qi(T_vals, T_c, alpha, h_nu):
    inverter = invert_eff_Tc(0.001, 0.9, 1e-2)
    real_Tc, real_alpha = inverter(T_c, alpha / T_c)
    NV = calc_NV(real_Tc, real_alpha)
    results = np.ndarray(len(T_vals))
    for i, T in enumerate(T_vals):
        delta = calc_delta_int(real_Tc, real_alpha, T, NV)
        results[i] = np.abs(calc_sigma2_thermal(T, delta, h_nu, alpha
                ) / calc_sigma1_thermal(T, delta, h_nu, alpha))
        #results[i] = calc_sigma1_thermal(T, delta, h_nu, alpha)

    return results


def N_qp(T, Delta, alpha, N_0):
    def integrand(E):
        x_val = get_usadel_x2(E, Delta, alpha)
        x_sq_mag = np.real(x_val * np.conj(x_val))
        cos_val = np.real(x_val * x_sq_mag + x_val) / x_sq_mag
        pop_level = 1. - np.tanh(0.5 * E / T)
        return cos_val * pop_level

    E_g = analytical_Eg(Delta, alpha)
    N_int, _ = quad(integrand, E_g, np.inf)

    return N_0 * N_int


def S2_calc(T, Delta, alpha, N_0, h_nu):
    n_thermal = N_qp(T, Delta, alpha, N_0)
    S2_prefix = 0.5 * calc_sigma2_thermal(T, Delta, h_nu, alpha) / (
            calc_sigma2_thermal(0., Delta, h_nu, alpha) * n_thermal)

    return S2_prefix


def GR_Sxx(T, Delta, alpha, N_0, h_nu, tau, tau_max, V):
    #Delta = calc_delta_int(T_c, alpha, T, N_V)
    #Delta_0 = calc_delta_int(T_c, alpha, 0., N_V)
    n_thermal = N_qp(T, Delta, alpha, N_0)
    if n_thermal == 0.:
        return 0.
    sigma2_0 = calc_sigma2_thermal(0., Delta, h_nu, alpha)
    S2_prefix = 0.5 * (calc_sigma2_thermal(T, Delta, h_nu, alpha) - sigma2_0) \
            / (sigma2_0 * n_thermal)
    fluctuation_term = 4. * tau * (1. + tau / tau_max) * n_thermal / V

    return S2_prefix * S2_prefix * fluctuation_term


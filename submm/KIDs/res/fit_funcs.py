from typing import NamedTuple, Union, Optional, Callable

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

from submm.KIDs.res.utils import cardan, derived_text, calc_qc_qi


# at import time, create a dictionary of all the fitting functions in this module
fitting_functions = {}


def fit_func(func):
    """Decorator to add a function to the fitting_functions dictionary"""
    name_type = func.__name__
    fitting_functions[name_type] = func
    return func


@fit_func
@jit(nopython=True)
def nonlinear_mag(f_hz, fr, Qr, amp, phi, a, b0, b1, flin):
    """
    function to describe the magnitude S21 of a non-linear resonator

    This is based of fitting code from MUSIC
    The idea is we are producing a model that is described by the equation below
    the first two terms in the large parenthesis and all other terms are familiar to me,
    but I am not sure where the last term comes from though it does seem to be important for fitting

                              /        (j phi)            (j phi)    |  2
    |S21|^2 = (b0+b1 x_lin)* |1 -amp*e^           +amp*(e^       -1) |^
                             |   ------------      ----              |
                             \     (1+ 2jy)         2               /

        where the nonlinearity of y is described by the following equation taken from Response of superconducting
        micro-resonators with nonlinear kinetic inductance:
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr

    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers in Hertz
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    a : float
        The nonlinearity parameter, bifurcation occurs at a = 0.77
    b0 : float
        DC level of s21 away from resonator
    b1 : float
        Frequency dependant gain variation
    flin : float
        This is probably the frequency of the resonator when a = 0

    Returns
    -------
    ResFit
    """
    xlin = (f_hz - flin) / flin
    xg = (f_hz - fr) / fr
    yg = Qr * xg
    y = np.zeros(f_hz.shape[0])
    # find the roots of the y equation above
    for i in range(0, f_hz.shape[0]):
        """
        4y^3+ -4yg*y^2+ y -(yg+a)
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        roots = cardan(4.0,-4.0*yg[i],1.0,-(yg[i]+a))
        print(roots)
        # more accurate version that doesn't seem to change the fit at all
        # only cares about real roots
        roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))       
        where_real = np.where(np.imag(roots) == 0)
        # analytic version has some floating point error accumulation
        where_real = np.where(np.abs(np.imag(roots)) < 1e-10) 
        """
        # np.max(np.real(roots[where_real]))
        y[i] = cardan(4.0, -4.0 * yg[i], 1.0, -(yg[i] + a))
    abs_val = np.abs(1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * y) + amp / 2. * (np.exp(1.0j * phi) - 1.0))
    z = (b0 + b1 * xlin) * abs_val ** 2
    return z


@fit_func
@jit(nopython=True)
def linear_mag(f_hz, fr, Qr, amp, phi, b0):
    """
    This is based of fitting code from MUSIC
    The idea is we are producing a model that is described by the equation below
    the first two terms in the large parenthesis and all other terms are familiar to me,
    but I am not sure where the last term comes from though it does seem to be important for fitting

                     /        (j phi)            (j phi)    |  2
    |S21|^2 = (b0)* |1 -amp*e^           +amp*(e^       -1) |^
                    |   ------------      ----              |
                    \     (1+ 2jxg)         2              /

        no y just xg, with no non-linear kinetic inductance

    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    b0 : float
        DC level of s21 away from resonator

    """
    if not np.isscalar(fr):  # vectorize breaks numba though
        f_hz = np.reshape(f_hz, (f_hz.shape[0], 1, 1, 1, 1, 1))
    xg = (f_hz - fr) / fr
    z = (b0) * np.abs(
        1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * xg * Qr) + amp / 2. * (np.exp(1.0j * phi) - 1.0)) ** 2
    return z


@fit_func
@jit(nopython=True)
def nonlinear_iq(f_hz, fr, Qr, amp, phi, a, i0, q0, tau, f0):
    """
    To describe the I-Q loop of a nonlinear resonator

    This is based of fitting code from MUSIC

    The idea is we are producing a model that is described by the equation below
    the first two terms in the big parenthesis and all other terms are familiar to me,
    but I am not sure where the last term comes from though it does seem to be important for fitting

                       (-j 2 pi deltaf tau)  /        (j phi)            (j phi)   |
           (i0+j*q0)*e^                    *|1 -amp*e^           +amp*(e^       -1) |
                                            |   ------------      ----              |
                                             \     (1+ 2jy)         2              /

        where the nonlinearity of y is described by the following eqution taken from Response of superconducting
        micro resonators with nonlinear kinetic inductance:
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr

    Parameters
    ----------
    f_hz : numpy.array
        The frequencies in your iq sweep covers
    fr : float
        The center frequency of the resonator
    Qr : float
       The quality factor of the resonator
    amp : float
        Amplitude as Qr/Qc
    phi : float
        The rotation parameter for an impedance mismatch between the resonator and the readout system
    a : float
        The nonlinearity parameter, bifurcation occurs at a = 0.77
    i0 : float
    q0 : float
        these are constants that describes an overall phase rotation of the iq loop + a DC gain offset
    tau : float
        The cable delay
    f0 : float
        The center frequency, not sure why we include this as a secondary parameter should be the same as fr
    """
    deltaf = (f_hz - f0)
    xg = (f_hz - fr) / fr
    yg = Qr * xg
    y = np.zeros(f_hz.shape[0])
    # find the roots of the y equation above
    for i in range(0, f_hz.shape[0]):
        """
        4y^3+ -4yg*y^2+ y -(yg+a)
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        # more accurate version that doesn't seem to change the fit at al     
        roots = np.roots((16.,-16.*yg[i],8.,-8.*yg[i]+4*a*yg[i]/Qr-4*a,1.,-yg[i]+a*yg[i]/Qr-a+a**2/Qr))
        only care about real roots
        where_real = np.where(np.imag(roots) == 0)
        y[i] = np.max(np.real(roots[where_real]))
        """
        y[i] = cardan(4.0, -4.0 * yg[i], 1.0, -(yg[i] + a))
    big_parenthesis = (1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * y) + amp / 2. * (np.exp(1.0j * phi) - 1.0))
    z = (i0 + 1.j * q0) * np.exp(-1.0j * 2 * np.pi * deltaf * tau) * big_parenthesis
    return z


@fit_func
@jit(nopython=True)
def nonlinear_iq_for_fitter(f_hz, fr, Qr, amp, phi, a, i0, q0, tau, f0):
    """
    when using a fitter that can't handel complex number
    one needs to return both the real and imaginary components separately
    """

    deltaf = (f_hz - f0)
    xg = (f_hz - fr) / fr
    yg = Qr * xg
    y = np.zeros(f_hz.shape[0])

    for i in range(0, f_hz.shape[0]):
        """
        roots = np.roots((4.0,-4.0*yg[i],1.0,-(yg[i]+a)))
        where_real = np.where(np.imag(roots) == 0)
        y[i] = np.max(np.real(roots[where_real]))
        """
        y[i] = cardan(4.0, -4.0 * yg[i], 1.0, -(yg[i] + a))
    z = (i0 + 1.j * q0) * np.exp(-1.0j * 2 * np.pi * deltaf * tau) * (
            1.0 - amp * np.exp(1.0j * phi) / (1.0 + 2.0 * 1.0j * y) + amp / 2. * (np.exp(1.0j * phi) - 1.0))
    real_z = np.real(z)
    imag_z = np.imag(z)
    return np.hstack((real_z, imag_z))



""" All the types
    fit_types: tuple
    fr: Optional[float] = None
    Qr: Optional[float] = None
    amp: Optional[float] = None
    phi: Optional[float] = None
    a: Optional[float] = None
    b0: Optional[float] = None
    b1: Optional[float] = None
    i0: Optional[float] = None
    q0: Optional[float] = None
    tau: Optional[float] = None
    f0: Optional[float] = None
    Qi: Optional[float] = None
    Qc: Optional[float] = None
    popt: Optional[np.ndarray] = None
    pcov: Optional[np.ndarray] = None
    f_data: Optional[np.ndarray] = None
    z_data: Optional[np.ndarray] = None
"""

derived_params = {'qi', 'qc'}
field_to_first_format_int = {'fr': 5, 'qr': 7, 'amp': 1, 'phi': 2, 'a': 1, 'b0': 2, 'b1': 2, 'i0': 1, 'q0': 1, 'tau': 6,
                             'f0': 5, 'qi': 7, 'qc': 7}
field_to_decimal_format_int = {'fr': 4, 'qr': 0, 'amp': 2, 'phi': 2, 'a': 2, 'b0': 2, 'b1': 2, 'i0': 2, 'q0': 2,
                               'tau': 2, 'f0': 4, 'qi': 0, 'qc': 0}
field_to_format_letter = {'fr': 'f', 'qr': 'f', 'amp': 'f', 'phi': 'f', 'a': 'f', 'b0': 'E', 'b1': 'E', 'i0': 'E',
                          'q0': 'E', 'tau': 'f', 'f0': 'f', 'qi': 'f', 'qc': 'f'}
filed_to_field_label = {'fr': 'fr (MHz)', 'qr': 'Qr', 'amp': 'amp', 'phi': 'phi', 'a': 'a', 'b0': 'b0', 'b1': 'b1',
                        'i0': 'i0', 'q0': 'q0', 'tau': 'tau (ns)', 'f0': 'f0 (MHz)', 'qi': 'Qi', 'qc': 'Qc'}
field_to_multiplier = {'fr': 1.0e-6, 'tau': 1.0e9, 'f0': 1.0e-6}

field_to_format_strs = {}
field_to_text_len = {}
for a_field in field_to_first_format_int.keys():
    first_format_int = field_to_first_format_int[a_field]
    decimal_format_int = field_to_decimal_format_int[a_field]
    format_letter = field_to_format_letter[a_field]
    field_to_format_strs[a_field] = f'{first_format_int}.{decimal_format_int}{format_letter}'
    # the + 1 is to account for negative numbers with the extra '-'
    text_len = first_format_int + decimal_format_int + 1
    if decimal_format_int > 0:
        # add a space for the decimal point
        text_len += 1
    if format_letter.lower() == 'e':
        text_len += 4
    field_to_text_len[a_field] = max(text_len, len(filed_to_field_label[a_field]))


def format_field_value(field, value):
    field = field.lower().strip()
    if field in field_to_multiplier.keys():
        value *= field_to_multiplier[field]
    number_str = value.__format__(field_to_format_strs[field])
    return number_str.center(field_to_text_len[field])


def calc_left_right_spaces(text, total_space_available):
    if len(text) > total_space_available:
        return '', '', ' ' * (len(text) - total_space_available)
    else:
        right_space = (total_space_available - len(text)) // 2
        left_space = total_space_available - right_space - len(text)
        return ' ' * left_space, ' ' * right_space, ''


class NonlinearIQResBase(NamedTuple):
    """ The resonator parameters for a NonLinear IQ fit (nonlinear_iq_for_fitter)"""
    fr: Optional[float] = None
    Qr: Optional[float] = None
    amp: Optional[float] = None
    phi: Optional[float] = None
    a: Optional[float] = None
    i0: Optional[float] = None
    q0: Optional[float] = None
    tau: Optional[float] = None
    f0: Optional[float] = None
    Qi: Optional[float] = None
    Qc: Optional[float] = None


class Res(NamedTuple):
    """
    base class for resonator parameters then a determined by a single fit
    Needs to be mixed in with a typing.NamedTuple
    """

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        elif isinstance(item, int):
            return self._asdict()[self._fields[item]]
        elif isinstance(item, slice):
            return tuple([self._asdict()[field] for field in self._fields[item]])
        else:
            raise TypeError(f"ResFit indices must be str, int, or slice, not {type(item)}")

    def header_items(self):
        for field in self._fields:
            yield field

    def input_items(self):
        for field in self.header_items():
            if field.lower() not in derived_params:
                yield field

    def __str__(self):
        return_str = ''
        for field in self.header_items():
            return_str += f"{self.__getattribute__(field)},"
        return return_str[:-1]

    def console(self, label: str = None, print_header: bool = True, fields=None):
        if label is None:
            label = ''
        if fields is None:
            found_fields = {}
            # non-derived parameters
            for field in self.input_items():
                value = self.__getattribute__(field)
                if value is not None:
                    found_fields[field.lower()] = value
            # derived parameters
            found_derived = {}
            Qi = self.Qi
            Qc = self.Qc
            if (Qc is None or Qi is None) and 'qr' in found_fields.keys() and 'amp' in found_fields.keys():
                Qc, Qi = calc_qc_qi(qr=self.Qr, amp=self.amp)
            if Qi is not None:
                found_derived['qi'] = Qi
            if Qc is not None:
                found_derived['qc'] = Qc

        else:
            found_fields = {}
            found_derived = {}
            for field in fields:
                if field in derived_params:
                    found_derived[field.lower()] = self.__getattribute__(field)
                else:
                    found_fields[field.lower()] = self.__getattribute__(field)
        # determine console output
        len_label = max(5, len(label))
        label_spaces = ' ' * len_label
        total_len_var_fit = sum([field_to_text_len[field] for field in found_fields.keys()])
        total_len_var_fit += len(found_fields.keys()) - 1
        var_fit_label = ' Variables fit '
        var_fit_left_space, var_fit_right_space, var_fit_extra_header = \
            calc_left_right_spaces(text=var_fit_label, total_space_available=total_len_var_fit)
        derived_label = ' Derived variables '
        total_len_derived = sum([field_to_text_len[field] for field in found_derived.keys()])
        total_len_derived += len(found_derived.keys()) - 1
        derived_fit_left_space, derived_fit_right_space, derived_fit_extra_header = \
            calc_left_right_spaces(text=derived_label, total_space_available=total_len_derived)
        # header only console output
        if print_header:
            if self.fr is None:
                fr_mhz = '(Frequency not found)'
            else:
                fr_mhz = f'{self.fr / 1e6:.2f} MHz'
            print(f'Resonator at {fr_mhz} MHz" % (vals[0] * 1.0e-6)')
            super_header = f'{label_spaces}|{var_fit_left_space}{var_fit_label}{var_fit_right_space}'
            sub_header = f'{label_spaces}|'
            first_field = True
            for field in found_fields.keys():
                field_len = field_to_text_len[field]
                formatted_field = filed_to_field_label[field].center(field_len)
                if first_field:
                    sub_header += f'{formatted_field}'
                    first_field = False
                else:
                    sub_header += f'|{formatted_field}'
            sub_header += var_fit_extra_header
            if found_derived:
                super_header += '|' + derived_text(f'{derived_fit_left_space}{derived_label}{derived_fit_right_space}')
                sub_header += '|'
                first_field = True
                for field in found_derived.keys():
                    field_len = field_to_text_len[field]
                    formatted_field = filed_to_field_label[field].center(field_len)
                    if first_field:
                        sub_header += derived_text(f'{formatted_field}')
                        first_field = False
                    else:
                        sub_header += derived_text(f'|{formatted_field}')

                if derived_fit_extra_header != '':
                    sub_header += derived_text(derived_fit_extra_header)
            print(super_header)
            print(sub_header)
        # values output for console
        values_str = label.__format__(f'<{len_label}')
        for field in found_fields.keys():
            values_str += '|' + format_field_value(field, found_fields[field])
        values_str += var_fit_extra_header
        if found_derived:
            first_field = True
            values_str += '|'
            for field in found_derived.keys():
                if first_field:
                    values_str += derived_text(format_field_value(field, found_derived[field]))
                    first_field = False
                else:
                    values_str += derived_text('|' + format_field_value(field, found_derived[field]))
            if derived_fit_extra_header != '':
                values_str += derived_text(derived_fit_extra_header)
        print(values_str)


class NonlinearIQRes(NonlinearIQResBase, Res):
    pass


class Fit(NamedTuple):
    origin: str
    func: Callable
    guess: Optional[Res] = None
    result:  Optional[Res] = None
    pcov: Optional[np.ndarray] = None
    f_data: Optional[np.ndarray] = None
    z_data: Optional[np.ndarray] = None
    red_chi_sqr: Optional[float] = None

    """ Base class for a resonator fit results """
    def z_fit(self) -> np.ndarray:
        """Return the complex impedance of the fit."""
        return self.func(self.f_data, *self.result[0:-2])

    def z_guess(self) -> np.ndarray:
        """Return the complex impedance of the guess."""
        return self.func(self.f_data, *self.guess[0:-2])

    def plot(self, show=True):
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.plot(np.real(self.z_data), np.imag(self.z_data), 'o', label="Input Data")
        z_guess = self.z_guess()
        plt.plot(np.real(z_guess), np.imag(z_guess), label="Initial Guess")
        z_fit = self.z_fit()
        plt.plot(np.real(z_fit), np.imag(z_fit), label="Fit Results")
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.legend()
        if show:
            plt.show()

    def console(self, print_header=True, fields=None):
        if self.guess is None:
            if self.result is None:
                print('No fit results')
            else:
                self.result.console(label='Fit', print_header=print_header, fields=fields)
        else:
            self.guess.console(label='Guess', print_header=print_header, fields=fields)
            self.result.console(label='Fit', print_header=False, fields=fields)







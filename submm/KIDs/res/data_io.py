"""
Automated data storage and inspections tools for the various fitting functions
"""

import os
from operator import attrgetter
from typing import NamedTuple, Optional, Callable, Sequence

import toml
import numpy as np
import matplotlib.pyplot as plt

from submm.sample_data.abs_paths import abs_path_output_dir_default
from submm.KIDs.res.utils import derived_text, filename_text, write_text, calc_qc_qi, line_format
from submm.KIDs.res.sweep_tools import InteractivePlot


derived_params = {'qi', 'qc', 'red_chi_sqr'}
field_to_first_format_int = {'fr': 5, 'qr': 7, 'amp': 1, 'phi': 2, 'a': 1, 'b0': 2, 'b1': 2, 'i0': 1, 'q0': 1, 'tau': 6,
                             'f0': 5, 'qi': 7, 'qc': 7, 'flin': 5}
field_to_decimal_format_int = {'fr': 4, 'qr': 0, 'amp': 2, 'phi': 2, 'a': 2, 'b0': 2, 'b1': 2, 'i0': 2, 'q0': 2,
                               'tau': 2, 'f0': 4, 'qi': 0, 'qc': 0, 'flin': 4}
field_to_format_letter = {'fr': 'f', 'qr': 'f', 'amp': 'f', 'phi': 'f', 'a': 'f', 'b0': 'E', 'b1': 'E', 'i0': 'E',
                          'q0': 'E', 'tau': 'f', 'f0': 'f', 'qi': 'f', 'qc': 'f', 'flin': 'f'}
field_to_field_label = {'fr': 'fr (MHz)', 'qr': 'Qr', 'amp': 'amp', 'phi': 'phi', 'a': 'a', 'b0': 'b0', 'b1': 'b1',
                        'i0': 'i0', 'q0': 'q0', 'tau': 'tau (ns)', 'f0': 'f0 (MHz)', 'qi': 'Qi', 'qc': 'Qc','flin': 'flin (MHz)'}
field_to_multiplier = {'fr': 1.0e-6, 'tau': 1.0e9, 'f0': 1.0e-6, 'flin': 1.0e-6}

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
    field_to_text_len[a_field] = max(text_len, len(field_to_field_label[a_field]))


def format_field_value(field, value):
    field = field.lower().strip()
    if field in field_to_multiplier.keys():
        value *= field_to_multiplier[field]
    number_str = value.__format__(field_to_format_strs[field])
    return number_str


def format_and_center(field, value):
    return format_field_value(field, value).center(field_to_text_len[field])


def calc_left_right_spaces(text, total_space_available):
    if len(text) > total_space_available:
        return '', '', ' ' * (len(text) - total_space_available)
    else:
        right_space = (total_space_available - len(text)) // 2
        left_space = total_space_available - right_space - len(text)
        return ' ' * left_space, ' ' * right_space, ''


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

    def header(self):
        header_str = ''
        for field in self.header_items():
            header_str += f'{field},'
        return header_str[:-1]

    def input_items(self):
        for field in self.header_items():
            if field.lower() not in derived_params:
                yield field

    def __str__(self):
        return_str = ''
        for field in self.header_items():
            return_str += f"{self.__getattribute__(field)},"
        return return_str[:-1]

    def __call__(self, *args):
        if not args:
            args = self._fields
        values_list = []
        for arg in args:
            if arg in self._fields:
                value = self.__getattribute__(arg)
            else:
                value = None
            if value is None:
                value = float('nan')
            values_list.append(value)
        return tuple(values_list)

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
                if field.lower() in derived_params:
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
            print(f'Resonator at {fr_mhz}')
            super_header = f'{label_spaces}|{var_fit_left_space}{var_fit_label}{var_fit_right_space}'
            sub_header = f'{label_spaces}|'
            first_field = True
            for field in found_fields.keys():
                field_len = field_to_text_len[field]
                formatted_field = field_to_field_label[field].center(field_len)
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
                    formatted_field = field_to_field_label[field].center(field_len)
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
            values_str += '|' + format_and_center(field, found_fields[field])
        values_str += var_fit_extra_header
        if found_derived:
            first_field = True
            values_str += '|'
            for field in found_derived.keys():
                if first_field:
                    values_str += derived_text(format_and_center(field, found_derived[field]))
                    first_field = False
                else:
                    values_str += derived_text('|' + format_and_center(field, found_derived[field]))
            if derived_fit_extra_header != '':
                values_str += derived_text(derived_fit_extra_header)
        print(values_str)


class NonlinearIQResBase(NamedTuple):
    """ Base resonator parameters for a NonLinear IQ fit, nonlinear_iq_for_fitter()"""
    fr: Optional[float] = None
    Qr: Optional[float] = None
    amp: Optional[float] = None
    phi: Optional[float] = None
    a: Optional[float] = None
    i0: Optional[float] = None
    q0: Optional[float] = None
    tau: Optional[float] = None
    f0: Optional[float] = None
    red_chi_sq: Optional[float] = None
    Qi: Optional[float] = None
    Qc: Optional[float] = None


class NonlinearIQRes(NonlinearIQResBase, Res):
    """ The resonator parameters for a NonLinear IQ fit, nonlinear_iq_for_fitter()"""


class NonlinearMagResBase(NamedTuple):
    """ Base resonator parameters for a NonLinear Mag fit, fit_nonlinear_mag()"""
    fr: Optional[float] = None
    Qr: Optional[float] = None
    amp: Optional[float] = None
    phi: Optional[float] = None
    a: Optional[float] = None
    b0: Optional[float] = None
    b1: Optional[float] = None
    flin: Optional[float] = None
    red_chi_sq: Optional[float] = None
    Qi: Optional[float] = None
    Qc: Optional[float] = None


class NonlinearMagRes(NonlinearMagResBase, Res):
    """ The resonator parameters for a NonLinear Mag fit, fit_nonlinear_mag()"""


class LinearMagResBase(NamedTuple):
    """ Base resonator parameters for a Linear Mag fit, fit_linear_mag(), linear_mag()"""
    fr: Optional[float] = None
    Qr: Optional[float] = None
    amp: Optional[float] = None
    phi: Optional[float] = None
    b0: Optional[float] = None
    red_chi_sq: Optional[float] = None
    Qi: Optional[float] = None
    Qc: Optional[float] = None


class LinearMagRes(LinearMagResBase, Res):
    """ The resonator parameters for a Linear Mag fit, fit_linear_mag(), linear_mag()"""


class Fit(NamedTuple):
    origin: str
    func: Callable
    guess: Optional[Res] = None
    result:  Optional[Res] = None
    popt: Optional[np.ndarray] = None
    pcov: Optional[np.ndarray] = None
    f_data: Optional[np.ndarray] = None
    z_data: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def __getitem__(self, item):
        if isinstance(item, str):
            # this could be a field of the result
            if item in self._fields:
                return getattr(self, item)
            elif item in self.result._fields:
                return getattr(self.result, item)
            elif item.lower() == 'fit':
                return self.popt, self.pcov
            elif item.lower() == 'fit_result':
                return self.z_fit()
            elif item.lower() == 'x0_result':
                return self.z_guess()
            elif item.lower() == 'x0':
                return self.guess
            elif item.lower() == 'z':
                return self.z_data
            else:
                raise KeyError(f'Unknown string key: {item}')
        elif isinstance(item, int):
            return self._asdict()[self._fields[item]]
        elif isinstance(item, slice):
            return tuple([self._asdict()[field] for field in self._fields[item]])
        else:
            raise TypeError(f"ResFit indices must be str, int, or slice, not {type(item)}")

    def __call__(self, *args):
        if not args:
            args = self._fields
        values_list = []
        for arg in args:
            if arg in self._fields:
                value = self.__getattribute__(arg)
            else:
                value = None
            if value is None:
                value = float('nan')
            values_list.append(value)
        return tuple(values_list)

    def __repr__(self):
        return f"Fit-{str(self.result).replace(',', '-')}"

    """ Base class for a resonator fit results """
    def z_fit(self) -> np.ndarray:
        """Return the complex impedance of the fit."""
        return self.func(self.f_data, *self.result[0:-len(derived_params)])

    def z_guess(self) -> np.ndarray:
        """Return the complex impedance of the guess."""
        return self.func(self.f_data, *self.guess[0:-len(derived_params)])

    def plot(self, show=True):
        plt.figure()
        z_guess = self.z_guess()
        z_fit = self.z_fit()
        if "mag" in self.origin:
            plt.plot(self.f_data/10**6,20*np.log10(np.abs(self.z_data)), 'o', label="Input Data")
            plt.plot(self.f_data/10**6,20*np.log10(np.abs(z_guess)), '-', label="Initial Guess")
            plt.plot(self.f_data/10**6,20*np.log10(np.abs(z_fit)), '-', label="Fit Results")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Power (dB)")
            plt.legend()
        else:
            plt.axes().set_aspect('equal')
            plt.plot(np.real(self.z_data), np.imag(self.z_data), 'o', label="Input Data")
            plt.plot(np.real(z_guess), np.imag(z_guess), label="Initial Guess")
            plt.plot(np.real(z_fit), np.imag(z_fit), label="Fit Results")
            plt.xlabel('I')
            plt.ylabel('Q')
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


class ResSet:
    def __init__(self, path: str = None, res_results: Sequence[Res] = None, res_fits: Sequence[Fit] = None,
                 verbose: bool = True):
        self.verbose = verbose
        if path is None:
            # use the default output directory, make it if it doesn't exist
            if not os.path.exists(abs_path_output_dir_default):
                os.mkdir(abs_path_output_dir_default)
            res_set_dir = os.path.join(abs_path_output_dir_default, 'res_set')
            if not os.path.exists(res_set_dir):
                os.mkdir(res_set_dir)
            # use a default file name buy do not overwrite an existing file
            count = 0
            base_path = os.path.join(res_set_dir, f'results{count:02}')
            path_test_csv = f'{base_path}.csv'
            path_test_toml = f'{base_path}.toml'
            # loop until we find a pair of file names that do not exist
            while os.path.exists(path_test_csv) and os.path.exists(path_test_toml):
                count += 1
                base_path = os.path.join(res_set_dir, f'results{count:02}')
                path_test_csv = f'{base_path}.csv'
                path_test_toml = f'{base_path}.toml'
            self.res_set_dir = res_set_dir
            self.path = base_path
        else:
            self.res_set_dir, basename = os.path.split(path)
            if '.' in basename:
                basename, ext = basename.rsplit('.', 1)

            # use the user specified path
            self.path = os.path.join(self.res_set_dir, basename)
            # make a new directory if it doesn't exist
            if not os.path.exists(self.res_set_dir):
                os.mkdir(self.res_set_dir)

        # set in self.read(), self.add_results(), self.add_res_fits methods
        self._results = set()
        # set only in the self.add_res_fits method
        self._fit_results = {}
        # set in self.order_and_validate() method
        self._fields_results = None
        self._list_results = None
        self._list_fit_results = None
        self._structured_array_results = None

        if res_results is None and res_fits is None:
            if self.verbose:
                print(f'No fits on initialization, triggering a read in of the the results.')
            self.read()
        else:
            if res_results is not None:
                self.add_results(res_results)
            if res_fits is not None:
                self.add_res_fits(res_fits)

    def __iter__(self):
        # iterate over the results in order of resonator frequency
        for result in sorted(self._results, key=attrgetter('fr')):
            yield result

    def __len__(self):
        return len(self._results)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._list_results[item]
        elif isinstance(item, slice):
            return self._list_results[item]
        elif isinstance(item, str):
            if item in self._fields_results:
                return self._structured_array_results[item]
            elif item in Fit._fields:
                return self._structured_array_fit_results[item]
            else:
                raise KeyError(f'Unknown string key: {item}')
        else:
            raise TypeError(f"ResSet indices must be str, int, or slice, not {type(item)}")

    def __call__(self, *args) -> dict:
        if not args:
            # when no args are specified
            args = list(self._fields_results) + list(Fit._fields)
        results_fields = []
        fits_fields = []
        for arg in args:
            if arg in self._fields_results:
                results_fields.append(arg)
            elif arg in Fit._fields:
                fits_fields.append(arg)
        results_dict = {arg: self._structured_array_results[arg] for arg in results_fields}
        if fits_fields:
            fits_dict = {arg: [] for arg in fits_fields}
            for fit in self._list_fit_results:
                for arg in fits_fields:
                    if fit is None:
                        fits_dict[arg].append(None)
                    else:
                        fits_dict[arg].append(fit.__getattribute__(arg))
            results_dict.update(fits_dict)
        return results_dict

    def order_and_validate(self):
        # make list data objects
        self._list_results = list(self)
        # get the fields list from the first item
        self._fields_results = self._list_results[0]._fields
        # get the fit results list, which can have None values
        self._list_fit_results = []
        for result in self._list_results:
            if result in self._fit_results:
                self._list_fit_results.append(self._fit_results[result])
            else:
                self._list_fit_results.append(None)
        # make the numpy structured array data object
        values_results = [result() for result in self._list_results]
        columns_values_results = [np.array(column_values) for column_values in zip(*values_results)]
        dtypes_results = [(arg, str(column_array.dtype)) for arg, column_array in zip(self._fields_results,
                                                                                      columns_values_results)]
        self._structured_array_results = np.array(values_results, dtype=dtypes_results)

    def read(self, res_tuple=NonlinearIQRes):
        read_file_name_toml = f'{self.path}.toml'
        read_file_name_csv = f'{self.path}.csv'
        if os.path.exists(read_file_name_toml):
            self.read_toml(read_file_name=read_file_name_toml, res_tuple=res_tuple)
        elif os.path.exists(read_file_name_csv):
            self.read_csv(read_file_name=read_file_name_csv, res_tuple=res_tuple)
        else:
            raise FileNotFoundError(f'No results file found at {self.path} with .toml or .csv extension.')

    def read_toml(self, read_file_name, res_tuple=NonlinearIQRes):
        raise NotImplementedError('reading a toml method is not yet implemented.')
        list_fit_results = toml.load(read_file_name)
        for fit_result in list_fit_results:
            print('here')



    def read_csv(self, read_file_name, res_tuple=NonlinearIQRes):
        """Read in the results from a csv file."""
        if self.verbose:
            print(f'Reading results from file at: {filename_text(self.path)}')
        header_keys = []
        with open(read_file_name, 'r') as f:
            for line in f.readlines():
                if not header_keys:
                    # read the header, see what can be mapped into the NamedTuple specified with res_tuple
                    header_keys_raw = line.strip().split(',')
                    one_key_found = False
                    for header_key in header_keys_raw:
                        if header_key in res_tuple._fields:
                            header_keys.append(header_key)
                            one_key_found = True
                        else:
                            header_keys.append(None)
                    if not one_key_found:
                        raise ValueError(f'No valid header keys found in {self.path} ' +
                                         f'for the header keys {header_keys_raw} and tuple fields: {res_tuple._fields}')
                else:
                    # read the data after a successful header read
                    row_dict = {header_key: value_formatted for header_key, value_formatted
                                in zip(header_keys, line_format(line)) if header_key is not None}
                    self._results.add(res_tuple(**row_dict))
        self.order_and_validate()

    def write(self, csv: bool = True, toml: bool = False):
        if csv:
            self.write_csv()
        if toml:
            raise NotImplementedError('Writing to toml is not yet implemented.')
            self.write_toml()

    def write_csv(self):
        write_file_name = f'{self.path}.csv'
        with open(write_file_name, 'w') as f:
            first_row = True
            for result in self:
                # write the header
                if first_row:
                    f.write(result.header() + '\n')
                    first_row = False
                # write the results
                f.write(str(result) + '\n')
        if self.verbose:
            print(f'{write_text("Results Written")} to file at: {filename_text(self.path)}')

    def write_toml(self):
        write_file_name = f'{self.path}.toml'
        dict_of_fits = {repr(self._fit_results[result]): self._fit_results[result] for result in self._fit_results}
        with open(write_file_name, 'w') as f:
            toml.dump(dict_of_fits, f)
        if self.verbose:
            print(f'{write_text("Results Written")} to file at: {filename_text(self.path)}')

    def add_results(self, res_results: Sequence[Res]):
        self._results.update(res_results)
        self.order_and_validate()
        if self.verbose:
            print(f'Added {len(res_results)} results.')

    def add_res_fits(self, res_fits: Sequence[Fit]):
        for res_fit in res_fits:
            result = res_fit.result
            self._results.add(result)
            self._fit_results[result] = res_fit
        self.order_and_validate()
        if self.verbose:
            print(f'Added {len(res_fits)} results and fit_results.')

    def plot(self,flags = None):
        # plotter want frequencies and z values with shape n_frequency_point x n_res x n_sweeps
        # also for fitted paramters it wants a list fitted names for the legend
        # and a array that is n_res x len(fitted paramters)

        # Make the frequency and z arrays firs

        # Make the frequency and z arrays for original data and fitted
        n_iq_points = len(self._fit_results[next(iter(self))].z_data) # assumes all data is the same size
        frequencies = np.zeros((n_iq_points,len(self)))
        fitted_frequencies = np.zeros((n_iq_points,len(self)))
        z_values = np.zeros((n_iq_points,len(self)),dtype = 'complex')
        fitted_z_values = np.zeros((n_iq_points,len(self)),dtype = 'complex')
        for i,result in enumerate(self): 
            fit_result = self._fit_results[result]
            frequencies[:,i] = fit_result.f_data # input frequencies nominally the same as res_freq_array
            fitted_frequencies[:,i] = fit_result.f_data # input frequencies nominally the same as res_freq_array
            z_values[:,i] = fit_result.z_data
            fitted_z_values[:,i] = fit_result.z_fit()

        # stack the data with fit data
        multi_sweep_freqs = np.dstack((np.expand_dims(frequencies, axis=2), np.expand_dims(fitted_frequencies, axis=2)))
        multi_sweep_z = np.dstack((np.expand_dims(z_values, axis=2), np.expand_dims(fitted_z_values, axis=2)))

        # now get the fitted values
        # first get names of fitted values
        data_names = []
        formats = []
        result = next(iter(self)) # grab first fit for inspection
        for field in result._fields:
            if getattr(result, field) != None:
                data_names.append(field)
                formats.append(field_to_field_label[field.lower()] + ': {:' +field_to_format_strs[field.lower()]+'}')

        fitted_parameters = np.zeros((len(self),len(data_names)))
        for i, result in enumerate(self): # don't forget to sort
            for j, field in enumerate(data_names): # this gets rid of the the None
                if field in field_to_multiplier.keys():
                    fitted_parameters[i,j] = getattr(result,field)*field_to_multiplier[field.lower()]
                else:
                    fitted_parameters[i,j] = getattr(result,field)


        ip = InteractivePlot(multi_sweep_freqs, multi_sweep_z, retune=False, combined_data=fitted_parameters,
                              combined_data_names=data_names,combined_data_format = formats,
                              sweep_labels=['Data', 'Fit'],flags = flags)

        return ip

        


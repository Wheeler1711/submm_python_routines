import os
from setuptools import setup, find_packages

# Example data files
desired_example_file_types = {'csv', 'pkl', 'txt', 'xlsx', 'xls', 'fits'}

local_samples_dir = os.path.join('sample_data')
abs_samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_samples_dir)
# loop over  all the sample data files and get  with the appropriate extension
sample_data_files = []
for file_basename in os.listdir(abs_samples_dir):
    filename_prefix, filename_extension = file_basename.split('.', 1)
    if filename_extension in desired_example_file_types:
        sample_data_files.append(os.path.join(local_samples_dir, file_basename))


setup(name='submm_python_routines',
      version='1.0.0',
      description='Python routines for submm astronomy instrumentation',
      author='Jordan Wheeler',
      author_email='wheeler1711@gmail.com',
      packages=find_packages(),
      url="https://github.com/simonsobs/DetMap",
      data_files=[(local_samples_dir, sample_data_files)],
      install_requires=['PyQt5-Qt5', 'six', 'PyQt5-sip', 'pyparsing', 'pillow', 'numpy', 'llvmlite', 'kiwisolver',
                        'fonttools', 'cycler', 'scipy', 'python-dateutil', 'PyQt5', 'packaging', 'numba', 'matplotlib']
      )

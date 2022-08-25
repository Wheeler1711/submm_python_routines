import os
from setuptools import setup, find_packages

# Example data files
desired_example_file_types = {'csv', 'pkl', 'txt', 'xlsx', 'xls', 'fits', 'mat'}

local_samples_dir = os.path.join('submm', 'sample_data')
# loop over  all the sample data files and get  with the appropriate extension
sample_data_files = []
for file_basename in os.listdir(local_samples_dir):
    try:
        filename_prefix, filename_extension = file_basename.rsplit('.', 1)
    except ValueError:
        # ignore files without an extension
        pass
    else:
        # do this if no ValueError was raised
        if filename_extension in desired_example_file_types:
            sample_data_files.append(os.path.join(local_samples_dir, file_basename))


setup(name='submm',
      version='0.2.1',
      description='Python routines for submm astronomy instrumentation',
      author='Jordan Wheeler',
      author_email='wheeler1711@gmail.com',
      packages=find_packages(),
      url="https://github.com/Wheeler1711/submm_python_routines",
      data_files=[(local_samples_dir, sample_data_files)],
      include_package_data=True,
      python_requires='>3.8',
      install_requires=['numpy<1.23,>=1.18',
                        'matplotlib>=3.5.2',
                        'scipy',
                        'numba',
                        'importlib-metadata']
      )

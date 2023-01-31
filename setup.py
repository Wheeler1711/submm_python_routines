import os
from setuptools import setup, find_packages


def get_file_paths(local_dir, desired_file_types):
    data_files = []
    # loop over all the sample data files and get  with the appropriate extension
    for file_basename in os.listdir(local_dir):
        try:
            filename_prefix, filename_extension = file_basename.rsplit('.', 1)
        except ValueError:
            # ignore files without an extension
            pass
        else:
            # do this if no ValueError was raised
            if filename_extension in desired_file_types:
                data_files.append(os.path.join(local_dir, file_basename))
    return data_files


# Example data files
desired_example_file_types = {'csv', 'pkl', 'txt', 'xlsx', 'xls', 'fits', 'mat'}
local_samples_dir = os.path.join('submm', 'sample_data')
sample_data_files = get_file_paths(local_dir=local_samples_dir, desired_file_types=desired_example_file_types)


# Demo files
desired_demo_file_types = {'ipynb'}
local_demos_dir = os.path.join('submm', 'demo')
demo_files = get_file_paths(local_dir=local_demos_dir, desired_file_types=desired_demo_file_types)


setup(name='submm',
      version='0.3.3',
      description='Python routines for submm astronomy instrumentation',
      author='Jordan Wheeler',
      author_email='wheeler1711@gmail.com',
      packages=find_packages(),
      url="https://github.com/Wheeler1711/submm_python_routines",
      data_files=[(local_samples_dir, sample_data_files),
                  (local_demos_dir, demo_files)],
      include_package_data=True,
      python_requires='>3.7',
      install_requires=['numpy',
                        'matplotlib>=3.5.2',
                        'scipy',
                        'numba',
                        'tqdm',
                        'importlib-metadata',
                        'toml']
      )

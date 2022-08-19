import os
dir_this_file = os.path.dirname(os.path.realpath(__file__))
abs_path_parent_dir, _ = dir_this_file.rsplit("submm_python_routines", 1)
abs_path_submm = os.path.join(abs_path_parent_dir, "submm_python_routines")
print(f'The absolute path for submm_python_routines: {abs_path_submm}')

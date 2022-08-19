import os
dir_this_file = os.path.dirname(os.path.realpath(__file__))
abs_path_parent_dir = dir_this_file.rsplit("submm", 1)[0]
abs_path_submm = os.path.join(abs_path_parent_dir, "submm")
abs_path_sample_data = os.path.join(abs_path_submm, "sample_data")


if __name__ == "__main__":
    print(f'abs_path_parent_dir:               {abs_path_parent_dir}')
    print(f'The absolute path for submm:       {abs_path_submm}')
    print(f'The absolute path for sample_data: {abs_path_sample_data}')

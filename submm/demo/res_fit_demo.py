import os
import sys
dir_this_file = os.path.dirname(os.path.realpath(__file__))
dir_notebook = os.path.join(dir_this_file, "res_fit.ipynb")
print("demo notebook located at")
print(dir_notebook)

try:
    import jupyterlab
except ModuleNotFoundError:
    input_str = input("This example requires a jupyterlab notebooks. Would you like to install that (y/n)?")
    if input_str.lower() == 'y':
        os.system("pip install jupyterlab")
    else:
        sys.exit("Exiting")


def run():
    os.system("jupyter notebook "+dir_notebook)


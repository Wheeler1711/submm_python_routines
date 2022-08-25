# Submm Python Routines
Python tools for submm astronomy instrumentation started at the University of Colorado Boulder

## About This Project
This project was started to make tools, used by everyone in the submm lab, available to 
everyone via version control on GitHub. This project continues to grow and expend in scope as
University of Colorado graduates and collaborators continue to contribute to the project.
Contributions and feature requests are not only welcome, but are a fundamental part of 
the culture of the submm astronomy.

# Required Software
## Git
Git and GitHub are standard tools across astronomy, computer science, computer engineering,
and other fields. This is a tool set that everyone should learn for code management. It is 
always useful, and often expected for contributors to a project of any size.

https://git-scm.com/downloads

This installation will include _gitbash_ on Windows machines. This is a terminal application
that simulates a unix terminal on Windows. It is useful for managing code and working with
git and github. If this is the first terminal/shell you have installed on your
Windows computer, you will be able to use this to enter all of the terminal
commands present in the rest of this README.md file.

A 15-minute introduction video for *git* is available at  https://www.youtube.com/watch?v=USjZcfj8yxE


## Python3
Get the latest version of Python3

https://www.python.org/downloads/

This project was tested with Python 3.8.5 to 3.10.x. It probably works with older
versions of Python, but it is not guaranteed.

Some python 
installations will need to call `python3` instead of `python` from the terminal. 
Check your python version with `python --version` if it comes back with 
2.7.x, try `python3 --version` and expect to get 3.x.x. 

Modern Python installations require you to sign and SSL certificate after
installation. This is done by simply running a script in the installation 
directory. Programs like, [Homebrew](https://brew.sh/) will do this step for
you. If you see SSL errors in Python, it is probably because you skip the
certificate step.

# Installation

## Download
From a commandline interface (terminal, gitbash, cmd, or powershell)
Clone this Repository (https://github.com/Wheeler1711/submm_python_routines)
using the following:

`git clone https://github.com/Wheeler1711/submm_python_routines`

then cd into the directory using:

`cd submm_python_routines`

Remember to update the repository periodically with:

`git pull`

## Virtual Environment Setup and Activation (recommended)

Configure the virtual environment for DetMap or from the terminal.

### Initial Setup 
This step is done one time to initialize the virtual environment.

Window and Linux Tested in Windows PowerShell and OSX Terminal. Some python 
installations will need to call `python3` instead of `python`. Check your python version with
`python --version` if it comes back with 2.7.x, try `python3 --version` and expect to get 3.x.x. 

If you have a many python installations, or if you just downloaded the latest version
of python then you should replace `python` with the full path to the python executable.

```
python -m pip install --upgrade pip
pip install virtualenv
virtualenv venv
```

### Activating a Virtual Environment
This step needs to be done everytime you want to use the virtual environment.

For unix-like environment activation:

```source venv/bin/activate```

Windows CMD environment activation:

```.\venv\Scripts\activate```

for Windows powershell environment activation:

```.\venv\Scripts\activate.ps1```

After activation, the term ail will add a `(venv)` to the command prompt. At this point
you can drop the full paths to the python and pip executables into the terminal, 
and the `python3` in place of `python` commands.

I like test that everything went as expected with the following:

```
python --version
pip list
```

### Package Installation
This step needs to be done everytime you want need to install new python packages.

The requirements.txt has the list of all packages needed for submm_python_routines. 
From a terminal with the virtual environment activated, do:

```pip install -r requirements.txt```

Packages can all be installed individually with `pip install <package_name>`

## Install as Python Packaage
If you are a user of the submm_python_routines but are unlikely modify (develop) 
the code in this project, then it is recommended that you install the 
`submm_python_routines` as a python package. This can be done within a
virtual environment, or for a user, (or global version of python which makes a 
mess https://xkcd.com/1987/).

### Install using setup.py
If you intend to use a virtual environment, activate the environment before this step. 
You can always check on the version of python you have installed with `python --version` 
and you current package list with `pip list`.

Run setup.py in the usual way:

```python setup.py install```

check the installation by starting a python console session with:

```python```

Then test with:

```
from KIDs.find_resonances_interactive import InteractiveFilterPlot
```

Exit the console session with:

```quit()```

## Run a Jupiter Server with Docker
To create a Jupyter notebook from the code in this repository _without installing python_. 

A Docker image is built with the code in this repository. 
From that image, a new container is launched with the permissions
and settings so that the full repository can be demonstrated.
This container reads/writes the code in the directory submm_python_routines; 
edits in the Jupyter Server window will propagate to the local repository, and local
changes are visible in the Jupyter Server. With git, this is all a user needs to
test and contribute to this repository.

Tested and designed to work on any platform.

1. Docker and docker-compose must be installed
   - get/install docker for your computer at
   [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

   - get/install docker-compose at:
[https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

2. Start the docker engine
   - Check Apps for Windows, OSX
   - Use a command-line call in unix
   - It is possible configure the docker engine to start on start-up for your machine.

3. Testing Docker Engine and Client
The docker *engine/Server*, not only client must both be running. Check for them in terminal with:

`docker version`

4. Go into submm_python_routines

```cd submm_python_routines```

5. One command to build and deploy the docker container with a Jupyter server.
   - Depending on how your permissions and operating system you may need use `sudo` 
   - If you have issues make sure the docker engine is running (step 3).
   - The first time build takes about 10 minutes.
   - Subsequent builds will take 2-3 seconds using cached data.
   - The notebook is accessible on any browsers (I like Chrome) at:
[http://127.0.0.1:8888/lab?token=](http://127.0.0.1:8888/lab?token=).
   - Use default password/token: `docker`

```docker-compose up --build```

6. Edit and run code

Changes made in the browser to files in submm_python_routines directory in the container are saved 
in the local computer at the same path, submm_python_routines. You can shut down the server and not
lose any work.

7. Shut Down the Jupyter server
    - Shut Down using the browser's drop down menu.
    - or by using `ctrl-c` in the terminal.

8. Clean up the container network built with docker-compose with one terminal command
    - This is best practice operation.

```docker-compose down```

## Add the submm_python_routines folder to your python path
This method works and is simple. However, is not recommended for most users. Simply put,
it teaches bad habits that can cause very hard to diagnose problems years later. The main issue
is the pollution of the python name space. It is easy to imagine a laboratory computer with
many users and a lot of code. If everyone adds to the python name spaces by appending to the
PYTHONPATH, then it is easy to get into a situation where there are multiple copies
modules with the same name (KIDs, utils, instruments, etc.) even multiple copies of the same 
repository from  different users and versions/branches of the code. 

To modify your PYTHONPATH environment variable, add the following line to your `~/.bashrc` (linux) or
`~/.bash_profile` (mac)  or for  windows see [creating-and-modifying-environment-variables-on-windows](https://docs.oracle.com/en/database/oracle/machine-learning/oml4r/1.5.1/oread/creating-and-modifying-environment-variables-on-windows.html).

`export PYTHONPATH="${PYTHONPATH}:/Users/jordan/submm_python_routines`

where you replace `/Users/jordan/submm_python_routines` with the local installation path.

## Examples 
### Resonance fitting
Maybe the most useful part of this code is the fitting of non-linear resonators. Care has been taken to make sure that resonance fitting is very fast so you can fit many resonators in real-time.
You can try it out by doing the following

```
cd submm_python_routines
python
```
```
from submm.demo import res_fit_demo
res_fit_demo.run()
```
### Interactive plotting tools
Real systems are never ideal. At least not anything I make. So when you have a lot of resonators some will have collided or some will just be weird. We try to make our automated code very good but when all else fails some human intervention will be needed. As such we have made some interactive plotting tools using matplotlib that allow for more sophisticated interaction with your data to better facilitate human interaction. In particular, there are interactive tools for identifying resonators and interactive tools for plotting resonators for examination. Try it out by running the below.
```
cd submm_python_routines
python submm/demo/find_res_and_fit.py
```
or 
```
python3 submm/demo/find_res_and_fit.py
```


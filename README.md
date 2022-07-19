# Submm Python Routines
Python routines for submm astronomy instrumentation at the University of Colorado Boulder

General Information for submm_python_routines

## About This Project
The idea was to make a bunch of coding tools that could be used by everyone in the submm lab
Then to post it all on GitHub.com so that there could be some version control.

# Expected software
## Git
https://git-scm.com/downloads

Make sure to install gitbash it you are on Windows.

A 15 minute introduction video for *git* is available at  https://www.youtube.com/watch?v=USjZcfj8yxE



## Python3
Get the latest version of Python3

https://www.python.org/downloads/

## QT5
This is for the matplotlib backend based on QT5

https://doc.qt.io/qt-5/supported-platforms.html

# Installation

## Download
From a commandline interface (terminal, gitbash, cmd, or powershell)
Clone this Repository (https://github.com/Wheeler1711/submm_python_routines)

`git clone https://github.com/Wheeler1711/submm_python_routines`

then cd into the directory using

`cd submm_python_routines`

Remember to update the repository periodically with.  

`git pull`

## Virtual Environment Setup and Activation (recommended)

Configure the virtual environment for DetMap or from the terminal.

### Initial Setup 
This step is only done one time to initialize the virtual environment.

Window and Linux Tested in Windows PowerShell and OSX Terminal. Some python 
installations will need to call `python3` instead of `python`. Check your python version with
`python --version` if it comes back with 2.7.x, try `python3 --version` and expect to get 3.x.x. 

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

### Package Installation
This step needs to be done everytime you want need to install new python packages.

The requirements.txt has the list of all packages needed for submm_python_rotines. 
From a terminal with the virtual environment activated, do:

```pip install -r requirements.txt```

## Install as Python Packaage
If you are a user of the submm_python_routines but are unlikely modify (develop) the code in this project,
then it is recommended that you install the `submm_python_routines` as a python package. This can be done within a
virtual environment, or for a user, (or global version of python which makes a mess https://xkcd.com/1987/).

### Install using setup.py
If you intend to use a virtual environment activate it before this step. You can always check on the version 
of python you have installed with `python --version` and you current package list with `pip list`.

Run setup.py in the usual way:

```python setup.py install```

check the installation by starting a python console session with:

```python```

Then test with:

```
from KIDs.find_resonances_interactive import InteractiveFilterPlot
```

then exit the console session with 

```quit()```

## Run a Jupiter Server with Docker
To create a Jupyter notebook from the code in this repository without python. 

A Docker image is built with the code in this repository. 
From that image, a new container is launched with the permissions
and settings so that the full repository can be demonstrated.
This container reads/writes the code in DetMap/detmap, edits in the Jupyter
Server propagate to the local repository. With git, this is all a user needs to
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
The docker *engine/Server* and client must both be running. Check for them in terminal with:

`docker version`

4. Go into DetMap

```cd DetMap```

5. One command to build and deploy the docker container with a Jupyter server.
   - Depending on how your permissions and operating system you may need use `sudo` 
   - If you have issues make sure the docker engine is running (step 3).
   - The first time build takes about 10 minutes.
   - Subsequent builds will take 2-3 seconds using cached data.
   - The notebook is arable on any browsers (I like Chrome) at:
[http://127.0.0.1:8888/lab?token=](http://127.0.0.1:8888/lab?token=).
   - Use the password/token: `docker`

```docker-compose up --build```

6. Edit and run code

Changes made in the browser to files in DetMap/detmap directory in the container are saved 
in the local computer at the same path, DetMap/detmap. You can shut down the server and not
lose any work.

7. Shut Down the Jupyter server
    - Shut Down using the browser's drop down menu.
    - or by using `ctrl-c` in the terminal.

8. Clean up the container network built with docker-compose with one terminal command
    - This is best practice operation.

```docker-compose down```

## Add the submm_python_routines folder to your python path
to modify your PYTHONPATH environment variable, add the following line to your `~/.bashrc` (linux) or
`~/.bash_profile` (mac)  or for  windows see [creating-and-modifying-environment-variables-on-windows](https://docs.oracle.com/en/database/oracle/machine-learning/oml4r/1.5.1/oread/creating-and-modifying-environment-variables-on-windows.html).

`export PYTHONPATH="${PYTHONPATH}:/Users/jordan/submm_python_routines`

where you replace `/Users/jordan/submm_python_routines` with whatever the local installation path is.


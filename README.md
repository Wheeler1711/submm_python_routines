# submm_python_routines
Python routines for submm instrumentation astronomy at the University of Colorado Boulder

General Information for submm_python_routines

The idea was to make a bunch of coding tools that could be used by everyone in the submm lab
Then to post it all on github so that there could be some version control

basically just update this folder to run the new version of things.
to import the funciton located in here you need to have a blank file
called __init__.py that turns the folder into a Package such that you
can import the function read_multitone.py from the subdirectory
multitone_kidPy with the command import multitone_kidPy.read_multitone

to modifiy you python path and the following to your .bash_profile on mac or .bash_rc on Linux
export PYTHONPATH="${PYTHONPATH}:/Users/jordan/submm_python_routines"

where you replace /Users/jordan/submm_python_routines with whatever the destiantion of the python routines is on your
machine.


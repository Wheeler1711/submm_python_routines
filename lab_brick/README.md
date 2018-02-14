# lab-brick-attenuator



udev rule

Create a new udev rule in /etc/udev/rules.d/99-lab-brick.rules

~~~~
ACTION=="add", SUBSYSTEMS=="usb", ATTRS{idVendor}=="041f", ATTRS{idProduct}=="1208", MODE="660", GROUP="plugdev"
~~~~

ensure the user you are running  is a part of plugdev

Restart after the rule is installed for it to take effect

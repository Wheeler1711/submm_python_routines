import numpy as np
from LabBrick import core

#to use this do 
#sudo python
#execfile("labbrick_startup.py")

#attn2 = core.Attenuator(0x041f,0x1208,"16776")
attn1 = core.Attenuator(0x041f,0x1208,"15915")
attn2 = core.Attenuator(0x041f,0x1208,"16776")


print "15915 is attn1"
print "16776 is attn2"
print "example command: attn1.get_attenuation()"
print "example command: attn2.set_attenuation(29)"

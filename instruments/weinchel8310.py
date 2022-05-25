import pyvisa
import time


class attenuator(object):

    # Default constructor used to call the instrument at the given
    # GPIBaddress and initializes the object to the instrument
    def __init__(self,address = None):
        if not address:
            self.GPIBaddress = 'GPIB0::10::INSTR'
        else:
            self.GPIBaddress = address
        rm = pyvisa.ResourceManager("@py")
        self.obj1 = rm.open_resource(self.GPIBaddress)

    def get_attenuation(self, channel=1):
        cmd_str = 'CHAN %d' % (channel)
        self.obj1.write(cmd_str)
        time.sleep(.25)
        attenuation = float(self.obj1.query('ATTN?'))
        return attenuation

    def set_attenuation(self, attenuation, channel=1):

        cmd_str = 'CHAN %d' % (channel)
        self.obj1.write(cmd_str)
        time.sleep(.15)
        cmd_str = 'ATTN %d' % (attenuation)
        self.obj1.write(cmd_str)
        time.sleep(.15)

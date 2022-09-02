import pyvisa as visa

class temp_control(object):

    # Default constructor used to call the instrument at the given
    # GPIBaddress and initializes the object to the instrument
    def __init__(self):
        self.GPIBaddress = 'GPIB0::12::INSTR'
        rm = visa.ResourceManager("@py")
        self.obj1 = rm.open_resource(self.GPIBaddress)

    def get_temp(self,ch = 'A'):
        temp = float(self.obj1.query('INP '+  ch + ':TEMP?'))
        return temp


    def set_temp(self,temp,ch = 1):
        self.obj1.write('LOOP 1:SETP '+ str(temp))
        return

    def get_set_temp(self):
        set_temp = float(self.obj1.query('LOOP 1:SETP?')[:-1])
        return set_temp

    def turn_heater_on(self):
        self.obj1.write('CONTROL')
        return

    def turn_heater_off(self):
        self.obj1.write('STOP')
        return


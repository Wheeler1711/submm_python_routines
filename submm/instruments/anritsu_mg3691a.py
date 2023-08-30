import pyvisa

class synthesizer(object):

    # Default constructor used to call the instrument at the given                                                                              
    # GPIBaddress and initializes the object to the instrument                                                                                  
    def __init__(self):
        self.GPIBaddress = 'GPIB0::5::INSTR'
        rm = pyvisa.ResourceManager("@py")
        self.obj1 = rm.open_resource(self.GPIBaddress)

    # This method returns the CURRENT frequency that synthesizer is at                                                                          
    # Using the built in command - OF1                                                                                                          
    def get_frequency(self):  # does not work yet                                                                                               
        frequency = float(self.obj1.query('OF0'))
        return frequency

    # This method will set the frequency of the Anritsu to the                                                                                  
    # specified amount. The given freq_value needs to be within                                                                                 
    # the range of 10MHz - 20GHz.                                                                                                               
    def set_frequency(self, freq_value):

        # If value is below/above a certain threshold,                                                                                          
        # let user know to re-enter                                                                                                             

        if(freq_value < 9000):
            print(freq_value)
            print("Invalid value - Needs to be greater than 10MHz")
        elif(freq_value > 8400000000):
            print("Invalid value - needs to be smaller than 20GHz")
        else:  # matlab code 'FREQ %.9f GHZ\n',fGHz                                                                                             
            cmd_str = 'CF0 %9.9fGH' % (freq_value/1e9)
            self.obj1.write(cmd_str)


    def get_power(self):
        power = float(self.obj1.query('OL0'))
        return power

    # This method will set the power to the specified amount                                                                                      
    def set_power(self, power_value):

        # If the value is below/above a certain threshold,                                                                                        
        # let user know to re-enter                                                                                                               
        if (power_value < -20):
            print("Invalid value - Needs to be greater than -20dbm")
        elif (power_value > 30):
            print("Invalid value - Needs to be less than 30dbm")
        else:
            cmd_str = 'L0 %3.2fDM' % power_value
            self.obj1.write(cmd_str)

    # This method will turn the output to ON                                                                                                      
    def turn_outputOn(self):
        self.obj1.write('RF1')

    # This method will turn the output to OFF                                                                                                     
    def turn_outputOff(self):
        self.obj1.write('RF0')

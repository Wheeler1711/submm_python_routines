import pyvisa

class synthesizer(object):

    # Default constructor used to call the instrument at the given                                                                              
    # GPIBaddress and initializes the object to the instrument                                                                                  
    def __init__(self,address = None):
        if not address:
            self.GPIBaddress = 'GPIB0::21::INSTR'
        else:
            self.GPIBaddress = address
        rm = pyvisa.ResourceManager("@py")
        self.obj1 = rm.open_resource(self.GPIBaddress)
                                                                                                        
    def get_frequency(self):  # does not work yet                                                                                               
        frequency = float(self.obj1.query('FREQ?'))
        return frequency

    # This method will set the frequency of the Anritsu to the                                                                                  
    # specified amount. The given freq_value needs to be within                                                                                 
    # the range of 10MHz - 20GHz.                                                                                                               
    def set_frequency(self, freq_value):
        '''
        freq_value in Hz
        '''                                                                                                           

        if(freq_value < 9000):
            print(freq_value)
            print("Invalid value - Needs to be greater than 10MHz")
        elif(freq_value > 8400000000):
            print("Invalid value - needs to be smaller than 20GHz")
        else:  # matlab code 'FREQ %.9f GHz\n',fGHz                                                                                          
            cmd_str = 'FREQ %9.9f GHz' % (freq_value/1e9)
            self.obj1.write(cmd_str)


    def get_power(self):
        power = float(self.obj1.query('POW:AMPL?')) #
        return power

    # This method will set the power to the specified amount                                                                                      
    def set_power(self, power_value):

        # If the value is below/above a certain threshold,                                                                                        
        # let user know to re-enter                                                                                                               
        if (power_value < -20):
            print("Invalid value - Needs to be greater than -20dbm")
        elif (power_value > 30):
            print("Invalid value - Needs to be less than 30dbm")
        else: #'POW:AMPL %f dBm\n', pdBm)
            cmd_str = 'POW:AMPL %f dBm' % power_value
            self.obj1.write(cmd_str)

    # This method will turn the output to ON                                                                                                      
    def turn_outputOn(self):
        self.obj1.write('OUTP:STAT ON')

    # This method will turn the output to OFF                                                                                                     
    def turn_outputOff(self):
        self.obj1.write('OUTP:STAT OFF')
        
    def get_output_state(self):
        state = bool(int(self.obj1.query('OUTP:STAT?')))
        return state

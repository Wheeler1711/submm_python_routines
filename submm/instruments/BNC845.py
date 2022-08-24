import pyvisa


class synthesizer(object):

    # Default constructor used to call the instrument at the given
    # GPIBaddress and initializes the object to the instrument
    def __init__(self):
        self.GPIBaddress = 'USB0::1003::45055::141-213330000-0152::0::INSTR'
        rm = pyvisa.ResourceManager("@py")
        self.obj1 = rm.open_resource(self.GPIBaddress)

    # This method returns the CURRENT frequency that synthesizer is at
    # Using the built in command - OF1
    def get_frequency(self):  # does not work yet
        frequency = float(self.obj1.query('FREQ?'))
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
        elif(freq_value > 26500000000):
            print("Invalid value - needs to be smaller than 20GHz")
        else:  # matlab code 'FREQ %.9f GHZ\n',fGHz
            cmd_str = 'FREQ %.9f GHZ' % (freq_value/1e9)
            self.obj1.write(cmd_str)
#freq = self.get_frequency()
# print(freq)

    # This method will return the CURRENT power that the frequency
    # is on
    def get_power(self):
        power = float(self.obj1.query('POW:LEV?'))
        return power

    # This method will set the power to the specified amount
    def set_power(self, power_value):  # matlab code POW:LEV %f dBM\n'

        # If the value is below/above a certain threshold,
        # let user know to re-enter
        if(power_value < -20):
            print("Invalid value - Needs to be greater than -20dbm")
        elif(power_value > 30):
            print("Invalid value - Needs to be less than 30dbm")
        else:
            cmd_str = 'POW:LEV %f dBM' % power_value
            self.obj1.write(cmd_str)

    # This method will turn the output to ON 'OUTP:STAT?'
    def turn_output_on(self):
        self.obj1.write('OUTP:STAT ON')

    # This method will turn the output to OFF
    def turn_output_off(self):
        self.obj1.write('OUTP:STAT OFF')

    def get_output(self):
        output = bool(int(self.obj1.query('OUTP:STAT?')))
        return output

    # This method will print out to terminal all of the available
    # functions that are included in this class
    def help(self):
        print("")
        print("\t Available Functions")
        print("\t --------------------")
        print("\t set_frequency(Hz)")
        print("\t get_frequency()")
        print("\t get_power()")
        print("\t set_power(Dbm)")
        print("\t turn_output_on()")
        print("\t turn_output_off()")
        print("\t get_output()")

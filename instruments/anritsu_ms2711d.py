'''
Spectrum analyzer object for remote reading out of a
Anritsu MS2711D handheld spectrum analyzer
Author Jordan Wheeler Wheeler1711@gmail.com

Note: commands must be entered in remote mode and then to
resume sweeping you must exit remote mode
enter_remote_mode waits until the sweep is finished and thus 
the serial timeout must be longer than the sweep time 
                               Span    
sweep time is propotional to ----------   
                              RBW*VBW          
'''
#check for usb to serial with
# ls tty.usbserial*
# written for python3
# requires pyserial and numpy
# pip install pyserial

import serial
import numpy as np
import struct

class spectrum_analyzer(object):

    def __init__(self):
        self.ser = serial.Serial('/dev/tty.usbserial') # mac usb to serial converter Windows COM3 or something
        #print(self.ser.name)
        self.ser.timeout = 30

    def check_response_for_error(self,rx): # standard response from ms2711d to confirm data tranmission
        if struct.unpack('B',rx)[0] == 255: #success
            return True
        elif struct.unpack('B',rx)[0] == 224:
            print("invaled paramter")
            return False
        elif struct.unpack('B',rx)[0] == 238:
            print("time out error")
            return False
        else:
            print("unknown error")
            return False

    def enter_remote_mode(self):
        SerialCommand = struct.pack('B', 69) # 69 is the Enter Remote Mode serial command
        self.ser.write(SerialCommand)
        rx = self.ser.read(13)
        if len(rx) == 13:
            return True
        else:
            return False

    def exit_remote_mode(self):
        SerialCommand = struct.pack('B', 255) # 255 is the exit Remote Mode serial command
        self.ser.write(SerialCommand)
        rx = self.ser.read(1)
        #print(rx)
        if len(rx) == 1:
            return True
        else:
            return False

    def grab_last_sweep_command(self):
        # there is a bunch of stuff in the returned byte array
        # see programming manual for details
        # should proabably grab more info return a dictionary
        SerialCommand = struct.pack('B', 33) # grab sweep
        SerialCommand2 = struct.pack('B', 0) #last sweep
        self.ser.write(SerialCommand)
        self.ser.write(SerialCommand2)
        rx = self.ser.read(2035)

        # get frequencies
        start_freq_bytes = rx[56:60]
        stop_freq_bytes =rx[60:64]
        freq_scale_factor_bytes = rx[334:336]
        freq_scale_factor = struct.unpack('>H',freq_scale_factor_bytes )[0]
        start_freq = struct.unpack('>I',start_freq_bytes )[0]/freq_scale_factor
        stop_freq = struct.unpack('>I',stop_freq_bytes )[0]/freq_scale_factor
        freqs = np.linspace(start_freq,stop_freq,401)

        # get data
        if len(rx) != 2035:
            return False
        else:
            #data is 432 to 2035 4 bytes per point
            data = np.zeros(401)
            for i in range(0,401):
                data[i] = (struct.unpack('>i', rx[431+i*4:435+i*4])[0]-270000)/1000

            return freqs, data

    def grab_last_sweep(self): # wrapper includes enter exit remote mode
        self.enter_remote_mode()
        freqs,data = self.grab_last_sweep_command()
        self.exit_remote_mode()
        return freqs, data

    def set_resolution_bandwidth_command(self,bandwidth):
        bandwidth_options = np.asarray((100,300,1000,3000,10000,30000,100000,300000,1000000)) #only some bandwiths are availible
        if ~np.any(bandwidth == bandwidth_options):
            print("Please select a proper bandwidth option i.e.")
            print(bandwidth_options)
            return
        else:
            SerialCommand = struct.pack('B', 141) # set bandwidth
            SerialCommand2 = struct.pack('>I', bandwidth) #unsiged int
            self.ser.write(SerialCommand)
            self.ser.write(SerialCommand2)
            rx = self.ser.read(1)
            return self.check_response_for_error(rx)  # success?   

    def set_resolution_bandwidth(self,bandwidth): # wrapper includes enter exit remote mode
        self.enter_remote_mode()
        success = self.set_resolution_bandwidth_command(bandwidth)
        self.exit_remote_mode()
        return success

    def set_start_and_stop_frequency_command(self,start,stop):
        # must be integer
        SerialCommand = struct.pack('B', 99) # set start and stop frequency
        # in Hertz
        SerialCommand2 = struct.pack('>I', start) #unsigned int
        SerialCommand3 = struct.pack('>I', stop)
        self.ser.write(SerialCommand)
        self.ser.write(SerialCommand2)
        self.ser.write(SerialCommand3)
        rx = self.ser.read(1)
        return self.check_response_for_error(rx)

    def set_start_and_stop_frequency(self,start,stop): # wrapper includes enter exit remote mode
        self.enter_remote_mode()
        success = self.set_start_and_stop_frequency_command(start,stop)
        self.exit_remote_mode()
        return success

    def set_attenuation_command(self,atten):
        if atten < 0:
            print("attenuation value too low")
            return False
        elif atten>51:
            print("attenuation value too high")
            return False
        else:
            pass #just right
        
        SerialCommand = struct.pack('B', 143) # set start and stop frequency
        SerialCommand2 = struct.pack('B', atten) #unsigned char
        self.ser.write(SerialCommand)
        self.ser.write(SerialCommand2)
        rx = self.ser.read(1)
        return self.check_response_for_error(rx)

    def set_attenuation(self,atten):
        self.enter_remote_mode()
        success = self.set_attenuation_command(atten)
        self.exit_remote_mode()
        return success

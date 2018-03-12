# Last Updated: 02/07/2018

import visa

class Anritsu(object):
	
	# Default constructor used to call the instrument at the given
	# GPIBaddress and initializes the object to the instrument
	def __init__(self):
		self.GPIBaddress = 'GPIB1::23::INSTR'
		rm = visa.ResourceManager()
		self.obj1 = rm.open_resource(self.GPIBaddress)
	
	# This method returns the CURRENT frequency that Anritsu is at
	# Using the built in command - OF1
	def get_frequency(self):
		frequency = float(self.obj1.query('OF1'))
		return frequency*10**6
	
	# This method will set the frequency of the Anritsu to the 
	# specified amount. The given freq_value needs to be within
	# the range of 10MHz - 20GHz. 
	def set_frequency(self, freq_value):
		
		# If value is below/above a certain threshold, 
		# let user know to re-enter
		
		if(freq_value < 10000000):
			print "Invalid value - Needs to be greater than 10MHz"
		elif(freq_value > 20000000000):
			print "Invalid value - needs to be smaller than 20GHz"
		else:
			cmd_str = 'CF1 %9.9fGH' % (freq_value/1e9)
			self.obj1.write(cmd_str)
		
    
	# This method will return the CURRENT power that the frequency
	# is on
	def get_power(self):
		power = float(self.obj1.query('OL1'))
		return power
	
	# This method will set the power to the specified amount 
	def set_power(self, power_value):
		
		# If the value is below/above a certain threshold,
		# let user know to re-enter
		if(power_value < -20):
			print "Invalid value - Needs to be greater than -20dbm"
		elif(power_value > 30):
			print "Invalid value - Needs to be less than 30dbm"
		else:
			cmd_str = 'L1 %3.2fDM' % power_value
			self.obj1.write(cmd_str)
	
	# This method will turn the output to ON
	def turn_outputOn(self):
		self.obj1.write('RF1')
	
	# This method will turn the output to OFF
	def turn_outputOff(self):
		self.obj1.write('RF0')
	
	# This method will print out to terminal all of the available
	# functions that are included in this class
	def help(self):
		print ""
		print "\t Available Functions"
		print "\t --------------------"
		print "\t set_frequency(Hz)"
		print "\t get_frequency()"
		print "\t get_power()"
		print "\t set_power(Dbm)"
		print "\t turn_outputOn"
		print "\t turn_outputOff"
		
	


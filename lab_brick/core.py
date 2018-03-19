import usb.core 
import threading
from datetime import datetime
import sys
import numpy as np
import platform


'''
Got this of some guys github and made a few edits 
see https://github.com/markafarrell/lab-brick-attenuator.git for the original
-edited so that it can connect to more that one labbrick that have different serial numbers
-edited to put the call back function in the core file for attenuation_change and attenuation response
-edited to return the value for get_attenuation()

Currently all other function besides get_attenuation and set_attenuation will fail unless 
User deffined callback are defined for each commmand or that part of the command is commented 
out example(set_attenuation is currently commmented out)
'''

def attenuation_change(device, status_byte, count, byteblock, val):
	print "User Defined Callback for attenuation_level_change"
	print "Attenuation: " + str(val)
	
def attenuation_response(device, status_byte, count, byteblock, val):
	#print "User Defined Callback for attenuation_level_response"
	print "Attenuation: " + str(val)
	#return val

class Attenuator(object):
	"""Control a Vaunix Lab-brick attenuator"""

	#################################################################################
	#								USB VARIABLES									#
	#################################################################################
	
	__vid = 0x041f #0x00 
	__pid = 0x1208 #0x00
	serial = "15915" #
	
	__dev = False
	__write_endpoint = True 
	__read_endpoint = None
	
	#################################################################################
	#									VARIABLES									#
	#################################################################################
	
	attenuation_level = -1
	ramp_dwell_time = -1
	ramp_dwell_time_bidirectional = -1
	ramp_start_attenuation_level = -1
	ramp_stop_attenuation_level = -1
	ramp_step_size = -1
	ramp_step_size_bidirectional = -1
	
	ramp_status = -1
	
	# 0 - Ramping up
	# 1 - Ramping down
	
	ramp_mode = -1
	
	# 1 - Single ramp
	# 2 - Continuous ramp
	
	ramp_wait_time = -1
	ramp_wait_time_bidirectional = -1
	
	maximum_attenuation = -1
	
	save_required = -1
	
	pll_lock = -1
	
	rf_hardware_state = -1
	
	connected = False

	__read_thread = False
	
	
	#################################################################################
	#									LOGGING										#
	#################################################################################
	
	__log = sys.stderr
	__write_log_string = ">>> %s - %s \t cmd: 0x%02x \t count: %d \t byteblock: 0x%02x 0x%02x 0x%02x 0x%02x 0x%02x 0x%02x\n"
	__read_log_string = "<<< %s - %s \t res: 0x%02x \t count: %d \t byteblock: 0x%02x 0x%02x 0x%02x 0x%02x 0x%02x 0x%02x\n"
	__cmd_log_string = ">>> %s - %s \t %s\n"
	__res_log_string = "<<< %s - %s \t %s %d\n"
	__status_log_string = "<<< %s - %s \t status: 0x%02x \t dev_status: 0b%x \t attenuation: %d\n"
	
	#################################################################################
	#									CONSTRUCTOR									#
	#################################################################################
		
	def __init__(self, vid, pid,serial_number, debug=False):
	
		self.__vid = vid
		self.__pid = pid

		if debug:
			self.__log.write("vid=0x%04x pid=0x%04x\n" % (self.__vid, self.__pid)) 
	
		# Jordan edited to handle more than one labbrick
		devices = usb.core.find(idVendor=self.__vid, idProduct=self.__pid, find_all = True)


		#loop through devices to find the right serial number
		for d in devices:
			self.__dev = d
			if self.__dev is None:
				raise ValueError('Device not found')
			
			# Does not work on windows
			if platform.system() != 'Windows':
				if self.__dev.is_kernel_driver_active(0):
					try:
						self.__dev.detach_kernel_driver(0)
						print "kernel driver detached"
					except usb.core.USBError as e:
						sys.exit("Could not detach kernel driver: %s" % str(e))
			
			try:
				print "marker 4"
				self.__dev.set_configuration()

				# get an endpoint instance
				print "marker 3"
				cfg = self.__dev.get_active_configuration()
				print "marker 2"
				intf = cfg[(0,0)]
				print "marker 1"
				self.__read_endpoint = usb.util.find_descriptor(
					intf,
				# match the first OUT endpoint
					custom_match = \
					lambda e: \
						usb.util.endpoint_direction(e.bEndpointAddress) == \
						usb.util.ENDPOINT_IN)

			#Store the serial number of the lab brick
			
				self.serial = self.get_serial_number()
				print self.serial

			
			#Spawn read thread to handle reading for lab brick
			
				self.connected = True
			except:
				# Something bad happened during connection
				self.connected = False
			print self.__read_endpoint	
			#assert self.__read_endpoint is not None #comment this out was breaking code if attenuators were called in wrong order
			if self.serial == "SN:"+str(serial_number):
				print "correct device"
				break
			else:
				self.connected = False
		

		self.spawn_read_thread(debug)

		assert self.connected == True

	def spawn_read_thread(self, debug=False):
		"""	Spawn the thread to get responses from the lab brick """
		
		self.__read_thread = threading.Thread(target=self.read_data, args=(debug,))
		self.__read_thread.daemon = True
		self.__read_thread.start()
	
	def get_serial_number(self):
		"""	Get the serial number of the connected device """
		return self.__dev.serial_number
		
	#################################################################################
	#										READ									#
	#################################################################################
	
	def read_data(self, debug=False):
	
		data = None

		while True:
			try:
				data = self.__read_endpoint.read(8)
								
				status_byte = data[0]
		
				count = data[1]
				
				byteblock = bytearray( [ data[2], data[3], data[4], data[5], data[6], data[7] ] )
				
				if debug:
					self.__log.write(self.__read_log_string % ( datetime.now(), 'read_data', status_byte, count, byteblock[0], 
						byteblock[1], byteblock[2], byteblock[3], byteblock[4], byteblock[5]))
						
				self.handle_response(status_byte, count, byteblock, debug)

			except usb.core.USBError as e:
				# USB Error
				
				# TODO: Handle this
				pass
	
	#################################################################################
	#									COMMANDS									#
	#################################################################################
	
	# All commands send to the device are USB Control Commands.
	
	# self.__dev.ctrl_transfer(bmRequestType, bmRequest, wValue, wIndex, data) 
	
	""" 
		bmRequestType 	- 	0x21
		bmRequest 		-	9
		wValue			-	0x0200
		wIndex			-	0x0000
	"""
	
	
	def send_command(self, command, count, byteblock, debug=False):
	
		bytes = bytearray( [ command, count ] )

		bytes += byteblock
		
		if debug:
			self.__log.write(self.__write_log_string % ( datetime.now(), 'send_command', command, count, byteblock[0], 
				byteblock[1], byteblock[2], byteblock[3], byteblock[4], byteblock[5]))
		
		#self.__endpoint.write(bytes)
		self.__dev.ctrl_transfer( 0x21, 0x9, 0x0200, 0x0000, bytes)
		
	def set_attenuation(self, attenuation, debug=False):
		#self.register_callback('attenuation_level_change', attenuation_change) # don't need to hear back every time
	
		"""	Set attenuation level in 0.25 dB steps. 
			0x00 is the maximum power, 0x02 is 0.5dB of attenuation. 
			Note that the resoulution of the power output setting is only 0.5dB, 
			the least significant bit of the value is ignored. 
					
			Input in dB
		"""
	
		command = 0x8D
		count = 1
		
		#Round down attenuation to nearest 0.25dB interval
		val = int(attenuation * 4)
		
		byteblock = bytearray( [ val & 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_attenuation', 'attenuation=' + str(attenuation)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_attenuation(self, debug=False):
		val = self.register_callback('attenuation_level_response', attenuation_response)
		
	
		"""	Get attenuation level in 0.25 dB steps. 
			0x00 is the maximum power, 0x02 is 0.5dB of attenuation. 
			Note that the resoulution of the power output setting is only 0.5dB, 
			the least significant bit of the value is ignored. 
		"""
	
		command = 0x0D
		
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_attenuation', ""))
			
		self.send_command(command, count, byteblock, debug)
		return float(self.attenuation_level)
	
	def set_ramp_dwell_time(self, time, debug=False):
	
		"""	Set time to dwell at each attenuation level during a ramp in 1 
			millisecond intervals
		
			Input in milliseconds
		"""
	
		command = 0xB3
		count = 4
		byteblock = bytearray( [ time & 0xFF, (time >> 8) & 0xFF, (time >> 16) & 0xFF, (time >> 24) & 0xFF, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_dwell_time', 'dwell_time=' + str(time)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_dwell_time(self, debug=False):
	
		"""	Get time to dwell at each attenuation level during a ramp in 1 
			millisecond intervals
		"""
	
		command = 0x33
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_ramp_dwell_time', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def set_ramp_dwell_time_bidirectional(self, time, debug=False):
	
		"""	Set time to dwell at each attenuation level during a ramp in 1 
			millisecond intervals
		
			Input in milliseconds
		"""
	
		command = 0xB7
		count = 4
		byteblock = bytearray( [ time & 0xFF, (time >> 8) & 0xFF, (time >> 16) & 0xFF, (time >> 24) & 0xFF, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_dwell_time_bidirectional', 'dwell_time=' + str(time)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_dwell_time_bidirectional(self, debug=False):
	
		"""	Get time to dwell at each attenuation level during a ramp in 1 
			millisecond intervals
		"""
	
		#TODO: Check this command byte
		command = 0x37
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_ramp_dwell_time_bidirectional', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def set_ramp_start_attenuation_level(self, attenuation, debug=False):
	
		"""	Set start level of the ramp in 0.25 dB units
	
			Input in dB
		"""
	
		command = 0xB0
		count = 4
		
		#Round down attenuation to nearest 0.25dB interval
		val = int(attenuation * 4)
		
		byteblock = bytearray( [ val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_start_attenuation_level', 'attenuation=' + str(attenuation)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_start_attenuation_level(self, debug=False):
	
		"""	Get start level of the ramp in 0.25 dB units """
	
		command = 0x30
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_ramp_start_attenuation_level', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def set_ramp_stop_attenuation_level(self, attenuation, debug=False):
	
		"""	Set end level of the ramp in 0.25 dB units
	
			Input in dB
		"""
	
		command = 0xB1
		count = 4
		
		#Round down attenuation to nearest 0.25dB interval
		val = int(attenuation * 4)
		
		byteblock = bytearray( [ val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_stop_attenuation_level', 'attenuation=' + str(attenuation)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_stop_attenuation_level(self, debug=False):
	
		"""	Get end level of the ramp in 0.25 dB units """
	
		command = 0x31
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )

		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_stop_attenuation_level', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def set_ramp_step_size(self, step_size, debug=False):
	
		""" Set ramp step size in 0.25 dB units
	
			Input in dB
		"""
	
		command = 0xB2
		count = 4
		
		#Round down step size to nearest 0.25dB interval
		val = int(step_size * 4)
		
		byteblock = bytearray( [ val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_step_size', 'step_size=' + str(step_size)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_step_size(self, step_size, debug=False):
	
		""" Get ramp step size in 0.25 dB units """
	
		command = 0x32
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_ramp_step_size', 'step_size=' + str(step_size)))
		
		self.send_command(command, count, byteblock, debug)
	
	def start_ramp(self, type, debug=False):
	
		""" Start the Ramp operation. 
			Type 
			 - 01 for a single ramp
			 - 02 for continuous ramps.
			 - 03 for continuous up then down.
		"""
	
		command = 0x89
		count = 3
		byteblock = bytearray( [ type, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'start_ramp', 'type=' + str(type)))
		
		self.send_command(command, count, byteblock, debug)
	
	def start_ramp_single(self, debug=False):
	
		""" Start single Ramp operation. """
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'start_ramp_single', ''))
			
		self.start_ramp(0x01, debug)	

	def start_ramp_continuous(self, debug=False):
	
		""" Start continuous Ramp operation. """
	
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'start_ramp_continuous', ''))
	
		self.start_ramp(0x02, debug)

	def start_ramp_down(self, debug=False):
	
		""" Start continuous Ramp operation. """
	
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'start_ramp_continuous', ''))
	
		self.start_ramp(0x05, debug)

	def start_ramp_continuous_bidirectional(self, debug=False):
	
		""" Start continuous Ramp operation. """
	
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'start_ramp_continuous', ''))
	
		self.start_ramp(0x13, debug)
	def stop_ramp(self, debug=False):
	
		""" Stop ramp operation. """ 
	
		command = 0x39
		count = 1
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'stop_ramp', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def set_ramp_wait_time(self, wait_time, debug=False):
	
		""" Set time to wait before starting a new ramp in 1 millisecond intervals, 
			when continuous ramps are in operation.
		"""
	
		command = 0xB6
		
		# Count is documented as 1 but input is specified as a DWORD count=4 makes more sense
		
		count = 4
		byteblock = bytearray( [ wait_time & 0xFF, (wait_time >> 8) & 0xFF, (wait_time >> 16) & 0xFF, (wait_time >> 24) & 0xFF, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_wait_time', 'wait_time=' + str(wait_time)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_wait_time(self, debug=False):
	
		""" Get time to wait before starting a new ramp in 1 millisecond intervals, 
			when continuous ramps are in operation.
		"""
	
		command = 0x36
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_ramp_wait_time', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def set_ramp_wait_time_bidirectional(self, wait_time, debug=False):
	
		""" Set time to wait before starting a new ramp in 1 millisecond intervals, 
			when continuous ramps are in operation.
		"""
	
		command = 0xB9
		
		# Count is documented as 1 but input is specified as a DWORD count=4 makes more sense
		
		count = 4
		byteblock = bytearray( [ wait_time & 0xFF, (wait_time >> 8) & 0xFF, (wait_time >> 16) & 0xFF, (wait_time >> 24) & 0xFF, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_wait_time_bidirectional', 'wait_time=' + str(wait_time)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_wait_time_bidirectional(self, debug=False):
	
		""" Get time to wait before starting a new ramp in 1 millisecond intervals, 
			when continuous ramps are in operation.
		"""
	
		command = 0x39
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_ramp_wait_time_bidirectional', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def set_ramp_step_size_bidirectional(self, step_size, debug=False):
	
		""" Set the step size for when stepping down.
		"""
	
		command = 0xB8
		
		#Round down step size to nearest 0.25dB interval
		val = int(step_size * 4)
		
		byteblock = bytearray( [ val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF, 0x00, 0x00 ] )
		
		count = 4
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'set_ramp_step_size_bidirectional', 'step_size=' + str(step_size)))
		
		self.send_command(command, count, byteblock, debug)
	
	def get_ramp_step_size_bidirectional(self, debug=False):
	
		""" Get the step size for when stepping down
		"""
	
		command = 0x38
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_ramp_step_size_bidirectional', ''))
		
		self.send_command(command, count, byteblock, debug)
	
	def restore_defaults(self, debug=False):
	
		"""	Resets all of the parameters to their factory default settings """
		
		command = 0x8F
		
		# Not sure what count or byteblock should be here as it isn't documented very well
		count = 1 
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'restore_defaults', ''))
		
		self.send_command(command, count, byteblock, debug)
		
	def get_maximum_attenuation(self, debug=False):
	
		""" Get the maximum possible attenuation value in 0.25dB units """
	
		### TODO: Confirm actual units
	
		command = 0x35
		count = 0
		byteblock = bytearray( [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'get_maximum_attenuation', ''))
		
		self.send_command(command, count, byteblock, debug)
		
	def save_user_parameters(self, debug=False):
	
		"""	The first three bytes of the byteblock must be set to 0x42, 0x55, 0x31 
			as a key to enable the save operation. 
			Save User Parameters records the attenuation and ramp settings into 
			non-volatile memory in the Lba Brick, The Lab Brick will reload these 
			parameters when it is powered on. 
		"""
	
		command = 0x8C
		count = 3
		byteblock = bytearray( [ 0x42, 0x55, 0x31, 0x00, 0x00, 0x00 ] )
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'save_user_parameters', ''))
		
		self.send_command(command, count, byteblock, debug)
			
	#################################################################################
	#									RESPONSES									#
	#################################################################################
	
	def handle_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Handle responses for the lab brick and call the correct handler as per 
			the status bit.
		"""
	
		try:
			(self.__responses[status_byte])(self, status_byte, count, byteblock, debug)
		except KeyError:
			1+1
			#self.__log.write(self.__read_log_string % ( datetime.now(), 'read_data', status_byte, count, byteblock[0], 
#						byteblock[1], byteblock[2], byteblock[3], byteblock[4], byteblock[5]))
			#self.__log.write("Status Byte of response not recognised\n")
	
	def attenuation_level_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x04 or 0x0D
		
			Byte = attenuation level in 0.25 dB steps. 0x00 is the maximum power, 
			0x02 is 0.5dB of attenuation. Note that the resoulution of the power 
			output setting is only 0.5dB, the least significant bit of the value 
			is ignored.
		"""
	
		if status_byte == 0x0D:
			val = byteblock[0]
		elif status_byte == 0x04:
			val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		val = val/4.0

		if self.__user_defined_callbacks['attenuation_level_response'] != None:
			self.__user_defined_callbacks['attenuation_level_response'](self, status_byte, count, byteblock, val)
		
		self.attenuation_level = val
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'attenuation_level_response', 'attenuation=' + str()))
	
	def ramp_dwell_time_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x33
		
			DWORD = Time to dwell at each attenuation level during a ramp in 1 
			millisecond intervals
		"""
		
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		
		if self.__user_defined_callbacks['ramp_dwell_time_response'] != None:
			self.__user_defined_callbacks['ramp_dwell_time_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_dwell_time = val
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_dwell_time_response', 'dwell_time=' + str(self.ramp_dwell_time)))
	
	def ramp_dwell_time_bidirectional_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x37
		
			DWORD = Time to dwell at each attenuation level during a ramp in 1 
			millisecond intervals
		"""
		
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		
		if self.__user_defined_callbacks['ramp_dwell_time_bidirectional_response'] != None:
			self.__user_defined_callbacks['ramp_dwell_time_bidirectional_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_dwell_time_bidirectional = val
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_dwell_time_response', 'dwell_time=' + str(self.ramp_dwell_time)))
	
	def ramp_start_attenuation_level_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x30
		
			DWORD = Upper level of the ram in 0.25 dB units
		"""
		
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		val = val/4.0

		if self.__user_defined_callbacks['ramp_start_attenuation_level_response'] != None:
			self.__user_defined_callbacks['ramp_start_attenuation_level_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_start_attenuation_level = val/4.0
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_start_attenuation_level_response', 'attenuation=' + str(self.ramp_start_attenuation_level)))
	
	def ramp_stop_attenuation_level_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x33
		
			DWORD = Upper level of the ram in 0.25 dB units
		"""
		
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		val = val/4.0
		
		if self.__user_defined_callbacks['ramp_stop_attenuation_level_response'] != None:
			self.__user_defined_callbacks['ramp_stop_attenuation_level_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_stop_attenuation_level = val/4.0
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_stop_attenuation_level_response', 'attenuation=' + str(self.ramp_stop_attenuation_level)))
	
	def ramp_step_size_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x33
		
			DWORD = Attenuation step size in 0.25 dB units.
		"""
		
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		val = val/4.0
		
		if self.__user_defined_callbacks['ramp_step_size_response'] != None:
			self.__user_defined_callbacks['ramp_step_size_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_step_size = val
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_step_size_response', 'step_size=' + str(self.ramp_step_size)))
	
	def ramp_step_size_bidirectional_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x37
		
			DWORD = Attenuation step size in 0.25 dB units.
		"""
		
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		val = val/4.0
		
		if self.__user_defined_callbacks['ramp_step_size_bidirectional_response'] != None:
			self.__user_defined_callbacks['ramp_step_size_bidirectional_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_step_size_bidirectional = val
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_step_size_bidirectional_response', 'step_size=' + str(self.ramp_step_size_bidirectional)))

	def ramp_wait_time_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x33
		
			DWORD = Time to wait at the end of each attenuation ramp before 
			beginning another attenuation ramp when the tram is in continuous 
			operation mode. This time is in 1 millisecond intervals.
		"""
	
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		
		if self.__user_defined_callbacks['ramp_wait_time_response'] != None:
			self.__user_defined_callbacks['ramp_wait_time_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_wait_time = val
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_wait_time_response', 'wait_time=' + str(self.ramp_wait_time)))
	
	def ramp_wait_time_bidirectional_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x39
		
			DWORD = Time to wait at the end of each attenuation ramp before 
			beginning another attenuation ramp when the tram is in continuous 
			operation mode. This time is in 1 millisecond intervals.
		"""
	
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		
		if self.__user_defined_callbacks['ramp_wait_time_bidirectional_response'] != None:
			self.__user_defined_callbacks['ramp_wait_time_bidirectional_response'](self, status_byte, count, byteblock, val)
		
		self.ramp_wait_time_bidirectional = val
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'ramp_wait_time_bidirectional_response', 'wait_time=' + str(self.ramp_wait_time)))
	
	def ramp_mode_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x33
		
			Byte = 01 for single ramp, and 02 for continuous ramps.
		"""
		
		self.ramp_mode = byteblock[0]
	
	def maximum_attenuation_response(self, status_byte, count, byteblock, debug=False):
	
		"""	Status byte = 0x35
		
			Byte = The maximum attenuation level provided by the attenuator.
		"""
		
		val = (byteblock[3] << 24) + (byteblock[2] << 16) + (byteblock[1] << 8) + byteblock[0]
		val = val/4.0
			
		if self.__user_defined_callbacks['maximum_attenuation_response'] != None:
			self.__user_defined_callbacks['maximum_attenuation_response'](self, status_byte, count, byteblock, val)
		
		self.maximum_attenuation = val/4.0
		
		if debug:
			self.__log.write(self.__cmd_log_string % ( datetime.now(), 'maximum_attenuation_response', 'attenuation=' + str(self.maximum_attenuation)))
		
	#################################################################################
	#								PERIODIC STATUS									#
	#################################################################################
	
	def status_report(self, status_byte, count, byteblock, debug=False):
		
		"""	Status byte = 0x0E
		
			byteblock[0-3] - Reserved
			
			byteblock[4] : dev_status
			
			1 1 1 X 1 1 1 1 	
			|--------------- PLL lock status bit
			  |------------- A parameter was set since the last "Save Settings" command
			    |----------- A command completed
				  |--------- NOT USED!
			        |------- The RF HW is on
			          |----- 0 for sweep up, 1 for sweep down
			            |--- 1 for continuous sweeping
			              |- 1 for single sweep
			        
			byteblock[5] - Current Attenuation as per get_attenuation
		"""

		val = byteblock[5] / 4.0
		
		if debug:
			self.__log.write(self.__status_log_string % ( datetime.now(), 'status_report', status_byte, byteblock[1], val))
		
		if val != self.attenuation_level:
			if debug:
				self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'attenuation_level_change', val))
		
			if self.__user_defined_callbacks['attenuation_level_change'] != None:
				self.__user_defined_callbacks['attenuation_level_change'](self, status_byte, count, byteblock, val)
		
		self.attenuation_level = val
		
		#PLL Lock
		if (byteblock[4] >> 7) & 0x01 == 1:
			val = 1
			
			if val != self.pll_lock:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'pll_lock_change', val))
			
				if self.__user_defined_callbacks['pll_lock_change'] != None:
					self.__user_defined_callbacks['pll_lock_change'](self, status_byte, count, byteblock, val)
			
			self.pll_lock = val
		elif (byteblock[4] >> 7) & 0x01 == 0:
			val = 0
			
			if val != self.pll_lock:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'pll_lock_change', val))

				if self.__user_defined_callbacks['pll_lock_change'] != None:
					self.__user_defined_callbacks['pll_lock_change'](self, status_byte, count, byteblock, val)
			
			self.pll_lock = val
		
		#Save Settings
		if (byteblock[4] >> 6) & 0x01 == 1:
			val = 1
			
			if val != self.save_required:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'save_required_change', val))

				if self.__user_defined_callbacks['save_required_change'] != None:
					self.__user_defined_callbacks['save_required_change'](self, status_byte, count, byteblock, val)
		
			self.save_required = val
		elif (byteblock[4] >> 6) & 0x01 == 0:
			val = 0
			
			if val != self.save_required:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'save_required_change', val))

				if self.__user_defined_callbacks['save_required_change'] != None:
					self.__user_defined_callbacks['save_required_change'](self, status_byte, count, byteblock, val)
		
			self.save_required = val
		
		#Command Complete
		if (byteblock[4] >> 5) & 0x01 == 1:
			val = 0 #Should something be encoded in val here?
			if debug:
				self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'command_complete', val))

			if self.__user_defined_callbacks['command_complete'] != None:
				self.__user_defined_callbacks['command_complete'](self, status_byte, count, byteblock, val)
		elif (byteblock[1] >> 5) & 0x01 == 0:
			pass
		
		#Bit 5 in dev_status not used.
		
		#RF hardware on
		if (byteblock[4] >> 3) & 0x01 == 1:
			val = 1
			
			if val != self.rf_hardware_state:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'rf_hardware_state_change', val))

				if self.__user_defined_callbacks['rf_hardware_state_change'] != None:
					self.__user_defined_callbacks['rf_hardware_state_change'](self, status_byte, count, byteblock, val)
			
			self.rf_hardware_state = val
		elif (byteblock[4] >> 3) & 0x01 == 0:
			val = 0
			
			if val != self.rf_hardware_state:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'rf_hardware_state_change', val))

				if self.__user_defined_callbacks['rf_hardware_state_change'] != None:
					self.__user_defined_callbacks['rf_hardware_state_change'](self, status_byte, count, byteblock, val)
			
			self.rf_hardware_state = val
			
		#Ramp status
		if (byteblock[4] >> 2) & 0x01 == 1:
			#Ramping down
			val = 1
			
			if val != self.ramp_status:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'ramp_status_change', val))

			if self.__user_defined_callbacks['ramp_state_change'] != None:
				self.__user_defined_callbacks['ramp_state_change'](self, status_byte, count, byteblock, val)
			
			self.ramp_status = val
		elif (byteblock[4] >> 2) & 0x01 == 0:
			#Ramping up
			val = 0
			
			if val != self.ramp_status:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'ramp_status_change', val))

			if self.__user_defined_callbacks['ramp_state_change'] != None:
				self.__user_defined_callbacks['ramp_state_change'](self, status_byte, count, byteblock, val)
			
			self.ramp_status = val
		
		#Sweep mode continuous
		if (byteblock[4] >> 1) & 0x01 == 1:
			val = 2
			
			if val != self.ramp_mode:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'ramp_mode_change', val))

			if self.__user_defined_callbacks['ramp_mode_change'] != None:
				self.__user_defined_callbacks['ramp_mode_change'](self, status_byte, count, byteblock, val)
			
			self.ramp_mode = val
			
		elif (byteblock[4] >> 1) | 0x01 == 0:
			pass
		
		#Sweep mode single
		if (byteblock[4]) & 0x01 == 1:
			val = 1
			
			if val != self.ramp_mode:
				if debug:
					self.__log.write(self.__res_log_string % ( datetime.now(), 'status_report', 'ramp_mode_change', val))
			
			if self.__user_defined_callbacks['ramp_mode_change'] != None:
				self.__user_defined_callbacks['ramp_mode_change'](self, status_byte, count, byteblock, val)
			
			self.ramp_mode = val
		elif (byteblock[4]) & 0x01 == 0:
			pass
	
	
	#################################################################################
	#								CALLBACK VARIABLES								#
	#################################################################################
	
	
	__responses = 	{ 	
						0x04 : attenuation_level_response,
						0x0d : attenuation_level_response,
						0x33 : ramp_dwell_time_response,
						0x37 : ramp_dwell_time_bidirectional_response,
						0x30 : ramp_start_attenuation_level_response,
						0x31 : ramp_stop_attenuation_level_response,
						0x32 : ramp_step_size_response,
						0x38 : ramp_step_size_bidirectional_response,
						0x36 : ramp_wait_time_response,
						0x39 : ramp_wait_time_bidirectional_response,
						0x09 : ramp_mode_response,
						0x0e : status_report
					}
	
	""" User defined callbacks must be of the format:
			callback_function(Attenuator device, int status_byte, int count, 
				bytearray byteblock)
		NOTE: Callback functions are called PRIO to updating the Attenuator object 
		for get commands.
	"""
		
	__user_defined_callbacks = 	{ 
									'attenuation_level_response' : None,
									'ramp_dwell_time_response' : None,
									'ramp_dwell_time_bidirectional_response' : None,
									'ramp_start_attenuation_level_response' : None,
									'ramp_stop_attenuation_level_response' : None,
									'ramp_step_size_response' : None,
									'ramp_step_size_bidirectional_response' : None,
									'ramp_wait_time_response' : None,
									'ramp_wait_time_bidirectional_response' : None,
									'ramp_mode_response' : None,
									'attenuation_level_change' : None,
									'pll_lock_change' : None,
									'save_required_change' : None,
									'command_complete' : None,
									'rf_hardware_state_change' : None,
									'ramp_state_change' : None,
									'ramp_mode_change' : None
								}

	def register_callback(self, event, func):
		""" Register a callback to be called when an event happens 
			Callback function to be in the format of:
			callback_function(Attenuator device, int status_byte, 
				int count, bytearray byteblock) 
		
			NOTE: Callback functions are called PRIO to updating the Attenuator object 
				for get commands. """

		if event in self.__user_defined_callbacks:
			self.__user_defined_callbacks[event] = func
			return True
		else:
			return False

	def deregister_callback(self, event):
		""" De-register a callback for an event """

		if event in self.__user_defined_callbacks:
			self.__user_defined_callbacks[event] = None
			return True
		else:
			return False 
		


from PyDAQmx import *
import matplotlib.pyplot as plt
import numpy as np
import time
from ctypes import *

class NIDAQ(object):
	
	# Constant Values // CHANGE ALL VALUES HERE
	set_sample_rate = 10000
	set_minimum_volt = -5
	set_maximum_volt = 5
	
	def __init__(self):
		# Initializing class values to user preference
		self.sample_rate = NIDAQ.set_sample_rate
		self.min_volt = NIDAQ.set_minimum_volt
		self.max_volt = NIDAQ.set_maximum_volt
		
		# Initializing Channel Names - CHANGE HERE
		chan1 = "Dev1/ai0"
		chan2 = "Dev1/ai1"
		chan3 = "Dev1/ai3"
		
		# Initializing array and AI input
		self.analog_input = Task()
		self.read = int32()
		
		# Channel names to pass to CreateAIVoltageChan
		self.nameof1CH = chan1
		self.nameof2CH = chan1 + ", " + chan2
		self.nameof3CH = chan1 + ", " + chan2 + ", " + chan3
		
	def stream_1ch(self, int_time):
		# This method opens 1 channel and streams that channel
		# Then returns the array of data
		
		# Gather necessary samples space for array
		timeOut = int_time + 100.0
		total_samples = int(self.sample_rate * int_time)
		self.data1 = np.zeros(((total_samples),), dtype=np.float64)
		
		# Open up the stream and gather data
		self.analog_input.CreateAIVoltageChan(self.nameof1CH,"",DAQmx_Val_Cfg_Default,self.min_volt,self.max_volt,DAQmx_Val_Volts,None)
		self.analog_input.CfgSampClkTiming("",int(self.sample_rate),DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,total_samples)
		
		self.analog_input.StartTask()
		
		self.analog_input.ReadAnalogF64(total_samples,timeOut,DAQmx_Val_GroupByChannel,self.data1,total_samples,byref(self.read),None)
		self.analog_input.StopTask()
		self.analog_input.ClearTask()
		self.analog_input = Task()
		
		'''
		# Plotting onto screen
		plt.plot(self.data1)
		plt.show()
		'''
		return self.data1
		
	
	def stream_2ch(self, int_time):
		# This method opens 2 channels and streams the 2
		# Then returns 2 arrays
		
		# Gather necessary size for 2 array
		# Then initialize enough space for 2
		timeOut = int_time + 100.0
		total_samples = int(self.sample_rate * int_time)
		temp_data = np.zeros(((total_samples*2),), dtype=np.float64)
		self.data1 = np.zeros(((total_samples),), dtype=np.float64)
		self.data2 = np.zeros(((total_samples),), dtype=np.float64)

		
		# Open up the stream and gather data
		self.analog_input.CreateAIVoltageChan(self.nameof2CH,"",DAQmx_Val_Cfg_Default,self.min_volt,self.max_volt,DAQmx_Val_Volts,None)
		self.analog_input.CfgSampClkTiming("",int(self.sample_rate),DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,total_samples)
		
		self.analog_input.StartTask()
		
		self.analog_input.ReadAnalogF64(total_samples,timeOut,DAQmx_Val_GroupByChannel,temp_data,total_samples*2,byref(self.read),None)
		
		self.analog_input.StopTask()
		self.analog_input.ClearTask()
		self.analog_input = Task()
		
		# Splitting the data into appropriate arrays
		self.data1 = temp_data[0:len(temp_data)/2]
		self.data2 = temp_data[len(temp_data)/2:]
		
		return self.data1, self.data2
		
	def stream_3ch(self, int_time):
		# This method opens 3 channels and streams the 3
		# Then returns 3 arrays
		
		timeOut = int_time + 100.0
		# Gather necessary size for 3 array
		# Then initialize enough space for 3
		total_samples = int(self.sample_rate * int_time)
		temp_data = np.zeros(((total_samples*3),), dtype=np.float64)
		self.data1 = np.zeros(((total_samples),), dtype=np.float64)
		self.data2 = np.zeros(((total_samples),), dtype=np.float64)
		self.data3 = np.zeros(((total_samples),), dtype=np.float64)
		
		# Open up the stream and gather data
		self.analog_input.CreateAIVoltageChan(self.nameof3CH,"",DAQmx_Val_Cfg_Default,self.min_volt,self.max_volt,DAQmx_Val_Volts,None)
		self.analog_input.CfgSampClkTiming("",int(self.sample_rate),DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,total_samples)
		
		self.analog_input.StartTask()
		
		self.analog_input.ReadAnalogF64(total_samples,timeOut,DAQmx_Val_GroupByChannel,temp_data,total_samples*3,byref(self.read),None)
		
		# Splitting the data stored in temp_data into
		# appropriate arrays
		self.data1 = temp_data[0:len(temp_data)/3]
		self.data2 = temp_data[len(temp_data)/3:(len(temp_data)-(len(temp_data)/3))]
		self.data3 = temp_data[(len(temp_data)-(len(temp_data)/3)):]
		
		self.analog_input.StopTask()
		self.analog_input.ClearTask()
		self.analog_input = Task()
		
		return self.data1, self.data2, self.data3
	
	def average_1ch(self, int_time):
		# Returns average for data1 set

		total = np.mean(self.stream_1ch(int_time))
		
		return total
	
	def average_2ch(self, int_time):
		# Returns 2 averages, one for data1 set
		# and the other for data2 set
		total1, total2 = self.stream_2ch(int_time)
		average1 = np.mean(total1)
		average2 = np.mean(total2)
		
		return average1, average2
		
	def help(self):
		# This method is used to show user all of the
		# available functions for NIDAQ
		
		print("\t Available Functions")
		print("\t -------------------")
		print("\t 1) stream_1ch")
		print("\t 2) stream_2ch")
		print("\t 3) stream_3ch")
		print("\t 4) average_1ch")
		print("\t 5) average_2ch")
		
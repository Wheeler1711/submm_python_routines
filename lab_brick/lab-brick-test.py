import sys
import getopt
import time

from lab_brick.core import Attenuator

def attenuation_change(device, status_byte, count, byteblock, val):
	print "User Defined Callback for attenuation_level_change"
	print "Attenuation: " + str(val)
	
def attenuation_response(device, status_byte, count, byteblock, val):
	print "User Defined Callback for attenuation_level_response"
	print "Attenuation: " + str(val)

if __name__ == '__main__':

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hv:p:", ["help", "vid=", "pid="])
	except getopt.GetoptError as err:
		# print help information and exit:
		print str(err)  # will print something like "option -a not recognized"
		usage()
		sys.exit(2)
		
	vid = -1
	pid = -1
		
	for o, a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-v", "--vid"):
			vid = a
		elif o in ("-p", "--pid"):
			pid = a
			
	if vid == -1 or pid == -1:
		usage()
		sys.exit()
	else:
		try:
			vid = int(vid, 0)
			pid = int(pid, 0)
		except:
			usage()

		att = Attenuator(vid = vid, pid = pid, debug=True)
		#att = Attenuator(vid = vid, pid = pid, debug=False)

		att.register_callback('attenuation_level_response', attenuation_response)
		
		att.get_attenuation(debug=True)
		
		att.register_callback('attenuation_level_change', attenuation_change)

		#att.set_ramp_start_attenuation_level(20)
		time.sleep(0.1)
		#att.set_ramp_stop_attenuation_level(1)
		#att.set_ramp_step_size(1)
		#att.start_ramp_down()

		att.set_ramp_start_attenuation_level(1, debug=True)
		att.set_ramp_stop_attenuation_level(5, debug=True)

		att.set_ramp_step_size(1, debug=True)
		att.set_ramp_dwell_time(500, debug=True)
		att.set_ramp_wait_time(500, debug=True)

		att.set_ramp_dwell_time_bidirectional(500, debug=True)
		att.set_ramp_wait_time_bidirectional(500, debug=True)
		att.set_ramp_step_size_bidirectional(1, debug=True)

		att.start_ramp_continuous_bidirectional(debug=True)

	while 1:
		#att.set_attenuation(5, debug=True)
		time.sleep(10)
		#att.set_attenuation(10, debug=True)
		#time.sleep(10)

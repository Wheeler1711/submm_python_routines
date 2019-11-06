import numpy as np
import time
import datetime

from instruments import anritsu2 as an

anritsu = an.Anritsu()
anritsu.set_power(10.)
ts = time.time()

times = np.array(())
sleep_time =48.0

time.sleep(50)

low_freq = 180.
high_freq =199.75
step = 0.25
mm_freqs = np.arange(low_freq,high_freq,step)

time_dir = "/home/superspec/Documents/data/kidpy_multitone/march2018/lo_chop_times/"+datetime.datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H_%M")

start = time.time()

for freq in mm_freqs:
    print(freq)
    anritsu.set_frequency(freq*10**9/18.)
    times = np.append(times,(time.time()-start))
    anritsu.turn_outputOn()
    time.sleep(sleep_time)
    times = np.append(times,(time.time()-start))
    anritsu.turn_outputOff()
    time.sleep(sleep_time)

#print times
np.save(time_dir,times)

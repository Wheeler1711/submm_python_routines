from submm.multitone_kidPy import analyze

skip_beginning = 10 #skip first few streaming points

# read in fine and gain scans
fine_filename = "1521699164-Mar-22-2018-00-12-44.dir"
gain_filename = "1521699121-Mar-22-2018-00-12-01.dir"
stream_filename = "1521699068-Mar-22-2018-00-11-08.dir"

cal_dict = analyze.calibrate_multi(fine_filename, gain_filename, stream_filename, skip_beginning = skip_beginning)

psd_dict = analyze.noise_multi(cal_dict)

from submm.KIDs import single_tone



st = single_tone.single_tone()

#set some paramaters
st.fine_numpoints = 25
st.gain_numpoints = 25
st.rough_numpoints = 25
st.med_numpoints = 25
st.fine_span = 50*10**6
st.iq_integration_time = 0.1
st.integration_time = 5

#st.set_input_attn(20)
st.take_noise_set(300*10**6,take_noise = False)

single_tone.plot_iq_dict(st.iq_dictionary)

#plot fine i & q w/ powers
#plot i^2 + q^2, frequency w/ powers in legends
st.power_sweep(20,19, 1, st.iq_dictionary['fine_center_freq'], filename = "poop2")
single_tone.plot_power_dict(st.power_dictionary)


'''

st.set_input_attn(6)
st.set_output_attn(0)

freq, i, q = st.iq_sweep(80*10**6, 60*10**6,2)

plt.figure(1)
plt.plot(freq, i**2 + q**2)

plt.figure(2)
plt.plot(i,q)

st.change_attn_balanced(0)
#print st.input_attn_value
#print st.output_attn_value
freq, i, q = st.iq_sweep(80*10**6, 60*10**6,2)

plt.figure(1)
plt.plot(freq, i**2 + q**2)

plt.figure(2)
plt.plot(i,q)

plt.show()


dict_1 = st.power_sweep(32, 31.5, 0.5, 80*10**6)
print dict_1
'''
'''
#st.dictionary = st.take_noise_set(200*10**6)
#single_tone.plot_dict(st.dictionary)
#st.export_file()

#sb = single_tone.import_file("test.txt")
#single_tone.plot_dict(sb)

#st.save_log("log.txt")

'''
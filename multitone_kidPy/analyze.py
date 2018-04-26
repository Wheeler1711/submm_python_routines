from matplotlib.backends.backend_pdf import PdfPages
from multitone_kidPy import read_multitone
from KIDs import resonance_fitting
import matplotlib.pyplot as plt
import numpy as np





# this function fits a fine and gain scan combo produced by the ASU multitone system
def fit_fine_gain(fine_name,gain_name):
        
	fine = read_multitone.read_iq_sweep(fine_name)
	gain = read_multitone.read_iq_sweep(gain_name)
	outfile_dir = fine_name

	pdf_pages = PdfPages(outfile_dir+"/"+"fit_plots.pdf")

	all_fits_mag = np.zeros((8,fine['freqs'].shape[1]))
	all_fits_iq = np.zeros((9,fine['freqs'].shape[1]))

	for i in range(0,fine['freqs'].shape[1]):
		fine_f = fine['freqs'][:,i]*10**6
		gain_f = gain['freqs'][:,i]*10**6
		fine_z = fine['I'][:,i]+1.j*fine['Q'][:,i]
		gain_z = gain['I'][:,i]+1.j*gain['Q'][:,i]

		fig = plt.figure(i,figsize = (12,12))

		plt.subplot(221)
		plt.title("Resonator Index "+str(i))
		plt.plot(fine['freqs'][:,i],10*np.log10(fine['I'][:,i]**2+fine['Q'][:,i]**2),'o',label = "gain")
		plt.plot(gain['freqs'][:,i],10*np.log10(gain['I'][:,i]**2+gain['Q'][:,i]**2),'o',label = "fine")
		plt.xlabel("Frequency (MHz)")
		plt.ylabel("Power (dB)")

		plt.subplot(223)
		plt.plot(fine['freqs'][:,i],10*np.log10(fine['I'][:,i]**2+fine['Q'][:,i]**2),'o')
		plt.plot(gain['freqs'][:,i],10*np.log10(gain['I'][:,i]**2+gain['Q'][:,i]**2),'o')
		plt.xlabel("Frequency (MHz)")
		plt.ylabel("Power (dB)")
		plt.xlim(np.min(fine['freqs'][:,i]),np.max(fine['freqs'][:,i]))

		# fit nonlinear magnitude
		try:
			x0 = resonance_fitting.guess_x0_mag_nonlinear_sep(fine_f,fine_z,gain_f,gain_z,verbose = True)
			fit_dict_mag = resonance_fitting.fit_nonlinear_mag_sep(fine_f,fine_z,gain_f,gain_z,x0=x0)#,bounds =bounds)
			all_fits_mag[:,i] = fit_dict_mag['fit'][0]
			plt.subplot(221)
			plt.plot(fit_dict_mag['fit_freqs']/10**6,10*np.log10(fit_dict_mag['fit_result']),"+",label = "fit")
			plt.plot(fit_dict_mag['fit_freqs']/10**6,10*np.log10(fit_dict_mag['x0_result']),"x",label = "x0 guess")
			plt.legend()
			plt.subplot(223)
			plt.plot(fit_dict_mag['fit_freqs']/10**6,10*np.log10(fit_dict_mag['fit_result']),"+")
			plt.plot(fit_dict_mag['fit_freqs']/10**6,10*np.log10(fit_dict_mag['x0_result']),"x")
		except Exception as e:
			print(e)
			print("could not fit the resonator")


		plt.subplot(222,aspect ='equal')
		plt.plot(fine['I'][:,i],fine['Q'][:,i],'o')
		plt.plot(gain['I'][:,i],gain['Q'][:,i],'o')
		plt.xlabel("I")
		plt.ylabel("Q")

		plt.subplot(224,aspect ='equal')
		plt.plot(fine['I'][:,i],fine['Q'][:,i],'o')
		plt.plot(gain['I'][:,i],gain['Q'][:,i],'o')
		plt.plot(fine['I'][:,i],fine['Q'][:,i])
		plt.xlabel("I")
		plt.ylabel("Q")
		plt.xlim(np.min(fine['I'][:,i]),np.max(fine['I'][:,i]))
		plt.ylim(np.min(fine['Q'][:,i]),np.max(fine['Q'][:,i]))

		# fit nonlinear iq 
		try:
			x0 = resonance_fitting.guess_x0_iq_nonlinear_sep(fine_f,fine_z,gain_f,gain_z,verbose = True)
			fit_dict_iq = resonance_fitting.fit_nonlinear_iq_sep(fine_f,fine_z,gain_f,gain_z,x0=x0)
			all_fits_iq[:,i] = fit_dict_iq['fit'][0]
			plt.subplot(222,aspect ='equal')
			plt.plot(np.real(fit_dict_iq['fit_result']),np.imag(fit_dict_iq['fit_result']),"+")
			plt.plot(np.real(fit_dict_iq['x0_result']),np.imag(fit_dict_iq['x0_result']),"x")
			plt.subplot(224,aspect ='equal')
			plt.plot(np.real(fit_dict_iq['fit_result']),np.imag(fit_dict_iq['fit_result']),"+")
			plt.plot(np.real(fit_dict_iq['x0_result']),np.imag(fit_dict_iq['x0_result']),"x")
		except Exception as e:
			print(e)
			print("could not fit the resonator")






		pdf_pages.savefig(fig)
		plt.close(fig)

	pdf_pages.close()

	pdf_pages = PdfPages(outfile_dir+"/"+"fit_results.pdf")

	fig = plt.figure(1, figsize = (12,6))
	plt.title("Center frequency")
	plt.plot(all_fits_mag[0,:]/10**6,'o',label = "Mag fit")
	plt.plot(all_fits_iq[0,:]/10**6,'o',label = "IQ fit")
	plt.xlabel("resonator index")
	plt.ylabel("Resonator Frequency (MHz)")
	plt.legend(loc = 4)
	pdf_pages.savefig(fig)
	plt.close()

	fig = plt.figure(2,figsize = (12,6))
	plt.title("Resonator Qs")
	plt.plot(all_fits_mag[1,:],'o',label = "Qr Mag",color = 'g')
	plt.plot(all_fits_iq[1,:],'*',label = "Qr IQ",color = 'g')
	plt.plot(all_fits_mag[1,:]/all_fits_mag[2,:],'o',label = "Qc Mag",color = 'b')
	plt.plot(all_fits_iq[1,:]/all_fits_mag[2,:],'*',label = "Qc IQ",color = 'b')
	plt.plot(1/(1/all_fits_mag[1,:]-1/(all_fits_mag[1,:]/all_fits_mag[2,:])),'o',label = "Qi Mag",color = 'r')
	plt.plot(1/(1/all_fits_iq[1,:]-1/(all_fits_iq[1,:]/all_fits_mag[2,:])),'*',label = "Qi IQ",color = 'r')
	plt.xlabel("Resonator index")
	plt.ylabel("Resonator Q")
	plt.yscale('log')
	plt.legend()
	pdf_pages.savefig(fig)
	plt.close()

	fig = plt.figure(3,figsize = (12,6))
	plt.title("Non linearity parameter a")
	plt.plot(all_fits_mag[4,:],'o',label = "a Mag")
	plt.plot(all_fits_iq[4,:],'o',label = "a IQ")
	plt.xlabel("Resonator index")
	plt.ylabel("Non-linearity parameter a")
	plt.ylim(0,1)
	plt.legend()
	pdf_pages.savefig(fig)
	plt.close()

	pdf_pages.close()

	np.save(outfile_dir+"/"+"all_fits_mag",all_fits_mag)
	np.save(outfile_dir+"/"+"all_fits_iq",all_fits_iq)
	np.savetxt(outfile_dir+"/"+"all_fits_mag.csv",all_fits_mag,delimiter = ',')
	np.savetxt(outfile_dir+"/"+"all_fits_iq.csv",all_fits_iq,delimiter = ',')



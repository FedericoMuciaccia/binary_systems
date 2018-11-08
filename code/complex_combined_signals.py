
import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

#numpy.random.seed(1234)

observation_time = 32 # s
sampling_frequency = 512 # Hz
Nyquist_frequency = sampling_frequency / 2
time_resolution = 1/sampling_frequency
time_samples = observation_time * sampling_frequency

signal_frequency = 4 # Hz
time = numpy.linspace(start=0, stop=observation_time, num=time_samples)
signal_omega = 2 * numpy.pi * signal_frequency

Hanford_relative_amplitude = 0.1
Livingston_relative_amplitude = 0.1
# NOTE: un limite per avere una visibilità media del picco apprezzabile è di 0.02 (con un tempo di coerenza di 32 secondi)
# NOTE: aumentare il tempo di coerenza condensa il segnale dentro un bin più piccolo e dunque lo fa svettare molto meglio
# TODO fare caso asimmetrico
# TODO introdurre anche il Doppler per il moto relativo tra detector e sorgente

#time_delay_from_Hanford_to_Livingston = 0#0.05#(1/(4*signal_frequency))
phase_delay_from_Hanford_to_Livingston = +numpy.pi/2#0
time_delay_from_Hanford_to_Livingston = phase_delay_from_Hanford_to_Livingston / signal_omega

Hanford_signal = Hanford_relative_amplitude * numpy.sin(signal_omega * time)
Hanford_noise = numpy.random.normal(size=time_samples)
Hanford_data = Hanford_signal + Hanford_noise

Livingston_signal = Livingston_relative_amplitude * numpy.sin(signal_omega * (time + time_delay_from_Hanford_to_Livingston))
Livingston_noise = numpy.random.normal(size=time_samples)
Livingston_data = Livingston_signal + Livingston_noise

## forcing thoroidal periodicity
#Hanford_data[-1] = Hanford_data[0]
#Livingston_data[-1] = Livingston_data[0]
## TODO capiere perché non mi risolve i problemi di errore numerico, ovvero che  ifft(fft(x)) != x

## subtracting an estimator of the mean value
#Hanford_data = Hanford_data - numpy.median(Hanford_data)
#Livingston_data = Livingston_data - numpy.median(Livingston_data)

complex_data = (Hanford_data + 1j*Livingston_data)/numpy.sqrt(2)

Hanford_Fourier = numpy.fft.fft(Hanford_data)
Hanford_Fourier = numpy.fft.fftshift(Hanford_Fourier)
Hanford_spectrum = numpy.square(numpy.abs(Hanford_Fourier))

Livingston_Fourier = numpy.fft.fft(Livingston_data)
Livingston_Fourier = numpy.fft.fftshift(Livingston_Fourier)
Livingston_spectrum = numpy.square(numpy.abs(Livingston_Fourier))

cross_spectrum = numpy.conjugate(Hanford_Fourier) * Livingston_Fourier

whole_Fourier = numpy.fft.fft(complex_data)
whole_spectrum = numpy.square(numpy.abs(whole_Fourier))
whole_spectrum = numpy.fft.fftshift(whole_spectrum)

frequencies = numpy.fft.fftfreq(time_samples, time_resolution)
frequencies = numpy.fft.fftshift(frequencies)



#spectrum = numpy.abs(cross_spectrum)
#spectrum = numpy.angle(cross_spectrum)
#spectrum = numpy.real(cross_spectrum)
#spectrum = numpy.imag(cross_spectrum)
#spectrum = Hanford_spectrum + Livingston_spectrum
result = 1/2 * (Hanford_spectrum + Livingston_spectrum + 2*numpy.real(1j*cross_spectrum))
# NOTE: whole_spectrum == result anche se con qualche trascurabile errore numerico
#spectrum = result
#spectrum = Hanford_spectrum
spectrum = whole_spectrum





## time delay corresponding to the peaks of the spectrum
#time_delay_estimator = numpy.angle(cross_spectrum)/(2*numpy.pi*frequencies)
#pyplot.scatter(x=time_delay_estimator, y=whole_spectrum, marker='.')
#pyplot.xlabel('time [s]')
#pyplot.show()
#fig = pyplot.figure()
#ax = fig.add_subplot(111, projection='polar')
#c = ax.scatter(numpy.angle(cross_spectrum), numpy.log(abs(cross_spectrum)), c=numpy.angle(cross_spectrum), cmap='hsv', alpha=0.75, marker='.')
#pyplot.show()
## NOTE: la distribuzione di fase del cross-spectrum è perfettamente sferica sul rumore

#fig = pyplot.figure()
#ax = fig.add_subplot(111, projection='polar')
#c = ax.scatter(numpy.angle(cross_spectrum), numpy.log(whole_spectrum), c=numpy.angle(cross_spectrum), cmap='hsv', alpha=0.75, marker='.')
#pyplot.show()
## TODO capire perché lo scatterplot polare del solo noise non è a simmetria sferica
#pyplot.scatter(x=whole_spectrum, y=numpy.angle(cross_spectrum), marker='.')
#pyplot.show()

pyplot.figure(figsize=[10,5])
pyplot.plot(frequencies, spectrum)
pyplot.title('spectral density')
pyplot.xlim([-Nyquist_frequency, Nyquist_frequency])
pyplot.xlim([-5, +5])
#pyplot.ylim([0,1.2e6])
#pyplot.ylim([1e0, 1e7])
pyplot.xlabel('frequency [Hz]')
pyplot.ylabel('power spectral density [1/Hz]') # TODO controllare
#pyplot.savefig('./(h+il)_over_sqrt(2).jpg')
#pyplot.savefig('./h.jpg')
#pyplot.savefig('./l.jpg')
pyplot.show()

# TODO vedere incremento di SNR considerando l'altezza del picco, la mediana del rumore e la deviazione standard robusta del rumore (fare analisi statistica considerando tante realizzazioni random)
# TODO una analisi unica (coerente) sui complessi VS due analisi separate (incoerenti) sui reali
# TODO vedere cosa succede con differenti delay e con differenti ampiezze relative (entrambi magari funzini del tempo)
# NOTE: la frequenza si sposta dai negativi ai positivi a seconda del delay temporale, che quindi può essere un'ottima segnatura per il Doppler (tra l'altro simmetrizzandolo e dunque rendendo forse facile l'amplificazione e filtraggio tramite fft in 2D)
# NOTE: a seconda del valore del delay posso avere contemporaneamente sia la frequenza positiva che la frequenza negativa
# NOTE: il delay temporale dovuto al tempo di volo derivante dalla diversa posizione di cielo non sembra essere influente ed indurre alcuno shift in frequenza
# NOTE: con delay nullo (dunque sorgente a metà tra i due detector) le due frequenze (positiva e negativa) non sono simmetriche in presenza di rumore, mentre lo sono perfettamente in totale assenza di rumore (col rumore il contributo simmetrico della riga di segnale che si super-somma (?) con quello asimmetrico di rumore)
# TODO vedere se la proprietà di assenza di shift a seconda della direzione di cielo dinamica di fatto magari elimina il moto relativo della terra col Doppler giornaliero?
# NOTE: sqrt(2)*H dà lo stesso identico risultato di H+iH
# NOTE: la struttura delle linee di segnale non sembra venire alterata dall'avere delle intensità asimmetriche (rispetto al rumore sbiancato) nei due detector, MA viene alterata la figura di interferenza tra le frequenze positive e negative


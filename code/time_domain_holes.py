
import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

numpy.random.seed(1234)

observation_time = 32 # s
sampling_frequency = 512 # Hz
Nyquist_frequency = sampling_frequency / 2
time_resolution = 1/sampling_frequency
time_samples = observation_time * sampling_frequency

relative_amplitude = 0.1
signal_frequency = 4 # Hz
time = numpy.linspace(start=0, stop=observation_time, num=time_samples)
omega = 2 * numpy.pi * signal_frequency
signal = relative_amplitude * numpy.sin(omega * time)
noise = numpy.random.normal(size=time_samples)

science_ready = numpy.ones(time_samples).astype(bool)
science_ready[(5 < time) & (time < 10)] = 0
science_ready[(20 < time) & (time < 25)] = 0

with_holes = True

if with_holes == True:
    signal = signal * science_ready
    noise = noise * science_ready
    filename = '../media/with_data_holes.jpg'
else:
    filename = '../media/without_holes.jpg'

data = signal + noise

Fourier = numpy.fft.rfft(data)
spectrum = numpy.square(numpy.abs(Fourier))
frequencies = numpy.fft.rfftfreq(time_samples, time_resolution)

# point spread function # TODO OR impulse responde
PSF = numpy.square(numpy.abs(numpy.fft.fftshift(numpy.fft.fft(science_ready)))) # TODO controllare se è giusto che ci sia il quadrato e poi controllare se il valore massimo va normalizzato ad 1
PSF_frequencies = numpy.fft.fftshift(numpy.fft.fftfreq(time_samples, time_resolution))

##########################

fig, [[time_domain, frequency_domain], [point_spread_function, ___]] = pyplot.subplots(nrows=2, ncols=2, figsize=[10, 10])

time_domain.set_title('time domain')
#time_domain.plot(time, data)
time_domain.plot(time, noise, label='noise')
time_domain.plot(time, signal, label='signal')
time_domain.set_xlim([0, observation_time])
time_domain.set_ylim([-5, +5])
time_domain.set_xlabel('time [s]')
time_domain.set_ylabel('strain')
time_domain.legend()

frequency_domain.set_title('frequency domain')
frequency_domain.plot(frequencies, spectrum)
frequency_domain.set_xlim([signal_frequency - 2, signal_frequency + 2])
frequency_domain.set_ylim([0, 1e6])
frequency_domain.set_xlabel('frequency [Hz]')

point_spread_function.set_title('point spread function')
point_spread_function.plot(PSF_frequencies, PSF)
point_spread_function.set_xlim([-2, +2])
point_spread_function.set_xlabel('frequency [Hz]')

pyplot.savefig(filename)
pyplot.show()

##########################

exit()

interesting_region = (-0.25 < PSF_frequencies) & (PSF_frequencies < +0.25)
kernel = PSF[interesting_region] # TODO i margini andrebbero scelti automaticamente
kernel_frequencies = PSF_frequencies[interesting_region]
kernel = kernel/kernel.max() # TODO controllare questa normalizzazione
pyplot.plot(kernel_frequencies, kernel)
pyplot.show()

import scipy.signal
impulse_response = kernel
recorded = spectrum/spectrum.max()
#recorded = signal.convolve(impulse_response, original)
recovered, remainder = scipy.signal.deconvolve(recorded, impulse_response) # TODO la deconvoluzione dovrebbe essere tra due trasformate di Fourier e non tra due spettri
print(recovered, remainder)

pyplot.plot(recovered)
pyplot.ylim([-1e200,+1e200])
pyplot.show()

# cross-correlation # TODO rimuovere il quadrato da entrambi gli oggetti o farlo addirittura sui complessi per vedere se per caso si media il rumore
a = scipy.signal.correlate(recorded, impulse_response, mode='same', method='auto') # TODO fare padding su toro
pyplot.plot(frequencies, a)
pyplot.xlim([signal_frequency - 2, signal_frequency + 2])
pyplot.show()




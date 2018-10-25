
import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

sampling_frequency = 4096 # Hz
time_resolution = 1/sampling_frequency

time_start = 0 # s
time_end = 8 # s
time = numpy.arange(time_start, time_end, time_resolution) # attenzione ai valori finali

signal_amplitude = 1
frequency = 32 # Hz
omega = 2 * numpy.pi * frequency
signal = signal_amplitude * numpy.sin(omega * time)

noise_amplitude = 1
noise = noise_amplitude * numpy.random.normal(size=time.size)

#data = signal
data = noise + signal

#pyplot.plot(time, data)
pyplot.plot(time, noise)
pyplot.plot(time, signal)
pyplot.show()

########################################

derivative = numpy.gradient(data)

pyplot.hist2d(x=data, y=derivative, bins=100)
pyplot.show()

pyplot.scatter(x=data, y=derivative)
pyplot.show()

#########################################

fft = numpy.fft.rfft(data)

modulus = numpy.abs(fft)
phase = numpy.angle(fft)

pyplot.semilogy(modulus)
pyplot.show()

pyplot.plot(phase)
pyplot.show()

pyplot.scatter(phase, numpy.log(modulus))
pyplot.show()

#########################################

x = numpy.linspace(start=-1, stop=1, num=100+1, endpoint=True)
y = numpy.linspace(start=-1, stop=1, num=100+1, endpoint=True)

X, Y = numpy.meshgrid(x, y)

C = X + 1j * Y

modulus = numpy.abs(C)
phase = numpy.angle(C)

pyplot.pcolormesh(x, y, modulus, cmap='gray')
pyplot.show()

pyplot.pcolormesh(x, y, phase, cmap='twilight')
pyplot.show()













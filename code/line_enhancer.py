
import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

image_shape = [128,256]
image = numpy.zeros(shape=image_shape)

noise_image = numpy.random.uniform(size=image_shape)

second = 1
minute = 60 * second
hour = 60 * minute
day = 24 * hour

observation_time = 16 * day
t = numpy.linspace(start=0, stop=observation_time, endpoint=True, num=image_shape[0])
f = numpy.arange(image_shape[1])

signal_amplitude = 10
signal_modulation_frequency = 0.5/day
signal_central_frequency = 50
omega = 2 * numpy.pi * signal_modulation_frequency
sinusoid = signal_amplitude * numpy.sin(omega * t) + signal_central_frequency
#signal = numpy.round(sinusoid).astype(int)

signal_image, x_bins, y_bins = numpy.histogram2d(x=t, y=sinusoid, bins=image_shape, range=[[t.min(), t.max()], [f.min(), f.max()]])
#signal_image = numpy.transpose(signal_image)

pyplot.imshow(signal_image.T, # TODO capire come mai
              origin='lower',
              interpolation='none',
              extent=[t.min()/day, t.max()/day, f.min(), f.max()],
              aspect='auto',
              cmap='gray')
pyplot.ylabel('frequency [Hz]')
pyplot.xlabel('time [days]')
pyplot.show()

spectrogram = noise_image + signal_image

pyplot.imshow(spectrogram.T, # TODO capire come mai
              origin='lower',
              interpolation='none',
              extent=[t.min()/day, t.max()/day, f.min(), f.max()],
              aspect='auto',
              cmap='gray')
pyplot.ylabel('frequency [Hz]')
pyplot.xlabel('time [days]')
pyplot.show()













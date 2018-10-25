
import numpy

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

shape = [1024, 1024]

x, y = numpy.indices(shape)

# y = 1/4 AND 3/4 (esatto)
# y = 1/3 AND 2/3 (spurio perché non rispetta la periodicità coi bordi)
# y = 1/6 AND 3/6 AND 5/6 (esatto)
first_center = [int(shape[0]*1/2), int(shape[1]*1/6)]
second_center = [int(shape[0]*1/2), int(shape[1]*3/6)]
third_center = [int(shape[0]*1/2), int(shape[1]*5/6)]

def distance(x,y,center):
    return numpy.sqrt(numpy.square(x - center[0]) + numpy.square(y - center[1]))

def corona_condition(distance):
    inner_radius = 80
    outer_radius = 100
    return (inner_radius < distance) & (distance < outer_radius)

first_distance = distance(x,y,first_center)
second_distance = distance(x,y,second_center)
third_distance = distance(x,y,third_center)

first_condition = corona_condition(first_distance)
second_condition = corona_condition(second_distance)
third_condition = corona_condition(third_distance)

image1 = first_condition
image2 = numpy.logical_or(first_condition, second_condition)
image3 = numpy.zeros(shape)
image3[numpy.transpose([first_center, second_center]).tolist()] = 1
image4 = numpy.zeros(shape)
image4 = numpy.stack([first_condition, second_condition, third_condition], axis=0).any(axis=0)
image5 = numpy.zeros(shape)
image5[numpy.transpose([first_center, second_center, third_center]).tolist()] = 1

image = image5 # image1

pyplot.imshow(image, origin='lower', interpolation='none')
pyplot.show()

fourier = numpy.fft.fftshift(numpy.fft.fft2(image))

modulus = numpy.abs(fourier)
phase = numpy.angle(fourier)

pyplot.imshow(modulus, origin='lower', interpolation='none')
pyplot.show()

pyplot.imshow(phase, origin='lower', interpolation='none', cmap='twilight')
pyplot.show()

pyplot.imshow(numpy.log(modulus), origin='lower', interpolation='none')
pyplot.colorbar()
pyplot.show()
# NOTE: attenzione che le feature ci sono anche nelle parti tra le barre verticali della griglia



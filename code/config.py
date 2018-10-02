
import numpy

second = 1
minute = 60*second
hour = 60*minute
day = 24*hour

sampling_rate = 512#1024#512#256 # Hz # subsampled from 4096 Hz data
observation_time = 3*day
FFT_length = 1024#4096#2048#8192 # seconds # frequency-domain time binning # TODO Paola usa 512 s
window = 'flat'#'gaussian' # or 'tukey' or 'flat'

time_resolution = 1/sampling_rate # s # time-domain time binning

Nyquist_frequency = sampling_rate/2 # Hz

# TODO vedere se esiste già la funzione implementata in numpy
def round_to_power_of_two(x):
    # FFT needs a data number that is a power of 2 to be efficiently computed
    exponent = numpy.log2(x)
    rounded_exponent = numpy.ceil(exponent)
    return numpy.power(2, rounded_exponent)

rounded_observation_time = int(round_to_power_of_two(observation_time)) # seconds (a potenze di 2: circa 6, 12, 24, 48 giorni)

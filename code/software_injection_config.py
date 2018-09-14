
import config

signal_starting_frequency = 96 # Hz
signal_modulation_frequency = 1/(0.5*config.day) # Hz # TODO binarie con modulazione ogni 0.5 giorni fino a 2 giorni
modulation_amplitude = 10 # TODO capire il problema dell'ampiezza che sembra troppo grande rispetto a ciò che si vede nello spettrogramma

## spindown = df/dt
#random_multiplier = 9*numpy.random.rand()+1 # uniform from 1 to 10
#signal_spindown = random_multiplier*-1e-9 # uniform from -10^-9 (small) to -10^-8 (big)

noise_amplitude = 1#1.5e-5 # deve dare 1e-6  # TODO hardcoded # TODO check normalizzazione

signal_scale_factor = 0.005#0.1 # from 0.001 to 0.005 (0.1 is a huge signal) # TODO hardcoded
signal_amplitude = signal_scale_factor*noise_amplitude




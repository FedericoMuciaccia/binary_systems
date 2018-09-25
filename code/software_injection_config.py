
import config

signal_starting_frequency = 96 # Hz
signal_modulation_frequency = 3/config.rounded_observation_time#1/(0.5*config.day) # Hz # TODO binarie con modulazione di un ciclo ogni 0.5 giorni, fino a uno ogni 2 giorni
modulation_amplitude = 0.005 # Hz

## spindown = df/dt
#random_multiplier = 9*numpy.random.rand()+1 # uniform from 1 to 10
#signal_spindown = random_multiplier*-1e-9 # uniform from -10^-9 (small) to -10^-8 (big)
#signal_spindown = -5 * 1e-9

noise_amplitude = 1#1.5e-5 # deve dare 1e-6  # TODO hardcoded # TODO check normalizzazione

signal_scale_factor = 0.05#0.005#0.1 # from 0.001 to 0.005 (0.1 is a huge signal) # TODO hardcoded
signal_amplitude = signal_scale_factor*noise_amplitude




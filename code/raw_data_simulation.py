
# Copyright (C) 2018  Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy
#import xarray

import tensorflow as tf

import config
import software_injection_config

tf.enable_eager_execution() # TensorFlow eager mode

print('sampling rate:', config.sampling_rate, 'Hz')

image_time_start = 0.0 # seconds # TODO hardcoded TODO utile nel caso di real noise
#image_time_interval = round_to_power_of_two(config.observation_time) # seconds (a potenze di 2: circa 6, 12, 24, 48 giorni)
#rounded_observation_time = int(image_time_interval)
print('(rounded) observation time:', '2^{}'.format(int(numpy.log2(config.rounded_observation_time))), 'seconds', '=', config.rounded_observation_time, 'seconds', '=~', numpy.around(config.rounded_observation_time/config.day, 2), 'days')
image_time_stop = image_time_start + config.rounded_observation_time

#number_of_time_data = len(t)
number_of_time_data = int(config.rounded_observation_time * config.sampling_rate)

#t = tf.linspace(start=image_time_start, stop=image_time_stop, num=image_time_interval*time_sampling_rate + 1) # last value included

#white_noise = software_injection_config.noise_amplitude*numpy.random.normal(size=number_of_time_data).astype(numpy.float32) # gaussian noise around 0
white_noise = tf.random_normal(shape=[number_of_time_data,], mean=0, stddev=software_injection_config.noise_amplitude).numpy()#.eval() # float32
# TODO la generazione di numeri random su numpy è molto lenta perché sembra essere single-core

## spindown = df/dt
#random_multiplier = 9*numpy.random.rand()+1 # uniform from 1 to 10
#signal_spindown = random_multiplier*-1e-9 # uniform from -10^-9 (small) to -10^-8 (big)

# TODO NOTE: con signal_scale_factor = 0.004 e observation_time di 3 giorni il doppio corno (informazione integrata) si vede abbastanza bene e l'immagine sinusoidale risulta appena visibile in scala log (ma con certezza)
# TODO NOTE: usando la finestra Tukey il doppio corno peggiora ma l'immagine sinuidale a bassissime ampiezze migliora leggermente e si riesce a vedere con certezza fino a signal_scale_factor = 0.003 (dunque tutto il gioco sta nel concentrare opportunamente tutta la potenza del segnale: provare a farlo anche con delle deconvoluzioni)
# TODO deconvolvere lo spettro contenente anche le frequenze negative e usare le condizioni periodiche a contorno per implementare la topologia toroidale della FFT

t = numpy.arange(start=image_time_start, stop=image_time_stop, step=config.time_resolution, dtype=numpy.float64) # NOTE: float64 is needed to guarantee enough time resolution
# NOTE: the number of values is a power of 2

def signal_waveform():
    # TODO dato che pesa molto in memoria, generare il segnale e fare l'FFT direttamente in chunks
    
    # signal = amplitude*exp(i phi(t))
    # phi(t) = integrate_0^t{omega(tau) d_tau}
    # omega = 2*pi*frequency
    # df/dt = s # linear (first order) spindown
    # f(t) = f_0 + s*t
    # phi(t) = integrate_0^t{2*pi*(f_0+s*tau) d_tau}
    # ==> phi(t) = 2*pi*(f_0*t + (1/2)*s*t^2 + C) # TODO capire perché è necessario mettere 'modulo 2 pi' nell'esponenziale complesso
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*signal_starting_frequency*t).astype(numpy.float32) # pure sinusoid # TODO CAPIRE perché la versione con la sola sinusoide pura è micidialmente lenta
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*(signal_starting_frequency + (1/2)*signal_spindown*t)*t).astype(numpy.float32) # sinusoid with spindown
    
    # signal = real(amplitude*exp(i phi(t)))
    # omega(t) = 2*pi*nu(t)
    # nu(t) = nu_0 + modulation_amplitude*cos(2*numpy.pi*modulation_frequency*t)
    # phi(t) = integrate_0^t{2*pi*nu(t) d_tau}
    # phi(t) = 2*pi*(f_0*t + (modulation_amplitude*(sin(2*numpy.pi*modulation_frequency*t)))/modulation_frequency + C)
    
    signal = software_injection_config.signal_amplitude*numpy.sin(2*numpy.pi*(software_injection_config.signal_starting_frequency*t + (software_injection_config.modulation_amplitude*(numpy.cos(2*numpy.pi*software_injection_config.signal_modulation_frequency*t)))/(2*numpy.pi*software_injection_config.signal_modulation_frequency)))
    # TODO usando invece l'esponenziale complesso serve mettere 'modulo 2 pi'
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*(signal_starting_frequency*t))
    #signal[:int(1/4*len(t))] = 0
    #signal[int(3/4*len(t)):] = 0
    return signal.astype(numpy.float32)

signal = signal_waveform() # TODO abbastana lento

time_data = white_noise + signal

numpy.save('./time_data.npy', time_data)

######################

make_plot = True

if make_plot is True:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib import pyplot
    matplotlib.rcParams.update({'font.size': 23}) # il default è 10 # TODO attenzione che fa l'override di tutti i settaggi precedenti
    
    time_slice = range(0, 2*config.sampling_rate) # 2 seconds
    fig = pyplot.figure(figsize=[12,8])#, dpi=150)
    ax = fig.subplots()
    zooming_factor = 50#100
    ax.set_title('injected signal (displayed {}x) and gaussian white noise\nR = {}'.format(zooming_factor, software_injection_config.signal_scale_factor))#, y=1.02)
    ax.plot(t[time_slice], white_noise[time_slice]) # TODO senza logy?
    ax.plot(t[time_slice], zooming_factor*signal[time_slice])
    #pyplot.xticks([0,config.sampling_rate,2*config.sampling_rate],[0,1,2]) # 2 seconds: 1 second before and 1 second after
    ax.xaxis.set_major_locator(pyplot.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(pyplot.MultipleLocator(1))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('normalized strain')
    pyplot.savefig('../media/white_noise_and_injected_signal.jpg')#, bbox_inches='tight')
    pyplot.savefig('../media/white_noise_and_injected_signal.svg')#, bbox_inches='tight')
    pyplot.show()
    pyplot.close()
    # NOTE: the y scale is linear: the noise is gaussian around 0






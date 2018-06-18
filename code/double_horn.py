
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

# TODO file già tutto incluso nella relazione

import numpy
import xarray
import pandas

import matplotlib
#matplotlib.use('SVG') # così non funziona pyplot.show()
from matplotlib import pyplot

import tensorflow as tf

import config

session = tf.InteractiveSession() # TODO usare la nuova greedy mode di TensorFlow

matplotlib.rcParams.update({'font.size': 25}) # il default è 10 # TODO attenzione che fa l'override di tutti i settaggi precedenti

second = 1
minute = 60*second
hour = 60*minute
day = 24*hour

signal_starting_frequency = 96 # Hz # TODO hardcoded

# TODO processi da questionare:
# mediana VS media (suo stimatore e varianza/errore)
# migliore finestra per segnali variabili ad una data scala
# interallacciamento multiplo
# tracking della sinusoide e utilizzo di questa informazione invece della semplice somma incoerente
# coincidenze
# aliasing
# miglior complomesso sul tempo di coerenza
# buchi non-science
# deconvoluzione per annullare il fringing della finestra
# calcolo parallelo ed out-of-core su GPU

# TODO calcolare out-of-core il segnale generato (per ogni chunk) e farlo per tutti e 131 i segnali di Paola con la formula corretta per la generazione del segnale. calcolare la FFT su GPU in modo da velocizzare la procedura. arrivare a sampling rate di 4096 Hz e replicare i risultati di Paola col tempo di coerenza 512 Hz e vedere cosa succede aumentandolo. analizzare i problemi derivanti dalla scelta della finestra con un dato tempo di coerenza (a seconda di quanto è il massimo df/df localmente).


## spindown = df/dt
#random_multiplier = 9*numpy.random.rand()+1 # uniform from 1 to 10
#signal_spindown = random_multiplier*-1e-9 # uniform from -10^-9 (small) to -10^-8 (big)

time_sampling_rate = config.sampling_rate # Hz
print('sampling rate:', time_sampling_rate, 'Hz')
time_resolution = 1/time_sampling_rate # s # time-domain time binning

def round_to_power_of_two(x):
    # FFT needs a data number that is a power of 2 to be efficiently computed
    exponent = numpy.log2(x)
    rounded_exponent = numpy.ceil(exponent)
    return numpy.power(2, rounded_exponent)

image_time_start = 0.0 # seconds
image_time_interval = config.observation_time # seconds (a potenze di 2: circa 6, 12, 24, 48 giorni)
image_time_interval = round_to_power_of_two(image_time_interval)
print('observation time:', int(image_time_interval/day), 'days')
image_time_stop = image_time_start + image_time_interval

t = numpy.arange(start=image_time_start, stop=image_time_stop, step=time_resolution, dtype=numpy.float64)
# NOTE: float64 is needed to guarantee enough time resolution

# TODO dato che pesa molto in memoria, generare il segnale e fare l'FFT rirettamente in chunks

number_of_time_data = len(t)

#t = tf.linspace(start=image_time_start, stop=image_time_stop, num=image_time_interval*time_sampling_rate + 1) # last value included

# TODO trovare il valore corretto usando a ritroso la distribuzione chi quadro con due gradi di libertà
noise_amplitude = 1#1.5e-5 # deve dare 1e-6  # TODO hardcoded # TODO check normalizzazione
#white_noise = noise_amplitude*numpy.random.randn(t.size).astype(numpy.float32) # gaussian noise around 0
white_noise = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude).eval() # float32

signal_scale_factor = 0.1 # from 0.001 to 0.005 (0.1 is a huge signal) # TODO hardcoded
signal_amplitude = signal_scale_factor*noise_amplitude

make_plot = True

def signal_waveform(t):
    # signal = exp(i phi(t))
    # phi(t) = integrate_0^t{omega(tau) d_tau}
    # omega = 2*pi*frequency
    # df/dt = s # linear (first order) spindown
    # f(t) = f_0 + s*t
    # ==> phi(t) = integrate_0^t{2*pi*(f_0+s*tau) d_tau}
    # ==> phi(t) = 2*pi*(f_0*t + (1/2)*s*t^2 + C) # TODO capire perché è necessario mettere 'modulo 2 pi' nell'esponenziale complesso
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*signal_starting_frequency*t).astype(numpy.float32) # pure sinusoid # TODO CAPIRE perché la versione con la sola sinusoide pura è micidialmente lenta
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*(signal_starting_frequency + (1/2)*signal_spindown*t)*t).astype(numpy.float32) # sinusoid with spindown
    signal = signal_amplitude*numpy.sin(2*numpy.pi*(signal_starting_frequency*t + 10*numpy.cos(2*numpy.pi*(1/(0.5*day))*t))).astype(numpy.float32) # TODO binarie con modulazione ogni 0.5 giorni fino a 2 giorni
    # TODO usando invece l'esponenziale complesso serve mettere 'modulo 2 pi'
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*(signal_starting_frequency*t))
    return signal

signal = signal_waveform(t) # TODO abbastana lento

del t # TODO scrivere funzione in modo che ci sia il garbage collector automaticamente

if make_plot is True:
    time_slice = range(0, 2*time_sampling_rate) # 2 seconds
    pyplot.figure(figsize=[15,10])
    pyplot.title('injected signal (displayed 10x) and gaussian white noise\nR = {}'.format(signal_scale_factor), y=1.02)
    pyplot.plot(white_noise[time_slice]) # TODO senza logy?
    pyplot.plot(10*signal[time_slice])
    pyplot.xticks([0,time_sampling_rate,2*time_sampling_rate],[0,1,2]) # 2 seconds: 1 second before and 1 second after
    pyplot.xlabel('time [s]')
    pyplot.ylabel('normalized strain')
    pyplot.savefig('../media/white_noise_and_injected_signal.jpg', bbox_inches='tight')
    pyplot.savefig('../media/white_noise_and_injected_signal.svg', bbox_inches='tight')
    pyplot.show()
    pyplot.close()
    # NOTE: the y scale is linear: the noise is gaussian around 0

FFT_length = config.FFT_length # s # frequency-domain time binning
print('coherence time:', FFT_length, 's')
Nyquist_frequency = time_sampling_rate/2 # 128 Hz
number_of_time_values_in_one_FFT = FFT_length*time_sampling_rate
unilateral_frequencies = numpy.linspace(0, Nyquist_frequency, int(number_of_time_values_in_one_FFT/2 + 1)) # TODO float32 or float64 ?
frequency_resolution = 1/FFT_length

def flat_top_cosine_edge_window(window_length = number_of_time_values_in_one_FFT):
    # 'flat top cosine edge' window function (by Sergio Frasca)
    # structure: [ascending_cosine, flat, flat, descending_cosine]

    half_length = int(window_length/2)
    quarter_length = int(window_length/4)
    
    index = numpy.arange(window_length)
    
    # TODO valutare la Hamming window e/o quella cos^2 (in modo che interallacciata sommi sempre a 1)
            
    # sinusoidal part at the edges
    factor = 0.5 - 0.5*numpy.cos(2*numpy.pi*index/half_length)
    # flat part in the middle
    factor[quarter_length:window_length-quarter_length] = 1
    
    # TODO attenzione all'ultimo valore:
    # factor[8191] non è 0
    # (perché dovrebbe esserlo invece factor[8192], ma che è fuori range)
    
    # calcolo delle normalizzazione necessaria per tenere in conto della potenza persa nella finestra
    # area sotto la curva diviso area del rettangolo totale
    # se facciamo una operazione di scala (tanto il rapporto è invariante) si capisce meglio
    # rettangolo: [x da 0 a 2*pi, y da 0 a 1]
    # area sotto il seno equivalente all'integrale del seno da 0 a pi
    # integrate_0^pi sin(x) dx = -cos(x)|^pi_0 = 2
    # area sotto il flat top: pi*1
    # dunque area totale sotto la finestra = 2+pi
    # area del rettangolo complessivo = 2*pi*1
    # potenza persa (rapporto) = (2+pi)/2*pi = 1/pi + 1/2 = 0.818310
    # fattore di riscalamento = 1/potenza_persa = 1.222031
    # TODO questo cacolo è corretto? nel loro codice sembra esserci un integrale numerico sui quadrati
    # caso coi quadrati:
    # integrate sin^2 from 0 to pi = x/2 - (1/4)*sin(2*x) |^pi_0 = 
    # = pi/2
    # dunque (pi/2 + pi)/2*pi = 3/4
    
    return factor.astype(numpy.float32)

window = flat_top_cosine_edge_window()

if make_plot is True:
    fig = pyplot.figure(figsize=[15,10])
    ax = pyplot.subplot()
    ax.set_title('flat top cosine edge window', y=1.02)
    time_values = numpy.arange(0, FFT_length, time_resolution) # TODO manca l'ultimo punto corrispondente a FFT_length
    ax.plot(time_values, window)
    ax.xaxis.set_major_locator(pyplot.MultipleLocator(FFT_length/4))
    ax.set_xlabel('time [s]')
    #fig.tight_layout()
    pyplot.savefig('../media/flat_top_cosine_edge_window.jpg', bbox_inches='tight')
    pyplot.savefig('../media/flat_top_cosine_edge_window.svg', bbox_inches='tight')
    pyplot.show()
    pyplot.close()

def make_chunks(time_data, windowed = False):
    # TODO prevedere anche la possibilità di interlacciamento multiplo e non solo a 1/2 (esempio: 7/8)
    number_of_chunks = int(number_of_time_data/number_of_time_values_in_one_FFT)
    chunks = numpy.split(time_data, number_of_chunks)
    time_shift = int(number_of_time_values_in_one_FFT/2)
    # TODO ottimizzare il codice e magari farlo con una funzione rolling
    middle_chunks = numpy.split(time_data[time_shift:-time_shift], number_of_chunks-1)
    middle_chunks.append(numpy.zeros_like(chunks[0])) # dummy empty chunk to be removed later # TODO far in modo che tutto sia sempre in potenze di 2
    # join and reorder odd and even chunks
    interlaced_chunks = numpy.transpose([chunks, middle_chunks], axes=[1,0,2]).reshape([2*number_of_chunks, -1])
    # TODO buttare l'ultimo chunk farlocco
    if windowed is False:
        return interlaced_chunks
    if windowed is True:
        window = flat_top_cosine_edge_window()
        windowed_interlaced_chunks = interlaced_chunks*window
        return windowed_interlaced_chunks

time_data = white_noise + signal

del white_noise
del signal # TODO scrivere funzione in modo che ci sia il garbage collector automaticamente

def make_whitened_spectrogram(time_data): # the fast Fourier transform needs power on two to be fast
    windowed_interlaced_chunks = make_chunks(time_data, windowed = True)
    print('Fourier transform')
    unilateral_fft_data = numpy.fft.rfft(windowed_interlaced_chunks).astype(numpy.complex64) # the one-dimensional real fft is done on the innermost dimension of the array. the results are the unilateral frequencies, from 0 to the Nyquist frequency
#unilateral_fft_data = numpy.fft.rfftn(windowed_interlaced_chunks, axes=[-1])
    #unilateral_fft_data = list(map(numpy.fft.rfft, windowed_interlaced_chunks))
    #unilateral_fft_data.pop()
    #unilateral_fft_data = numpy.array(unilateral_fft_data).astype(numpy.complex64)
    
    ## TODO fare l'FFT su GPU con TensorFlow
    #tensorflow_unilateral_fft_data = tf.spectral.rfft(windowed_interlaced_chunks).eval().astype(numpy.complex64) # TODO dà problemi di memoria
    
    # TODO vedere normalizzazione per la potenza persa
    spectra = numpy.square(numpy.abs(unilateral_fft_data)) # TODO sqrt(2), normd, normw, etc etc
    # TODO normd (normalizzare sul numero di dati)
    spectrogram = numpy.transpose(spectra[0:-1]) # remove the last dummy empty chunk
    whitened_spectrogram = spectrogram/numpy.median(spectrogram)
    # TODO in realtà facendo il whitening col periodogramma si eliminano tutti quei picchi che occupano diversi bin di frequenza, dunque si ripulirebbe l'effetto di allargamento sporco dovuto alla finestra della FFT
    # TODO fare plot zoomato attorno alla riga, in modo da vedere i ghost laterali prima e dopo lo sbiancamento
    return whitened_spectrogram # TODO valutare se inserire un dato fittizio per arrivare ad una potenza di 2

whitened_spectrogram = make_whitened_spectrogram(time_data)

frequency_pixels, time_pixels = whitened_spectrogram.shape
image_time_axis = pandas.date_range('2018-01-01', periods=time_pixels, freq='{}s'.format(int(FFT_length/2)))
#image_time_axis = numpy.arange(0, image_time_interval - FFT_length/2, FFT_length/2) # TODO perché attualmente stiam interlacciando solo di metà
image_frequency_axis = unilateral_frequencies

coordinate_names = ['frequency', 'time']
coordinate_values = [image_frequency_axis, image_time_axis]

whitened_spectrogram = xarray.DataArray(data=whitened_spectrogram, 
                                        dims=coordinate_names, 
                                        coords=coordinate_values)

averaged_spectrum = whitened_spectrogram.mean(dim='time')
#averaged_spectrum = numpy.mean(whitened_spectrogram, axis=1) # TODO media VS mediana? # TODO perché qui la mediana fa molto più schifo della media? # TODO mediana dello spettro VS spettro autoregressivo ?
# TODO usando la mediana le due corna spariscono completamente!
frequency_slice = slice(signal_starting_frequency - 0.02, signal_starting_frequency + 0.02) # TODO hardcoded
zoomed_whitened_spectrogram = whitened_spectrogram.sel(frequency=frequency_slice)
zoomed_averaged_spectrum = averaged_spectrum.sel(frequency=frequency_slice)

if make_plot is True:
    fig = pyplot.figure(figsize=[15,10])
    ax = pyplot.subplot()
    ax.set_title('averaged frequency spectrum', y=1.02)
    ax.semilogy(unilateral_frequencies, averaged_spectrum) # grafico obbligatoriamente col logaritmo in y
    ax.xaxis.set_major_locator(pyplot.MultipleLocator(32))
    #pyplot.xticks((unilateral_frequencies.size*numpy.arange(5)/4).astype(int), (Nyquist_frequency*numpy.arange(5)/4).astype(int))
    #pyplot.vlines(x=8192, ymin=1e-9, ymax=1e1, color='orange', label='1 Hz')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylim(1e-1, 1e3)
    #pyplot.legend(loc='lower right', frameon=False)
    pyplot.savefig('../media/averaged_spectrum.jpg', bbox_inches='tight')#, dpi=300)
    pyplot.show()
    pyplot.close()
    # TODO WISHLIST: spectrogram[all,1]

#frequency_values_in_one_Hz = FFT_length # 1/frequency_resolution = FFT_length
#middle_index = frequency_values_in_one_Hz*signal_starting_frequency
#half_zooming_window = int(numpy.ceil(frequency_values_in_one_Hz*0.02)) # 0.02 Hz # TODO hardcoded
#frequency_range = slice(middle_index-half_zooming_window,middle_index+half_zooming_window)

if make_plot is True:
    fig = pyplot.figure(figsize=[15,10])
    ax = pyplot.subplot()
    ax.set_title('zoomed averaged frequency spectrum', y=1.02)
    #ax.semilogy(unilateral_frequencies[frequency_range], averaged_spectrum[frequency_range]) # grafico obbligatoriamente col logaritmo in y
    ax.semilogy(zoomed_averaged_spectrum.frequency, zoomed_averaged_spectrum)
    ax.xaxis.set_major_locator(pyplot.MultipleLocator(0.01))
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylim(1e-1, 1e3)
    pyplot.savefig('../media/zoomed_averaged_spectrum.jpg', bbox_inches='tight')
    pyplot.savefig('../media/double_horn/coherence_{}s_observation_{}d_sampling{}Hz.jpg'.format(FFT_length, int(image_time_interval/day), time_sampling_rate), bbox_inches='tight')
    pyplot.show()
    pyplot.close()
# TODO fare deconvoluzione per accentuare e stringere il picco?

#image = whitened_spectrogram[frequency_range]
image = zoomed_whitened_spectrogram
normalized_image = numpy.log10(image)





pyplot.figure(figsize=[8,12])
plot = normalized_image.plot(cmap='gray')
pyplot.title('whitened spectrogram (log10 values)', y=1.02)
pyplot.xlabel('time [days]')
pyplot.ylabel('frequency [Hz]')
pyplot.xticks(rotation='45')
plot.colorbar.set_label('log10 values') # TODO diminuire la colorbar (shrink=0.5 ?)
pyplot.tight_layout()
pyplot.savefig('../media/white_noise_image_with_signal_scale_factor_{}.jpg'.format(signal_scale_factor), bbox_inches='tight')
pyplot.savefig('../media/double_horn/spectrogram_scale_factor_{}_coherence_{}s_observation_{}d_sampling{}Hz.jpg'.format(signal_scale_factor, FFT_length, int(image_time_interval/day), time_sampling_rate), bbox_inches='tight')
pyplot.show()
pyplot.close()





def compute_peakmap(image):
    copied_image = image.copy() # here without log10
    # TODO controllare i bordi
    maxima = numpy.array([numpy.convolve(numpy.sign(numpy.diff(copied_image, axis=0))[:,i], [-1,1]) for i in range(image.shape[1])]).T == 2
    # TODO sarebbe forse più corretto prendere i massimi locali lungo la direzione dello spindown, invece che in quella orizzontale della finestra
    under_threshold = copied_image < 2.5 # TODO provare poi a mettere la soglia a 2 per fare il confronto con la Hough per i segnali normalmente sottosoglia
    maxima[under_threshold] = 0
    return maxima.astype(int)
    # TODO line detector/enhancer usando la trasformata wavelet sull'immagine alla scala attesa (oppure )

peakmap = image.copy()
peakmap.data = compute_peakmap(image)

pyplot.figure(figsize=[8,12])
ax.set_title('whitened spectrogram (log10 values)', y=1.02)
plot = peakmap.plot(cmap='gray_r')
pyplot.title('peakmap', y=1.02)
pyplot.xlabel('time [days]')
pyplot.ylabel('frequency [Hz]')
pyplot.xticks(rotation='45')
plot.colorbar.remove()
pyplot.tight_layout()
pyplot.savefig('./white_noise_peakmap_with_signal_scale_factor_{}.jpg'.format(signal_scale_factor), bbox_inches='tight')
pyplot.show()
pyplot.close()


# TODO fare e plottare anche lo "spettro" medio ottenuto dalla peakmap


################################



exit()











fig = pyplot.figure(figsize=[10,10*256/148])
ax = pyplot.subplot()
ax.set_title('whitened spectrogram (log10 values)', y=1.02)
ax.imshow(numpy.log10(image), origin="lower", interpolation="none", cmap='gray') # TODO vedere se normalizzare i dati invece che col logaritmo con la funzione y = 1 - exp(-x) in modo da avere un riferimento assoluto dell'intervallo delle z, ovvero sempre compreso tra 0 e 1




ax.set_yticks(frequency_range, unilateral_frequencies[frequency_range])
ax.yaxis.set_major_locator(pyplot.MultipleLocator(1))






pyplot.xlabel('time [days]')
pyplot.ylabel('frequency [Hz]')

#labelytick = [80.000, 80.008,  80.016 ,  80.023,  80.031]#[0,23,46,69,92,115,138]
#posyTick = [0,64,128,192,256]#[0,20,40,60,80,100,120]
#pyplot.yticks(posyTick,labelytick)

labelxtick = [0,1,2,3,4,5,6]
posxTick = [0,21,42,63,84,105,126]
pyplot.xticks(posxTick,labelxtick)

# TODO mettere linea a time_index = 60 per far capire dove si posiziona il plot successivo
#pyplot.colorbar()
pyplot.savefig('./white_noise_image_with_signal_scale_factor_{}.jpg'.format(signal_scale_factor), bbox_inches='tight')
pyplot.show()
pyplot.close()




pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(compute_peakmap(image), origin="lower", interpolation="none", cmap='gray_r')
pyplot.title('peakmap', y=1.02)
pyplot.xlabel('time index')
pyplot.ylabel('frequency index')
#pyplot.show()
pyplot.savefig('./white_noise_peakmap_with_signal_scale_factor_{}.jpg'.format(signal_scale_factor), bbox_inches='tight')
pyplot.close()
# NOTA: per segnali giganteschi appare correttamente la regione di svuotamento attorno al segnale




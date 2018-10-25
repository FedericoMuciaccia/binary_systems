
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
import xarray
import pandas

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

import scipy.signal

#import tensorflow as tf

import config
import software_injection_config

#session = tf.InteractiveSession() # TODO usare la nuova greedy mode di TensorFlow
#tf.enable_eager_execution() # TensorFlow greedy mode

#matplotlib.rcParams.update({'font.size': 25}) # il default è 10 # TODO attenzione che fa l'override di tutti i settaggi precedenti

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

print('coherence time:', config.FFT_length, 's')
number_of_time_values_in_one_FFT = config.FFT_length*config.sampling_rate
unilateral_frequencies = numpy.linspace(0, config.Nyquist_frequency, int(number_of_time_values_in_one_FFT/2 + 1)) # TODO float32 or float64 ?
frequency_resolution = 1/config.FFT_length

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
    
    return factor.astype(numpy.float32) # TODO attenzionare dtype

def generate_window():
    print('window function:', config.window)
    # TODO vedere finestra che minimizza lo spectral leakage (per aumentare la visibilità dell'immagine)
    
    if config.window == 'flat':
        window = numpy.ones(number_of_time_values_in_one_FFT)
    elif config.window == 'tukey':
        window = flat_top_cosine_edge_window()
    elif config.window == 'gaussian':
        window = scipy.signal.windows.gaussian(number_of_time_values_in_one_FFT,
                                               number_of_time_values_in_one_FFT/8) # ±4 sigma inside the gaussian window
    else:
        print('Error: unknown window function')
        exit()
    
    make_plot = True
    
    if make_plot is True:
        fig = pyplot.figure(figsize=[6,4])
        ax = pyplot.subplot()
        ax.set_title('window function', y=1.02)
        time_values = numpy.arange(0, config.FFT_length, config.time_resolution) # TODO manca l'ultimo punto corrispondente a FFT_length
        ax.plot(time_values, window)
        ax.xaxis.set_major_locator(pyplot.MultipleLocator(config.FFT_length/4))
        ax.set_xlabel('time [s]')
        #fig.tight_layout()
        pyplot.savefig('../media/window_function.jpg', bbox_inches='tight')
        pyplot.savefig('../media/window_function.svg', bbox_inches='tight')
        pyplot.show()
        pyplot.close()
    
    return window

window = generate_window()

def make_chunks(time_data, stride=config.FFT_length/2):
    # TODO prevedere anche la possibilità di interlacciamento multiplo e non solo a 1/2 (esempio: 3/4 o 7/8)
    number_of_time_data = len(time_data) # TODO duplicato
    number_of_chunks = int(number_of_time_data/number_of_time_values_in_one_FFT)
    chunks = numpy.split(time_data, number_of_chunks)
    time_shift = int(number_of_time_values_in_one_FFT/2)
    # TODO ottimizzare il codice e magari farlo con una funzione rolling
    middle_chunks = numpy.split(time_data[time_shift:-time_shift], number_of_chunks-1)
    middle_chunks.append(numpy.zeros_like(chunks[0])) # dummy empty chunk to be removed later # TODO far in modo che tutto sia sempre in potenze di 2
    # join and reorder odd and even chunks
    interlaced_chunks = numpy.transpose([chunks, middle_chunks], axes=[1,0,2]).reshape([2*number_of_chunks, -1])
    # TODO buttare l'ultimo chunk farlocco
    windowed_interlaced_chunks = interlaced_chunks*window
    return windowed_interlaced_chunks

time_data = numpy.load('./time_data.npy')
#time_data = white_noise + signal
#
#del white_noise
#del signal # TODO scrivere funzione in modo che ci sia il garbage collector automaticamente

# TODO NOTA: penso che il giusto tempo di coerenza vada scelto in base a quanto varia la modulazione sinusoidale (in modo da massimizzarne la visibilità nello spettrogramma)(in pratica lo spectral leakage della finestra e la risoluzione in frequenza data dal tempo di coerenza danno il Delta_y (e il tempo di coerenza stesso dà il Delta_x), che va confrontato col Delta_x e Delta_y relativi alla modulazione sinusoidale). in pratica penso che lo spartiacque sia quando la derivata della modulazione sinusoidale eccede 1.
# TODO poi fare anche line enhancing e denoising, in modo fa far apparire ancora di più il segnale integrato del doppio corno
def make_whitened_spectrogram(time_data): # the fast Fourier transform needs power on two to be fast
    windowed_interlaced_chunks = make_chunks(time_data)
    # TODO poi mettere fattore correttivo alla normalizzazione dello spettro per tenere in conto la perdita di potenza dovuta al finestramento
    print('computing the Fourier transform...')
    # TODO quando si calcola la FFT bisogna sempre soottrarre la media
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
    complex_image = numpy.transpose(unilateral_fft_data[0:-1]) # TODO creare un'altra funzione da chiamare separatamente in precedenza
    return complex_image, whitened_spectrogram # TODO separare i due risultati da avere in output # TODO valutare se inserire un dato fittizio per arrivare ad una potenza di 2

complex_image, whitened_spectrogram = make_whitened_spectrogram(time_data)

frequency_pixels, time_pixels = whitened_spectrogram.shape
image_time_axis = pandas.date_range('2018-01-01', periods=time_pixels, freq='{}s'.format(int(config.FFT_length/2)))
#image_time_axis = numpy.arange(0, image_time_interval - FFT_length/2, FFT_length/2) # TODO perché attualmente stiam interlacciando solo di metà
image_frequency_axis = unilateral_frequencies

coordinate_names = ['frequency', 'time']
coordinate_values = [image_frequency_axis, image_time_axis]

whitened_spectrogram = xarray.DataArray(data=whitened_spectrogram, 
                                        dims=coordinate_names, 
                                        coords=coordinate_values)

complex_image = xarray.DataArray(data=complex_image, 
                                 dims=coordinate_names, 
                                 coords=coordinate_values)

averaged_spectrum = whitened_spectrogram.mean(dim='time')
#averaged_spectrum = numpy.mean(whitened_spectrogram, axis=1) # TODO media VS mediana? # TODO perché qui la mediana fa molto più schifo della media? # TODO mediana dello spettro VS spettro autoregressivo ?
# TODO usando la mediana le due corna spariscono completamente!
frequency_slice = slice(software_injection_config.signal_starting_frequency - 0.02, software_injection_config.signal_starting_frequency + 0.02) # TODO hardcoded
zoomed_whitened_spectrogram = whitened_spectrogram.sel(frequency=frequency_slice)
zoomed_averaged_spectrum = averaged_spectrum.sel(frequency=frequency_slice)

zoomed_complex_image = complex_image.sel(frequency=frequency_slice)

make_plot = True

if make_plot is True:
    fig = pyplot.figure(figsize=[6,4])
    ax = pyplot.subplot()
    ax.set_title('time-averaged spectrum', y=1.02)
    ax.semilogy(unilateral_frequencies, averaged_spectrum) # grafico obbligatoriamente col logaritmo in y
    ax.xaxis.set_major_locator(pyplot.MultipleLocator(64))
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

# TODO NOTA: usando la finestra gaussiana invece che una flat o tukey, si introduce una asimmetria nella parte conca centrale del doppio corno, MA non ci sono gli artifatti (tipo ringing) che rendono a un certo punto invisibile il corno quando si usano tempi di coerenza molto alti
if make_plot is True:
    fig = pyplot.figure(figsize=[6,4])
    ax = pyplot.subplot()
    ax.set_title('zoomed time-averaged spectrum', y=1.02)
    #ax.semilogy(unilateral_frequencies[frequency_range], averaged_spectrum[frequency_range]) # grafico obbligatoriamente col logaritmo in y
    ax.semilogy(zoomed_averaged_spectrum.frequency, zoomed_averaged_spectrum)
    ax.xaxis.set_major_locator(pyplot.MultipleLocator(0.01))
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylim(1e-1, 1e3)
    pyplot.savefig('../media/zoomed_averaged_spectrum.jpg', bbox_inches='tight')
    pyplot.savefig('../media/double_horn/coherence_{}s_observation_{}d_sampling{}Hz.jpg'.format(config.FFT_length, int(config.rounded_observation_time/config.day), config.sampling_rate), bbox_inches='tight')
    pyplot.show()
    pyplot.close()
# TODO fare deconvoluzione per accentuare e stringere il picco (soprattutto dato che usando la finestra gaussiana nel tempo si sa che lo spreading in frequenza è pure gaussiano)?
# TODO NOTA: quando il segnale è troppo basso il doppio corno non appare completamente

#image = whitened_spectrogram[frequency_range]
image = zoomed_whitened_spectrogram
normalized_image = numpy.log10(image)

numpy.save('./real_image.npy', zoomed_whitened_spectrogram)
numpy.save('./complex_image.npy', zoomed_complex_image)

#import numpy
#import matplotlib
#matplotlib.use('Qt5Agg')
#from matplotlib import pyplot
#a = numpy.load('./real_image.npy')
#b = numpy.load('./complex_image.npy')
##for spacing in numpy.arange(10,40):
#if True:
#    axis=1
#    spacing = 32
#    print(spacing)
#    truncated_lentgh = numpy.floor_divide(a.shape[axis], spacing) * spacing
#    truncated_a = a[:,:truncated_lentgh]
#    number_of_periods = int(truncated_lentgh / spacing)
#    A = numpy.array(numpy.split(truncated_a, number_of_periods, axis=axis)).mean(axis=0)
#    pyplot.imshow(numpy.log(A))
#    pyplot.show()
#
#exit()

#normalized_image = image - image.min()
#normalized_image = normalized_image / normalized_image.max()

# TODO vedere se la spaziatura è effettivamente un fenomeno di interferenza
# TODO vedere se l'interferenza si mitiga o si amplifica se si mantiene la fase nell'immagine iniiziale
# TODO plottare la fase dell'fft2 con interferenza
# TODO vedere sovrapposizione di sinusoidi fino ad ottenere la reticolatura sull'asse x (e capire come la spaziatura macroscopica si mappa sulla struttura microscopica del segnale)
# TODO capire il perché l'immagine risulta una V con interferenza
# TODO fare fft2 dell'fft2 per estrarre la periodicità giornaliera invece che fare il fit (e vedere se ci sono solo le prime 4 armoniche della modulazione siderale)
# TODO vedere di eliminare tutti i logaritmi per tenere lineare l'operazione di filtraggio
# TODO trovare la minima ampiezza visibile per il segnale in scala lineare (e non logaritmica)
# TODO vedere se la figura dell'fft2 cambia aumentando del 150% il tempo di osservazione
# TODO vedere denoising/whitening dello spettrogramma con convoluzione con gaussiana multivariata
# TODO plottare il modulo dell'fft2 in scala lineare, per evitare di essere soverchiati da tutto il noise
# TODO scaricare la colormap per fare i plot di fase
# TODO fare script in modo da poter eseguire tutto su wn100 o Google CoLab
# TODO trovare un modo per cumulare il segnale col tempo di osservazione, accumulando statistica per la significatività dei risultati
# TODO sistemare il valore della fase (che sembra funzionare bene solo con la finestra flat e bassi valori del tempo di coerenza) in modo che i vari bin verticali abbiano un riferimento comune confrontabile e che dunque la derivata sia continua lungo x
# TODO fare la derivata lungo y della fase perché sembra che questa abbia un salto attraversando la linea di segnale
# TODO una volta sistemato il grafico della fase, rifare le fft2 nel domminio complesso per vedere se si riesce a guadagnare qualcosa sfruttando anche quella informazione
# TODO le linee orizzontali di rumore nello spettrogramma non dovrebbero dare fastidio alla fft2 perché non dovrebbero esibire periodicità dovuta all'interferenza
# TODO aumentando il tempo di osservazione (a parità di frequenza di modulazione del segnale) si ha interferenza ogni 1/N pixel, permettendo di selezionarne sempre di meno rispetto al totale e dunque filtrare sempre di più il segnale
# TODO vedere se facendo una fft2 della fft2 si riescono a selezionare solo le frequenze buone/volute dell'immagine, filtrando dunque tutto ciò che non fa interferenza
# TODO scrivere a Pep per implementare un filtro iniziale nel dominio del tempo, in modo che si abbassi il livello basale di tutti gli spettri
# TODO vedere finestrature che vadano perfettamente a zero e che concentrino quanto più possibile il segnale in pochi bin (che che nel contenpo abbiano un basso livello di contaminazione nelle bande laterali)
# TODO vedere come funziona la general chirplet transform per l'enhancement del doppler annuale e giornaliero e confrontarlo col possibile risultato di una deconvoluzione
# TODO vedere come funziona la trasformata di Fourier frazionaria sui segnali delle binarie (lo spettrogramma normale risulta ruotato di un certo angolo)

# NOTE: fft e ifft sono effettivamente quasi la stessa operazione. cambiano solo la normalizzazione del modulo e la fase invertita


original_image = image#normalized_image # TODO
fourier2D = numpy.fft.fftshift(numpy.fft.fft2(original_image - numpy.median(original_image)))
mask = numpy.zeros_like(fourier2D, dtype='float32') # TODO float32 VS float64
half = int(((fourier2D.shape[1]+1)/2)-1) # TODO BUG: l'indicizzazione a partire da 0 spesso è davvero scomodissima
index = numpy.arange(0, half).astype(int)
multiplier = 24 # TODO hardcoded
multiples = index[numpy.remainder(index, multiplier) == 0]
mask[:,half + multiples] = 1
mask[:,half - multiples] = 1

#mask[:,half + multiples + 1] = 1
#mask[:,half + multiples - 1] = 1
#mask[:,half - multiples + 1] = 1
#mask[:,half - multiples - 1] = 1
mask[:,:int(half*1/2)] = 0
mask[:,int(half*3/2):] = 0
#pyplot.imshow(mask)
#pyplot.show()

filtered_image = numpy.abs(numpy.fft.ifft2(fourier2D * mask)) # TODO come mai non serve rishiftare?
#pyplot.imshow(filtered_image)
#pyplot.show()

fig, [ax1, ax2] = pyplot.subplots(nrows=2, ncols=1, sharex=True)
ax1.set_title('original image (log10 values)')
ax1.imshow(numpy.log10(original_image))
ax2.set_title('filtered image (log10 values)')
ax2.imshow(numpy.log10(filtered_image))
pyplot.show()

fig, [ax1, ax2] = pyplot.subplots(nrows=2, ncols=1, sharex=True)
ax1.set_title('original image')
ax1.imshow(original_image)
ax2.set_title('filtered image')
ax2.imshow(filtered_image)
pyplot.show()

fig, [ax1, ax2] = pyplot.subplots(nrows=2, ncols=1, sharex=True)
ax1.set_title('2D Fourier transform (log10 values)')
ax1.imshow(numpy.log10(numpy.abs(fourier2D)))
ax2.set_title('filter mask')
ax2.imshow(mask)
pyplot.show()

# NOTE:
# a R=0.004 (con finestra gaussiana, 3 giorni di osservazione, sampling a 512 Hz e fft di 1024 s) il doppio corno c'è ma è molto poco visibile (migliora abbastanza usando la finestra flat), mentre col filtro il segnale si vede benissimo.
# a R=0.003 (con finestra gaussiana, 3 giorni di osservazione, sampling a 512 Hz e fft di 1024 s) il doppio corno è sparito totalmente (non ha la forma di corno neanche con la finestra flat), mentre col filtro il segnale si continua a vedere un poco.


exit()




pyplot.imshow(normalized_image) # TODO vedere se mettere o meno il quadrato (anche nella costruzione dello spettrogramma)
pyplot.show()
fourier2D = numpy.fft.fftshift(numpy.fft.fft2(normalized_image - numpy.median(normalized_image)))
pyplot.imshow(numpy.log10(numpy.abs(fourier2D)))
pyplot.show()
# NOTE: con la finestra gaussiana il segnale appare più dolce ed uniforme rispetto alla finestra flat. vedere se però questa è effettivamente la finestra che permette di vedere il minimo segnale
# NOTE: questa serie di plot mi sembra dare il segnale più intenso di tutti usando la finestra flat (mentre il più debole sembra essere quello derivante dall'immagine complessa)


pyplot.imshow(numpy.square(numpy.abs(numpy.log10(zoomed_complex_image))))
pyplot.show() # TODO log square abs != square abs log
#fourier2D = numpy.fft.fftshift(numpy.fft.fft2(zoomed_complex_image))
fourier2D = numpy.fft.fftshift(numpy.fft.fft2(numpy.log10(zoomed_complex_image))) # TODO controllare come si fa a sottrarre la media (la mediana non credo si possa usare perché il piano complesso è isomorfo a R^2 che non è un insieme ordinabile)
#pyplot.imshow(numpy.abs(fourier2D))
pyplot.imshow(numpy.log10(numpy.square(numpy.abs(fourier2D))))
pyplot.show()
#import cmocean
pyplot.imshow(numpy.angle(zoomed_complex_image), cmap='twilight') #cmap=cmocean.cm.phase
pyplot.colorbar()
pyplot.show()
pyplot.imshow(numpy.angle(fourier2D), cmap='twilight')
pyplot.show()
# NOTE: la precisione sulla fase immagino sia legata alla precisione temporale, che è complementare alla precisione in frequenza, necessaria per la stima spettrale
# NOTE: il pattern di fase risulta chiaro usando FFT con basso tempo di coerenza, ma tale pattern non sembra essere agevolmente utilizzabile
# NOTE: a naso direi che la massima precisione nel riconoscimento dell'immagine si avrà quando la pendenza della sinusoide che ci interessa sarà circa 45° (questo comporta un trade-off sia in tempo VS frequenza che in frequenza VS ampiezza della modulazione)
# TODO forse l'accumulo di statistica può venire fatto proprio sfruttando l'interferenza dentro la fft2 per buttare tutto lo spazio intermedio


pyplot.imshow(image)
pyplot.show()
fourier2D = numpy.fft.fftshift(numpy.fft.fft2(image - numpy.median(image)))
pyplot.imshow(numpy.log10(numpy.abs(fourier2D)))
pyplot.show()
pyplot.imshow(numpy.abs(fourier2D))
pyplot.show()
# NOTE: non usando il logaritmo dell'immagine questa non risulta una linea ma un'insieme di punti, che hanno una trasformata di Fourier completamente diversa da quella che cerchiamo
# NOTE questa serie di plot (usando il logaritmo) sembra dare un segnale molto intenso con finestra gaussiana




exit()

# NOTE: computing the 2D Fourier transform is mathematically equivalent to computing the 1D transform of all the rows and then computing the 1D transform of all the columns of th result.
# TODO fare un passabanda per eliminare le alte frequenze dall'immagine, che tipicamente sono quelle del noise. è equivalente ad uno smoothing gaussiano e dunque si sta spargendo il segnale su più bin? magari invece fare un whitening nel secondo spazio di Fourier saltando i vari picchi che probabimente riguardano il segnale (però magari per fare questa stima spettrale utilizzare una finestratura gaussiana)?

#pyplot.imshow(normalized_image)
#pyplot.show()
fourier = numpy.fft.fft(normalized_image) # 1D tranform
#fourier = numpy.fft.fft(image) # 1D tranform # TODO valutare se fare FFT 1D senza il logaritmo
# TODO usare anche p'informazione sulla fase dell'immagine originaria
a = abs(fourier)
f = fourier.copy()
f[a < 5] = 0 # TODO filtraggio molto grezzo (magari fare lì un denoising con un gaussian blurring)
b = numpy.fft.ifft(f)
pyplot.imshow(abs(b))
pyplot.show()
c = numpy.abs(b).mean(axis=1)
pyplot.plot(c)
pyplot.show()
pyplot.plot(normalized_image.mean(axis=1))
pyplot.show()





# TODO NOTA: con la finestra gaussiana la linea nello spettrogramma si vede molto molto meglio rispetto al caso con la Tukey
pyplot.figure(figsize=[4,6])
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

# TODO NOTA: a seconda della pendenza della sinusiode nel tempo-frequenza, il segnale nella peakmap può risultare completamente invisibile
# TODO sviluppare una peakmap con blurring multiscala per riuscire sempre a tracciare l'immagine (e inoltre trovare i massimi locali non solo lungo le slice temporali)
pyplot.figure(figsize=[4,6])
plot = peakmap.plot(cmap='gray_r')
pyplot.title('peakmap', y=1.02)
pyplot.xlabel('time [days]')
pyplot.ylabel('frequency [Hz]')
pyplot.xticks(rotation='45')
plot.colorbar.remove()
pyplot.tight_layout()
pyplot.savefig('../media/white_noise_peakmap_with_signal_scale_factor_{}.jpg'.format(signal_scale_factor), bbox_inches='tight')
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




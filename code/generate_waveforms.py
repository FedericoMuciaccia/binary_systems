
import pycbc.waveform
import pycbc.detector
import pycbc.psd

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

import numpy

# generate the gravitation-wave waveform for the two polarizations
h_plus, h_cross = pycbc.waveform.get_td_waveform(approximant="SEOBNRv2",
                                                 mass1=20, # in sun masses
                                                 mass2=20,
                                                 delta_t=1/1024, 
                                                 f_lower=40.0,
                                                 distance=100, # Mpc
                                                 inclination=0)

pyplot.title('signal waveform') # TODO source-frame?
pyplot.plot(h_plus.sample_times, h_plus, label='+ polarization')
pyplot.plot(h_cross.sample_times, h_cross, label='x polarization')
pyplot.xlabel('time [s]')
pyplot.legend()
pyplot.grid()
pyplot.show()

# NOTE: l'ampiezza delle due polarizzazioni è adimensionale e rappresenta la deformazione relativa di un cerchio unitario

# calculate the strain that each detectors would observe (for a specific sky location and time of the signal)

# time, orientation and location of the source in the sky
# source coordinates
right_ascension = (2/7)*2*numpy.pi # radians
declination = -(1/3)*numpy.pi # radians
polarization = 0 # TODO polarization phase ?
time = 0 # GPS seconds # TODO
h_plus.start_time = h_cross.start_time = time # TODO
#end_time = 1192529720
#h_plus.start_time += end_time
#h_cross.start_time += end_time

# NOTE: right ascension and polarization phase runs from 0 to 2pi. declination runs from pi/2 to -pi/2 with the poles at pi/2 and -pi/2

H = pycbc.detector.Detector('H1')
H_signal_strain = H.project_wave(h_plus, h_cross, 
                          right_ascension, 
                          declination, 
                          polarization)

L = pycbc.detector.Detector('L1')
L_signal_strain = L.project_wave(h_plus, h_cross, 
                          right_ascension, 
                          declination, 
                          polarization)

V = pycbc.detector.Detector('V1')
V_signal_strain = V.project_wave(h_plus, h_cross, 
                          right_ascension, 
                          declination, 
                          polarization)

# NOTE: the project_wave() function also takes into account the rotation of the Earth

pyplot.title('detector-frame signal strain')
pyplot.plot(H_signal_strain.sample_times, H_signal_strain, label='H1', color='red')
pyplot.plot(L_signal_strain.sample_times, L_signal_strain, label='L1', color='green')
pyplot.plot(V_signal_strain.sample_times, V_signal_strain, label='V1', color='blue')
pyplot.xlabel('time [s]')
pyplot.ylabel('strain') # TODO vedere se lasciarlo in potenze di 10 o metterlo in deciBell
pyplot.legend()
pyplot.grid()
pyplot.show()
# TODO come mai lo zero del tempo non sta al momento del merger?

#pyplot.plot(L_signal_strain[101:], V_signal_strain[100:])
#pyplot.plot(H_signal_strain[101:], V_signal_strain[100:])
#pyplot.plot(H_signal_strain[101:], L_signal_strain[101:])
#pyplot.plot(numpy.sqrt(numpy.square(H_signal_strain[101:]).numpy() + numpy.square(L_signal_strain[101:]).numpy() + numpy.square(V_signal_strain[100:]).numpy()))

# NOTE: l'ampiezza dello strain nei vari detector rappresenta la deformazione relativa (Delta L)/L dove L è la lunghezza del braccio dell'interferometro

# reference location will be the Hanford detector. see the `time_delay_from_earth_center` method to use use geocentric time as the reference
reference_detector = pycbc.detector.Detector("H1")

# TODO
from astropy.utils import iers
# Make sure the documentation can be built without an internet connection
iers.conf.auto_download = False

## Time in GPS seconds that the GW passes
#time = 100000000

# time of flight with respect to Hanford
print('time of flight [s]:')
for ifo in ["H1", "L1", "V1"]:
    d = pycbc.detector.Detector(ifo)
    dt = d.time_delay_from_detector(reference_detector, right_ascension, declination, time) # TODO vedere se è il tempo di volo
    print('H1 --> {}: {}'.format(ifo, dt))
    # TODO il tempo di volo è dato dalla distanza-luce tra i vari detector e dall'angolo di arrivo dell'onda

# f_plus and f_cross antenna pattern weights 
#f_plus, f_cross = d.antenna_pattern(right_ascension, declination, polarization, time)
#ht = fp * hp + fc * hc # attenzione che non considera nè il Doppler nè la funzione di trasferimento dei due strumenti nè la curva di sensibilità
# attenzione che le funzioni di trasferimento (che derivano dalla calibrazione) sono sempre assunte essere perfette

# NOTE: linear matched filtering is the optimal method to extract the signal in gaussian noise

############################################

# TODO usare le curve pubblicate per la psd di LIGO
# TODO curve "best per O2 cleaned (C02)"
# TODO chiedere ad Andrew (Jeff Kissel (jkissel@ligo.mit.edu), ASCII file della curca di sensibilità)




psds = ['AdVBNSOptimizedSensitivityP1200087',
        'AdVDesignSensitivityP1200087',
        'AdvVirgo',
        'aLIGODesignSensitivityP1200087',
        'aLIGOZeroDetHighPower']

#pycbc.psd.lalsimulation.SimNoisePSDaLIGOZeroDetHighPower()


import pycbc.psd

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

observation_time = 32 # s
coherence_time = observation_time # full coherent analysis
sampling_frequency = 2048 # Hz
time_resolution = 1/sampling_frequency # s
frequency_resolution = 1/coherence_time # Hz
Nyquist_frequency = sampling_frequency/2
frequency_samples = int(Nyquist_frequency/frequency_resolution)
low_frequency_cutoff = 30.0 # Hz

#p1 = pycbc.psd.aLIGOZeroDetHighPower(frequency_samples, frequency_resolution, low_frequency_cutoff)
#
#pyplot.plot(p1.sample_frequencies, p1, label='HighPower')
#pyplot.legend()
#pyplot.show()

#for my_psd in pycbc.psd.get_lalsim_psd_list():
#    print(my_psd)

#my_psd = 'AdvVirgo'
#noise_psd = pycbc.psd.from_string(my_psd, frequency_samples, frequency_resolution, low_frequency_cutoff)


VIRGO_noise_psd = pycbc.psd.AdvVirgo(frequency_samples, 
                                     frequency_resolution, 
                                     low_frequency_cutoff)
LIGO_noise_psd = pycbc.psd.aLIGOZeroDetHighPower(frequency_samples, 
                                                 frequency_resolution, 
                                                 low_frequency_cutoff)

pyplot.semilogy(VIRGO_noise_psd.sample_frequencies, 
                VIRGO_noise_psd, 
                label='AdvVirgo')
pyplot.semilogy(LIGO_noise_psd.sample_frequencies, 
                LIGO_noise_psd, 
                label='aLIGOZeroDetHighPower')
pyplot.xlabel('frequency [Hz]')
pyplot.legend()
pyplot.show()
pyplot.close()



# gaussian noise colored by the given psd

import pycbc.noise

# Generate 32 seconds of noise at 2048 Hz
time_samples = int(observation_time * sampling_frequency)
ts = pycbc.noise.noise_from_psd(time_samples, time_resolution, VIRGO_noise_psd, seed=127)

pyplot.plot(ts.sample_times, ts)
pyplot.ylabel('Strain')
pyplot.xlabel('Time (s)')
pyplot.show()





# Now, let's generate noise that has the same spectrum
htilde = pycbc.noise.frequency_noise_from_psd(VIRGO_noise_psd, seed=857)

pyplot.semilogy(htilde.sample_frequencies, htilde)
#pyplot.plot(htilde.sample_frequencies, htilde)
pyplot.xlim(low_frequency_cutoff, Nyquist_frequency)
pyplot.xlabel('frequency [Hz]')
pyplot.grid()
pyplot.show()
# TODO i valori non dovrebbero andare sotto zero


# Equivelantly in the time domain
hoft = htilde.to_timeseries()


pyplot.plot(hoft.sample_times, hoft)
pyplot.show()




# Well zoom in around a short time
hoft_zoom = hoft.time_slice(2.5, 3)
pyplot.plot(hoft_zoom.sample_times, hoft_zoom)
pyplot.show()







import pycbc.noise

# The color of the noise matches a PSD which you provide
flow = 30.0
delta_f = 1.0 / 16
flen = int(2048 / delta_f) + 1
noise_psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)


# Generate 32 seconds of noise at 4096 Hz
delta_t = 1.0 / 4096
tsamples = int(32 / delta_t)
ts = pycbc.noise.noise_from_psd(tsamples, delta_t, noise_psd, seed=127)

pyplot.plot(ts.sample_times, ts)
pyplot.ylabel('Strain')
pyplot.xlabel('Time (s)')
pyplot.show()





import pylab

# Generate a PSD using an analytic expression for 
# the full design Advanced LIGO noise curve
f_lower = 10
duration = 128
sample_rate = 4096
tsamples = sample_rate * duration
fsamples = tsamples / 2 + 1
df = 1.0 / duration
psd = pycbc.psd.from_string('aLIGOZeroDetHighPower', fsamples, df, f_lower)

# Let's take a look at the spectrum
pylab.loglog(psd.sample_frequencies, psd)
pylab.xlim(20, 1024)
pylab.ylim(1e-48, 1e-45)
pylab.xlabel('Frequency (Hz)')
pylab.ylabel('Strain^2 / Hz')
pylab.grid()
pylab.show()


# Now, let's generate noise that has the same spectrum
htilde = pycbc.noise.frequency_noise_from_psd(psd, seed=857)

pylab.loglog(htilde.sample_frequencies, htilde)
pylab.xlim(20, 1024)
pylab.xlabel('Frequency (Hz)')
pylab.grid()
pylab.show()

############################3


import lal
from lalpulsar import simulateCW

def waveform(h0, cosi, freq, f1dot):
    def wf(dt):
        dphi = lal.TWOPI * (freq * dt + f1dot * 0.5 * dt**2)
        ap = h0 * (1.0 + cosi**2) / 2.0
        ax = h0 * cosi
        return dphi, ap, ax
    return wf

tref     = 900043200
tstart   = 900000000
Tdata    = 86400
h0       = 1e-24
cosi     = 0.123
psi      = 2.345
phi0     = 3.210
freq     = 10.0
f1dot    = -1.35e-8
dt_wf    = 5
alpha    = 6.12
delta    = 1.02
detector = 'H1'

wf = waveform(h0, cosi, freq, f1dot)
S = simulateCW.CWSimulator(tref, tstart, Tdata, wf, dt_wf, phi0, psi, alpha, delta, detector)

# To write SFT files
for file, i, N in S.write_sft_files(fmax=32, Tsft=1800, comment="simCW"):
    print('Generated SFT file %s (%i of %i)' % (file, i+1, N))

# To write frame files
for file, i, N in S.write_frame_files(fs=1, Tframe=1800, comment="simCW"):
    print('Generated frame file %s (%i of %i)' % (file, i+1, N))










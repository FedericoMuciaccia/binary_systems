
import pycbc.waveform
import pycbc.detector

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot

# generate the gravitation-wave waveform for the two polarizations
h_plus, h_cross = pycbc.waveform.get_td_waveform(approximant="SEOBNRv2",
                                                 mass1=20, 
                                                 mass2=20,
                                                 delta_t=1.0/1024, 
                                                 f_lower=40.0,
                                                 distance=100, # Mpc
                                                 inclination=0)

pyplot.title('signal waveform')
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
right_ascension = 1.7 # radians
declination = 1.7 # radians
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


# f_plus and f_cross antenna pattern weights 
#f_plus, f_cross = d.antenna_pattern(right_ascension, declination, polarization, time)
#ht = fp * hp + fc * hc # attenzione che non considera nè il Doppler nè la funzione di trasferimento dei due strumenti nè la curva di sensibilità
# attenzione che le funzioni di trasferimento (che derivano dalla calibrazione) sono sempre assunte essere perfette


pyplot.title('detector-frame signal strain')
pyplot.plot(H_signal_strain.sample_times, H_signal_strain, label='H1', color='red')
pyplot.plot(L_signal_strain.sample_times, L_signal_strain, label='L1', color='green')
pyplot.plot(V_signal_strain.sample_times, V_signal_strain, label='V1', color='blue')
pyplot.xlabel('time [s]')
pyplot.ylabel('strain') # TODO vedere se lasciarlo in potenze di 10 o metterlo in deciBell
pyplot.legend()
pyplot.grid()
pyplot.show()

# NOTE: l'ampiezza dello strain nei vari detector rappresenta la deformazione relativa (Delta L)/L dove L è la lunghezza del braccio dell'interferometro




# NOTE: linear matched filtering is the optimal method to extract the signal in gaussian noise




# TODO usare le curve pubblicate per la psd di LIGO
# TODO curve "best per O2 cleaned (C02)"
# TODO chiedere ad Andrew (Jeff Kissel (jkissel@ligo.mit.edu), ASCII file della curca di sensibilità)








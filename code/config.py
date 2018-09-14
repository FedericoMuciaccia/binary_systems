
second = 1
minute = 60*second
hour = 60*minute
day = 24*hour

sampling_rate = 512#1024#512#256 # Hz # subsampled from 4096 Hz data
observation_time = 3*day
FFT_length = 8192 # seconds
window = 'gaussian' # or 'tukey' or 'flat'

time_resolution = 1/sampling_rate # s # time-domain time binning

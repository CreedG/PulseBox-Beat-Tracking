import matplotlib.pyplot as plt
import wave, sys
import numpy as np
from scikits.talkbox.features import mfcc
import scipy.signal as sig #find_peaks_cwt

PATH_TO_OPEN_DATASET = '/home/josh/Documents/beats/training_set/open/'

def official_avg_bpm(filename, plot_len, plot_time):
#read in beat times
    f = open(PATH_TO_OPEN_DATASET + filename + '.txt')
    beat_times_raw = f.read().split()
    beat_times = [float(beat) for beat in beat_times_raw]

#get differences in between beat times, it is interesting to put a plot here
    b = []
    for i in range(1,len(beat_times)-1):
       b.append(60/(beat_times[i] - beat_times[i-1]))

#calculate average difference
    average =  (sum(b)/len(b))
    print("Official average BPM = " + str(average))

#generate plottable beats
    plottable_beats = [0] * plot_len
    for i in range(0,int(plot_time/(60 / average))+1):
        plottable_beats[int(i * (60 / average) * (plot_len / plot_time))] = 50

    return (average, plottable_beats)

def calculate_bpm(nceps, time, peaks):
# make overlay of delta function on plot
    peaks_to_plot = [100 if x in peaks else 0 for x in range(0,nceps)]
    plt.plot(peaks_to_plot,color='#cccccc')

#Find possible bpm times based on findpeaks data
    peaks_in_secs = [float(x) * time / nceps for x in peaks]
    interpeak_time = []
    for i in range(1,len(peaks_in_secs)-1):
        interpeak_time.append(peaks_in_secs[i] - peaks_in_secs[i-1])
    avg = sum(interpeak_time)/len(interpeak_time)
    opts = [60./avg, 30./avg, 15./avg]
    print("Calculated Bpm = " + str(opts))
    return opts

def check(actual, opts, thresh = 2):
#Function which checks if actual bpm is anywhere close to any of the opts.
# Play around with thresh
    for x in opts:
        if abs(actual - x) < thresh:
            return abs(actual - x)
    return 0

def tempo(fi):
    print(fi)
    wav = wave.open(PATH_TO_OPEN_DATASET + fi + '.wav')

#Parameters are arbitrarily set. I think we'll be using about 3 secs for the autocorr but who knows
    wav.setpos(wav.getnframes()/2)
    frames_to_read = wav.getnframes()/10

#Prep
    Y = wav.readframes(frames_to_read)
    time = frames_to_read / wav.getframerate()
    frames = np.fromstring(Y,dtype='int16') * 1.

#Run MFCC. Only using ceps currently. Do some magic on it suggested by a paper
    ceps, mspec, spec = mfcc(frames)
    w = (ceps[:,1] - ceps[:,0])**2 + (ceps[:,2] - ceps[:,1])**2 + (ceps[:,3] - ceps[:,2])**2

#Clean it up and run an autocorrelation
    w -= np.mean(w)
    w /= max(w)
    a = np.correlate(w,w,mode='full')
    a = a[len(a)/2:]

#Flail a find_peaks function onto the autocorr. Works OKAY
    b = sig.find_peaks_cwt(a,np.arange(1,20 * time))
    opts = calculate_bpm(len(a), time, b)
    actual, back = official_avg_bpm(fi, len(a), time)

#Plot some stuff
    plt.plot(a)
    plt.plot(back,color='#cc0000')
    plt.show()

    return check(actual, opts)

def main(argv):
#    if len(argv) != 1:
#        print "Read in file plx"
#        return
    results = []
    for i in range(1,9):
        if tempo("open_00"+str(i)) == 0:
            results.append(i)
    for i in range(10,26):
        if tempo("open_0"+str(i)) == 0:
            results.append(i)
    print(results)
    print(len(results))

if __name__ == "__main__":
    main(sys.argv[1:])

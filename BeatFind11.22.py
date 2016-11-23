#I3Creed - Starting point for our algorithm
import struct
import wave, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import sleep

import scipy.fftpack
import scipy.signal

import time

import math

from beat_detect_utils import correlate_onsets
from beat_detect_utils import find_consensus

import threading



#A LOT OF GLOBAL VARIABLES (SHOULD BE FINE)

song_over = False
processing_done = False
cur_time = 0
detected_beat_times = []

#ALGORITHM PARAMETERS
f_ranges = [0,125,250,500,1000,2000,4000,11000]

#maintain an x axis for vector data
time_vec = []

#for calculation of the newest onset for each frequency range
prev_range_pow = np.zeros(7, dtype=np.int)

#the newest onset is placed at the front of this 3 element queue (minimum length to find peak in real time)
power_onset_vecs = np.array([[],[],[],[],[],[],[]], dtype=np.int)

#the onset peak vectors that will be used for phase detection. not numpy arrays b/c they differ in length
power_onset_peaks_strength = [[],[],[],[],[],[],[]]
power_onset_peaks_time = [[],[],[],[],[],[],[]]

#the cutoffs will change to only capture a reasonable number of peaks per second (1-10)
power_onset_peaks_cutoff = [100,100,100,100,100,100,100]


#DEBUG GLOBALS (the whole volume history)
signal_volume_vec = []



#THE ALGORITHM

#Read in with chunks of 1024 samples at 22050Hz -> every 46ms
#Should complete in about 3 seconds to be fast enough for pi, I think



#Two threads of execution
#AQUISITION THREAD executes every audio chunk (every 256 samples /22050 = 11.6ms and gets information for that chunk)
    #Records the onset vectors and the peak locations
#PROCESSING THREAD executes as often as possible (or at the fastest every 200ms of song time or some other number),
    # gives a new guess for the beats each time. It uses the onset vectors and peak locations to guess the beats


#Currently, this generates a tempo guess from correlating the onset vectors. In the future, it should also guess the phase.
def processing_thread():
    global f_ranges, prev_range_pow, power_onset_peaks_strength, \
        power_onset_peaks_time, power_onset_peaks_cutoff, power_onset_vecs, song_over, \
        cur_time, processing_done

    #Don't even bother until one second in
    next_evaluation = 1.0

    while(song_over == False):

        #Wait until the next time to evaluate beats
        if (cur_time < next_evaluation):
            time.sleep(.1)     #Needs to be modified for real time
            next_evaluation += 1

        #print("EVALUATING AT ",cur_time)

        #PART 1: Period detection by autocorrelating correlating the onset vectors

        top_periods = []
        for i in range(0, 7):
            top_periods.append(correlate_onsets(power_onset_vecs[i], power_onset_vecs[i])[0][0])
        #print("top periods!", top_periods)
        testBool = True

        # Consider them all equally likely in the consensus process (for now)
        overall_period_z = np.ones(len(top_periods))

        best_period_of_all = find_consensus(top_periods, overall_period_z, 0.01)[0][0]

        print("best period guess at",cur_time,"is",best_period_of_all)


        #PART 2: Phase detection by applying the guessed period to the beats
        #Analyze power_onset_peaks_time and strength to see if they match the guessed period

        #Big ass introduction to this part for josh

        '''power_onset_peaks_time has the peak time and power_onset_peaks_strength has the
        corresponding strength at that time. As in, a given index on each array will contain a time,strength pair.
        power_onset_peaks_time is 7 dimensional tho, so index into the frequency range first '''

        '''The peak vectors were derived by averaging peaks on the onset vectors. they should occur pretty regularly but
        they are totally distinct (like a pulse train) with positions given by power_onset_peaks_time.
        They are shown in the graph that pops up when you run the code. Basing it on these should be
        better for finding the phase because we have exact moments rather than messy time series signals'''

        '''Probably write a function in beat_detect_utils that can be called on arbitrary peak data then we
        will run it for all 7 that way. My guess on how to do this is take a look at recent peaks and backtrack
        by subtracting the known period (with an error tolerance) and check for matching peaks as you traverse
        back to the beginning. Something on the phase should have more hits and more total power.'''




    print("Processing thread closed")
    processing_done = True





def aquisition_thread(wav):
    global f_ranges, prev_range_pow, power_onset_peaks_strength, \
        power_onset_peaks_time, power_onset_peaks_cutoff, power_onset_vecs, song_over, \
        cur_time, processing_done

    song_over = False

    cur_sample = 0
    cur_window = 0
    cur_time = 0
    time_step = 512.0/44100

    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while (len(sample_arr) == 1024):

        cur_time = cur_sample/44100
        time_vec.append(cur_time)

        signal_volume_vec.append(np.sum(np.abs(sample_arr)))

        windowed = np.hanning(1024)*sample_arr


        #STEP 1: GENERATE ONSET VECTORS THAT WILL BE USED LATER TO FIND THE BEATS

        x_psd, y_psd = scipy.signal.periodogram(windowed, 22050) #the power spectrum

        y_psd = np.sqrt(y_psd) #sqrt power values. get rid of huge differences and prevent int overflow with correlation

        #sum up the ranges
        range_pow = np.array([0,0,0,0,0,0,0])
        range_num = 1

        #TODO: Speed up this loop
        #COULD BE FURTHER SPED UP, SHOULD USE NUMPY TRICKS
        sum = 0
        for index, freq in enumerate(x_psd):
            #print(index,freq,range_num,y_psd[index])
            sum += y_psd[index]
            if freq > f_ranges[range_num]:
                range_pow[range_num-1] = sum
                sum = 0
                range_num += 1
                if (range_num >= 8):
                    break

        #Now we have the power in each range. Compute the derivative (simple difference) and get rid of negative power changes to get onset vectors

        new_range_onsets = range_pow-prev_range_pow
        new_range_onsets = new_range_onsets.clip(min=0)
        prev_range_pow = range_pow

        #Add the new onset values to the vectors so that we have power_onset_vecs[range_num][time_idx]
        new_range_onsets = new_range_onsets.reshape((7,1))
        power_onset_vecs = np.hstack([power_onset_vecs,new_range_onsets])

        #STEP 2: PEAK FINDING ON ALL ONSET VECTORS

        #Real time peak finding on the range_onset_vecs autocorrelation (requires the last 3 elements of the onset vec)
        #Check the middle to see if it is a local max, registers as peak if it's large enough

        #The cutoff dynamically adjusts to allow maximum 10 peaks per second, lowers if there are none (ideally at least 1 per sec)
        #Hard limit is 10 to get rid of noise

        check_peak_idx = len(time_vec)-2

        #The cutoff automatically assumes its new value and then drops off in time. Prevents too many close peaks.

        for f in range(0,7):
            if (power_onset_vecs[f][check_peak_idx] > power_onset_peaks_cutoff[f] and
                        power_onset_vecs[f][check_peak_idx] > power_onset_vecs[f][check_peak_idx+1] and power_onset_vecs[f][check_peak_idx] > power_onset_vecs[f][check_peak_idx-1]):

                power_onset_peaks_cutoff[f] = power_onset_vecs[f][check_peak_idx]

                power_onset_peaks_strength[f].append(power_onset_vecs[f][check_peak_idx])
                power_onset_peaks_time[f].append(cur_time-time_step)

            power_onset_peaks_cutoff[f]*=.98

        cur_sample = wav.tell()
        cur_window +=1

        sample_arr = sample_arr[256:]
        sample_arr = np.append(sample_arr,np.fromstring(wav.readframes(512), dtype='int16')[::2])

    print("Data aquisition thread closed (the song is over)")
    song_over = True


def beat_detect_simulate_realtime(wav):

    global detected_beat_times, processing_done

    t0 = time.time() #benchmark

    wav.rewind()
    detected_beat_times = []

    '''Start the processing thread, this will run every 200ms of song time and generate new guesses for beats. On the real
    time system, this may take longer than 200ms to run towards the end of a 30s song (more data). It will just run again
    whenever it can, which is fine since the predicted tempo and phase will likely be 'locked in' by then anyway'''
    t = threading.Thread(target=processing_thread)
    t.start()

    '''Start the aquisition thread (it's just the main thread), this will block until the song is totally processed. It will
    also prepare all the relevant data for the processing thread to use along the way'''
    aquisition_thread(wav)


    while(processing_done == False):
        pass


    t1 = time.time() #benchmark
    print("Total algorithm time",t1-t0)

    return detected_beat_times

#END ALGORITHM


#SET UP THE PLOT

def main(argv):

    wavfile = ""

    #Get the wave file set up (wave module does everything)
    if len(argv) != 1:
        wavfile = "open/open_001.wav"
    else:
        wavfile = argv[0]

    wav = wave.open(wavfile)
    print("Successfully read the wav file:",wavfile,wav.getparams())

    #Get the algorithm's beat times

    found_beats = beat_detect_simulate_realtime(wav)

    #Grab the known beat times from the text file with the same name

    known_beats = []
    txtfile_name = wavfile.split('.')[0]+".txt"

    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
            if 'str' in line:
                break

    print("The correct period from the text file was ",(known_beats[28]-known_beats[20])/8)



    #Create the plots (3 stacked plots each showing 10 seconds)
    #This was annoying to set up but is best for showing more of the data
    #Don't worry about the details, it's just some matplotlib bullshit


    wav.rewind()

    total_samples = wav.getnframes()

    display_div = 50

    sample_data = sample_arr = downsample_arr = x_axis_seconds = [0,0,0]
    plt.figure(1)
    sample_data = wav.readframes(wav.getnframes())
    sample_arr = np.fromstring(sample_data, dtype='int16')
    downsample_arr = sample_arr[::display_div]
    plt.plot(downsample_arr, color='#aaaaaa')
    plt.grid()



    #Add the beat indicators to the plots (known beats in red and beats found by
    #the algorithm in blue)

    for b in known_beats:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-36000,39000], 'k-', lw=1.5, color='r')

    '''
    for b in found_beats:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-39000,36000], 'k-', lw=1.5, color='b')
    '''

    print("Showing graph of onset peak times. These are the peaks of onset vectors from low frequency at bottom to high."
          "The red bar underneath is the real beat.")
    for b in power_onset_peaks_time[0]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-40000,-30000], 'k-', lw=1.5, color='g')
    for b in power_onset_peaks_time[1]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-30000,-20000], 'k-', lw=1.5, color='purple')
    for b in power_onset_peaks_time[2]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-20000,-10000], 'k-', lw=1.5, color='b')
    for b in power_onset_peaks_time[3]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-10000,0], 'k-', lw=1.5, color='g')
    for b in power_onset_peaks_time[4]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[0,10000], 'k-', lw=1.5, color='purple')
    for b in power_onset_peaks_time[5]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[10000,20000], 'k-', lw=1.5, color='blue')
    for b in power_onset_peaks_time[6]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[20000,30000], 'k-', lw=1.5, color='green')
    for b in found_beats:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[40000,50000], 'k-', lw=1.5, color='black')

    ax = plt.gca()
    ax.xaxis.set_ticks([n*44100/display_div for n in range (0,31)])
    ticks = ax.get_xticks()/(44100/display_div)
    ax.set_xticklabels(ticks)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])


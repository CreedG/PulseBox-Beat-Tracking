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



#ALGORITHM PARAMETERS
f_ranges = [0,125,250,500,1000,2000,4000,11000]

#maintain an x axis for vector data
time_vec = []

#for calculation of the newest onset for each frequency range
prev_range_pow = np.zeros(7, dtype=np.int)

#the newest onset is placed at the front of this 3 element queue (minimum length to find peak in real time)
power_onset_vecs = np.array([[],[],[],[],[],[],[]], dtype=np.int)

#the onset peak vectors that will be used for phase detection. not numpy arrays b/c they differ in length
range_onset_peaks_strength = [[],[],[],[],[],[],[]]
range_onset_peaks_time = [[],[],[],[],[],[],[]]

#the cutoffs will change to only capture a reasonable number of peaks per second (1-10)
range_onset_peaks_cutoff = [100,100,100,100,100,100,100]


#DEBUG GLOBALS
signal_volume_vec = []


#HELPERS
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]


#THE ALGORITHM



#Read in with chunks of 1024 samples at 22050Hz -> every 46ms

#SHOULD COMPLETE IN ABOUT 3 SECONDS TO BE FAST ENOUGH FOR PI

def beat_detect(wav):

    global f_ranges, prev_range_pow, range_onset_queue, range_onset_peaks_strength, range_onset_peaks_time, range_onset_peaks_cutoff
    global power_onset_vecs

    #SETUP VARS
    t0 = time.time() #benchmark
    beat_times = []

    cur_sample = 0
    cur_window = 0
    cur_time = 0
    time_step = 512.0/44100

    testBool = False

    #START
    wav.rewind()
    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while (len(sample_arr) == 1024):

        cur_time = cur_sample/44100
        time_vec.append(cur_time)


        signal_volume_vec.append(np.sum(np.abs(sample_arr)))


        windowed = np.hanning(1024)*sample_arr


        #STEP 1: GENERATE ONSET VECTORS THAT WILL BE USED LATER TO FIND THE BEATS
        #1A GENERATES A 10 POWER CHANGE ONSET VECTORS FOR 10 DIFFERENT FREQUENCY BANDS
        #1B GENERATES A CHORD CHANGE ONSET VECTOR
        #1C GENERATES A DRUM ONSET VECTOR IF DRUMS SEEM TO EXIST
        #AFTER STEP 1 WE WILL HAVE 12 ONSET VECTORS TO WORK WITH

        #STEP 1A: SIGNAL TO 10 ONSET VECTORS FOR POWER CHANGES (7 smaller bands, low band, medium band, all band)

        x_psd, y_psd = scipy.signal.periodogram(windowed, 22050) #the power spectrum

        y_psd = np.sqrt(y_psd)/10 #sqrt power values. peaks too large, some way higher than others. also prevents int overflow

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
        new_range_onsets = new_range_onsets.clip(min=0)       #maybe also clip low values above 0?
        prev_range_pow = range_pow

        #Add the new onset values to the vectors so that we have power_onset_vecs[range_num][time_idx]
        new_range_onsets = new_range_onsets.reshape((7,1))
        power_onset_vecs = np.hstack([power_onset_vecs,new_range_onsets])

        #END STEP 1A: power_onset_vecs is the power onset vector for each of the 7 ranges (still need to add 3 other power groups)

        #STEP 2 TEMPO ESTIMATES FROM THE ONSET VECTORS
        
        corr = []
        if (cur_time >= 20 and testBool == False):
            top_periods = []
            for i in range (0,7):
                top_periods.append(correlate_onsets(power_onset_vecs[i],power_onset_vecs[i])[0][0])
            print("top periods!",top_periods)
            testBool = True

            # Consider them all equally likely in the consensus process
            overall_period_z = np.ones(len(top_periods))

            best_period_of_all = find_consensus(top_periods,overall_period_z,0.003)[0][0]
            print("THE PERIOD",best_period_of_all)


        #Just pick some times here and there to show
        if (cur_window % 500 == 666):

            plt.figure(1)

            plt.subplot(311)
            #plt.plot(windowed,color='r')
            plt.plot(time_vec,power_onset_vecs[6], color='r')


            plt.subplot(312)
            #plt.plot(time_vec,power_onset_vecs[0], color='b')
            plt.plot(time_vec,power_onset_vecs[0])


            plt.subplot(313)
            #plt.plot(x_psd, y_psd, color='g')
            plt.plot(time_vec,signal_volume_vec,color='g')
            plt.grid()
            plt.show()



        #STEP 4: PEAK FINDING ON ALL ONSET VECTORS

        #Real time peak finding on the range_onset_vecs autocorrelation (requires the last 3 elements of the onset vec)
        #Check the middle to see if it is a local max, registers as peak if it's large enough

        #The cutoff dynamically adjusts to allow maximum 10 peaks per second, lowers if there are none (ideally at least 1 per sec)
        #Hard limit is 10 to get rid of noise

        #The cutoff automatically assumes value of half of new biggest value, it always drops off to 1000 within one second
        '''
        for f in range(0,7):
            if (range_onset_queue[1][f] > range_onset_peaks_cutoff[f] and
                        range_onset_queue[1][f] > range_onset_queue[2][f] and range_onset_queue[1][f] > range_onset_queue[0][f]):

                range_onset_peaks_cutoff[f] = range_onset_queue[1][f]

                range_onset_peaks_strength[f].append(range_onset_queue[1][f])
                range_onset_peaks_time[f].append(cur_time-time_step)

            range_onset_peaks_cutoff[f]*=.98


        #Perform correlations on the peaks so far. Each correlation gives a period t, phase p, and confidence z


        #Autocorrelation on the lowest band
        if (cur_time >= 5 and testBool == False):
            t,p,z = corr_peaks([range_onset_peaks_time[1],range_onset_peaks_strength[1]],
                               [range_onset_peaks_time[1],range_onset_peaks_strength[1]])

            print("pd,phase,z")
            print(t,p,z)

            beat = p
            while (beat > 0):
                beat -= t
            while (beat < 30):
                beat += t
                beat_times.append(beat)

            testBool = True

        '''
        #Debug view plots

        if (cur_window % 20 == 30):

            print(range_onset_peaks_strength)
            print(range_onset_peaks_time[0])
            print(len(range_onset_peaks_strength[0]))
            print(len(range_onset_peaks_time[0]))

            print(len(range_onset_peaks_time[6]))

            plt.figure(1)

            plt.subplot(311)
            plt.plot(windowed,color='r')

            plt.subplot(312)
            plt.plot(range_onset_peaks_time[6],range_onset_peaks_strength[6], color='b')
            #plt.plot(onset_cor)


            plt.subplot(313)
            plt.plot(x_psd, y_psd, color='g')
            plt.grid()
            plt.show()

        cur_sample = wav.tell()
        cur_window +=1

        sample_arr = sample_arr[256:]
        sample_arr = np.append(sample_arr,np.fromstring(wav.readframes(512), dtype='int16')[::2])

    t1 = time.time()
    print("Time elapsed")
    print(t1-t0)




    return beat_times

#END ALGORITHM


#SET UP THE PLOT

def main(argv):

    wavfile = ""

    #Get the wave file set up (wave module does everything)
    if len(argv) != 1:
        wavfile = "open/open_025.wav"
    else:
        wavfile = argv[0]

    wav = wave.open(wavfile)
    print("Successfully read the wav file:",wavfile,wav.getparams())




    #Get the algorithm's beat times

    found_beats = beat_detect(wav)

    #print("FOUND BEATS")
    #print(found_beats)

    #Grab the known beat times from the text file with the same name

    known_beats = []

    txtfile_name = wavfile.split('.')[0]+".txt"

    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
            if 'str' in line:
                break

    print("CORRECT BEATS")
    print(known_beats)

    print("Expected period of ",(known_beats[8]-known_beats[0])/8)
    print("At end period of ",(known_beats[28]-known_beats[20])/8)



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

    for b in range_onset_peaks_time[0]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-40000,-30000], 'k-', lw=1.5, color='g')
    for b in range_onset_peaks_time[1]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-30000,-20000], 'k-', lw=1.5, color='purple')
    for b in range_onset_peaks_time[2]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-20000,-10000], 'k-', lw=1.5, color='b')
    for b in range_onset_peaks_time[3]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-10000,0], 'k-', lw=1.5, color='g')
    for b in range_onset_peaks_time[4]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[0,10000], 'k-', lw=1.5, color='purple')
    for b in range_onset_peaks_time[5]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[10000,20000], 'k-', lw=1.5, color='blue')
    for b in range_onset_peaks_time[6]:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[20000,30000], 'k-', lw=1.5, color='green')
    for b in found_beats:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[40000,50000], 'k-', lw=1.5, color='black')

    print( len(range_onset_peaks_strength[0])+
            len(range_onset_peaks_strength[1])+
            len(range_onset_peaks_strength[2])+
            len(range_onset_peaks_strength[3])+
            len(range_onset_peaks_strength[4])+
            len(range_onset_peaks_strength[5])+
            len(range_onset_peaks_strength[6]))

    ax = plt.gca()
    ax.xaxis.set_ticks([n*44100/display_div for n in range (0,31)])
    ticks = ax.get_xticks()/(44100/display_div)
    ax.set_xticklabels(ticks)
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1:])


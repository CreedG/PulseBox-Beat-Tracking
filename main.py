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
from utils import correlate_onsets, find_consensus, detect_phase, KalmanFilter
import threading
import os

DEBUG = True
CORRECTNESS_THRESHOLD = .05
PATH = os.getcwd() + "/all/" #Change to where your wav files are
old_period_guesses = []
period_guesses = []
beat_guesses = []
k = KalmanFilter()

def debug(s):
    if DEBUG: print(s)


def initialize_values():
    global song_over, cur_time, time_vec, prev_range_pow, power_onset_vecs, peaks_strength, peaks_time, peaks_cutoff, signal_volume_vec
    song_over = False
    cur_time = 0
    time_vec = []
    prev_range_pow = np.zeros(7, dtype=np.int)
    power_onset_vecs = np.array([[],[],[],[],[],[],[]], dtype=np.int)

    #the peak vectors that will be used for phase detection
    peaks_strength = [[],[],[],[],[],[],[]]
    peaks_time = [[],[],[],[],[],[],[]]
    peaks_cutoff = [100,100,100,100,100,100,100]


#Currently not threaded, just called every 300ms to update beat time guesses
def processing_thread():
    global prev_range_pow, peaks_strength, peaks_time, peaks_cutoff, power_onset_vecs, song_over, cur_time, known_pd, k

    # PART 1: Period detection by autocorrelating correlating the onset vectors

    candidate_periods = []
    for band in range(0, 7):
        guesses = correlate_onsets(power_onset_vecs[band], power_onset_vecs[band])
        for guess in guesses:
            candidate_periods.append(guess)

    candidate_periods = np.array(candidate_periods)

    times = candidate_periods[:, 0]
    strengths = candidate_periods[:, 1]
    total_str = np.sum(strengths)
    voting_power = []

    for s in strengths:
        vote = math.sqrt(s / float(total_str)) * 10
        voting_power.append(vote)

    the_consensus = find_consensus(times, voting_power, .05)

    best_period = the_consensus[0][0]
    new_cool_period = k.iter(the_consensus[0][0])

    period_guesses.append(new_cool_period)
    old_period_guesses.append(best_period)



# PART 2: Phase detection by applying the guessed period to the beats
# Analyze peaks_time and strength to see if they match the guessed period
    beat_guesses.append(detect_phase(known_pd, peaks_time, peaks_strength, cur_time))

def acquisition_thread(wav):
    global prev_range_pow, peaks_strength, \
            peaks_time, peaks_cutoff, power_onset_vecs, song_over, \
            cur_time

    song_over = False
    cur_sample = 0
    cur_window = 0
    cur_time = 0
    time_step = 512.0/44100

    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while (len(sample_arr) == 1024):

        #process new beat times every once in a while (this will eventually be another thread)
        if (cur_window % 30 == 0 and cur_time >= 5):
            processing_thread()

        cur_time = cur_sample/44100
        time_vec.append(cur_time)

        windowed = np.hanning(1024)*sample_arr

        #STEP 1: GENERATE ONSET VECTORS THAT WILL BE USED LATER TO FIND THE BEATS

        x_psd, y_psd = scipy.signal.periodogram(windowed, 22050) #the power spectrum

        y_psd = np.sqrt(y_psd) #sqrt power values. get rid of huge differences and prevent int overflow with correlation

        #sum up the ranges
        range_pow = np.array([0,0,0,0,0,0,0])

        range_pow[0] = np.sum(y_psd[0:7])
        range_pow[1] = np.sum(y_psd[7:13])
        range_pow[2] = np.sum(y_psd[13:25])
        range_pow[3] = np.sum(y_psd[25:48])
        range_pow[4] = np.sum(y_psd[48:94])
        range_pow[5] = np.sum(y_psd[94:187])*(4/5)
        range_pow[6] = np.sum(y_psd[187:510])*(2/5)


        #Now we have the power in each range. Compute the derivative (simple difference) and get rid of negative power changes to get onset vectors
        new_range_onsets = range_pow-prev_range_pow
        new_range_onsets = new_range_onsets.clip(min=0)
        prev_range_pow = range_pow

        #Add the new onset values to the vectors so that we have power_onset_vecs[range_num][time_idx]
        new_range_onsets = new_range_onsets.reshape((7,1))
        power_onset_vecs = np.hstack([power_onset_vecs,new_range_onsets])

        #STEP 2: PEAK FINDING ON ALL ONSET VECTORS

        check_peak_idx = len(time_vec)-2

        #Loop below basically checks if it's a local max, and greater than cutoff (for all 7 bands)
        #The cutoff automatically assumes the most recent peaks value and then drops off in time. Prevents too many close peaks.

        for f in range(0,7):
            if (power_onset_vecs[f][check_peak_idx] > peaks_cutoff[f] and
                        power_onset_vecs[f][check_peak_idx] > power_onset_vecs[f][check_peak_idx+1] and power_onset_vecs[f][check_peak_idx] > power_onset_vecs[f][check_peak_idx-1]):

                peaks_cutoff[f] = power_onset_vecs[f][check_peak_idx]*2

                peaks_strength[f].append(power_onset_vecs[f][check_peak_idx])
                peaks_time[f].append(cur_time-time_step)

            peaks_cutoff[f]*=.97

        cur_sample = wav.tell()
        cur_window +=1

        #Shift over 256 samples for the next window
        sample_arr = sample_arr[256:]
        sample_arr = np.append(sample_arr,np.fromstring(wav.readframes(512), dtype='int16')[::2])

    song_over = True


def beat_detect_simulate_realtime(wav):
    global detected_beat_times

    initialize_values()

    t0 = time.time() #benchmark

    wav.rewind()
    detected_beat_times = []

    #Will block until the song is done
    acquisition_thread(wav)

    t1 = time.time() #benchmark

    debug("Total algorithm time "+str(t1-t0))

    return detected_beat_times, period_guesses[-1]

#END ALGORITHM

#AUX FUNCS
def grab_known_beats(wavfile):
    known_beats = []
    txtfile_name = PATH + wavfile.split('.')[0] + ".txt"

    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
            if 'str' in line:
                break

    pd = (known_beats[28]-known_beats[20])/8
    debug("The correct period from the text file was " + str(pd))
    return (known_beats, pd)

#SET UP THE PLOT
def plot_beats_and_peaks(wav, found_beats, known_beats):
    global period_guesses, old_period_guesses, beat_guesses, known_pd
    wav.rewind()

    display_div = 50

    plt.figure(1)
    sample_data = wav.readframes(wav.getnframes())
    sample_arr = np.fromstring(sample_data, dtype='int16')
    downsample_arr = sample_arr[::display_div]
    #plt.plot(downsample_arr, color='#aaaaaa')
    plt.grid()
    plt.plot([known_pd] * len(period_guesses), color='#00ff00')
    plt.plot([known_pd * 2] * len(period_guesses), color='#00ff00')
    plt.plot([known_pd / 2] * len(period_guesses), color='#00ff00')
    plt.plot(period_guesses, color='#aaaaaa')
    plt.plot(old_period_guesses, color='#ff0000')
    plt.ylim([min(old_period_guesses), max(old_period_guesses)])
    plt.show()

    for b in beat_guesses:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[-36000,0], 'k-', lw=1.5, color='#aaaaaa')

    if False:
        debug("Showing graph of onset peak times. These are the peaks of onset vectors from low frequency at bottom to high."
                "The red bar underneath is the real beat.")
        for b in peaks_time[0]:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[-40000,-30000], 'k-', lw=1.5, color='green')
        for b in peaks_time[1]:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[-30000,-20000], 'k-', lw=1.5, color='purple')
        for b in peaks_time[2]:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[-20000,-10000], 'k-', lw=1.5, color='blue')
        for b in peaks_time[3]:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[-10000,0], 'k-', lw=1.5, color='green')
        for b in peaks_time[4]:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[0,10000], 'k-', lw=1.5, color='purple')
        for b in peaks_time[5]:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[10000,20000], 'k-', lw=1.5, color='blue')
        for b in peaks_time[6]:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[20000,30000], 'k-', lw=1.5, color='green')
        for b in found_beats:
            xpos = 44100/display_div*b
            plt.plot([xpos,xpos],[40000,50000], 'k-', lw=1.5, color='black')
    for b in known_beats:
        xpos = 44100/display_div*b
        plt.plot([xpos,xpos],[0,39000], 'k-', lw=1.5, color='r')

    ax = plt.gca()
    ax.xaxis.set_ticks([n*44100/display_div for n in range (0,31)])
    ticks = ax.get_xticks()/(44100/display_div)
    ax.set_xticklabels(ticks)
    plt.show()

def run_algorithm(wavfile):
    global known_pd
    wav = wave.open(PATH + wavfile + ".wav")
    debug("Successfully read the wav file: " + wavfile)
    debug("(nchannels, sampwidth, framerate, nframes, comptype, compname)\n" + str(wav.getparams()))

    #Grab the known beat times from the text file with the same name
    known_beats, known_pd = grab_known_beats(wavfile)

    #Get the algorithm's beat times
    found_beats, found_pd = beat_detect_simulate_realtime(wav)
    print(found_beats)

    if DEBUG:
        print("Guessed period (at the end of the song) is",found_pd)
        plot_beats_and_peaks(wav, found_beats, known_beats)

    """print(found_pd, known_pd)
    if (abs(found_pd - known_pd) < CORRECTNESS_THRESHOLD or abs(found_pd*2 - known_pd) < CORRECTNESS_THRESHOLD or
            abs(found_pd/2 - known_pd) < CORRECTNESS_THRESHOLD):
        print(wavfile + " Passed")
        return True
    print(wavfile + " Failed")
    return False"""
    return found_beats


def test():
    results = []
    pref = "open_00"
    for i in range(1,26):
        if i == 10:
            pref = pref[:-1]
        if not run_algorithm(pref+str(i)):
            results.append(i)
    print("These didn't sync bpm properly")
    print(results)
    print(len(results))

def main(argv):
    if len(argv) != 1:
        test()
    else:
        wavfile = argv[0]
        run_algorithm(wavfile)

if __name__ == "__main__":
    main(sys.argv[1:])

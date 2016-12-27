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
import copy
import random
from random import randint
import threading
from datetime import datetime

from Eval import beatEvaluator

time_step = 512.0/44100
PATH = "C:/Users/Creed/Documents/!LightTower/I3CTestWav-master/Algorithm_11_26/all/" #Change to where your wav files are



#These are the ML parameters for tempo. I reset them (the ones below are just some defaults) so the performance is not as good
#as it once was. The parameters need to be retrained which I'll do as soon as I fix the annotations for the custom data set.

#1.5 seconds long
P_PEAK_PICK_ARRAY_SHORT = [
        [[[0.18, 4], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 1, [0], [1], False, [1, 1, 1, 1]]]]

#6 seconds long
P_PEAK_PICK_ARRAY_MED = [
        [[[0.18, 1], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 1], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 1], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 1], 1, [0], [1], False, [1, 1, 1, 1]]],
        [[[0.18, 1], 1, [0], [1], False, [1, 1, 1, 1]]]]

#All
P_PEAK_PICK_ARRAY_LONG = [
        [[[0.18, 4], 2, [0, 0], [5, 1], False, [1, 1, 1, 1]], [[0.375, 0.75], 1, [0], [.8], False, [1, 1, 1, 1]],     [[1.5, 4], 1, [0], [.8], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 2, [0, 0], [1, .5], False, [1, 1, 1, 1]], [[0.375, 0.75], 1, [0], [.8], False, [1, 1, 1, 1]],     [[1.5, 4], 1, [0], [.8], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 2, [0, 0], [1, .5], False, [1, 1, 1, 1]], [[0.375, 0.75], 1, [0], [.8], False, [1, 1, 1, 1]],     [[1.5, 4], 1, [0], [.8], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 2, [0, 0], [3, 1], False, [1, 1, 1, 1]], [[0.375, 0.75], 1, [0], [.8], False, [1, 1, 1, 1]],     [[1.5, 4], 1, [0], [.8], False, [1, 1, 1, 1]]],
        [[[0.18, 4], 2, [0, 0], [.5, .2], False, [1, 1, 1, 1]], [[0.375, 0.75], 1, [0], [.8], False, [1, 1, 1, 1]],      [[1.5, 4], 1, [0], [.4], False, [1, 1, 1, 1]]]]


P_CONSENSUS_TOLERANCE = 0.04
P_DISTINCT_PEAK_TOLERANCE = 0.03

DEBUG_TIME = 17.5
DEBUG_SET = 0

def find_consensus(data, confidence, max_group_size):

    bins = []
    # bins[[val,z],[val,z]...]
    for idx, d in enumerate(data):
        found_bin = False
        for b in bins:
            #3 checks for double/half consideration
            if (confidence[idx] == 0): continue
            if abs(d - b[0]) < max_group_size:
                b[0] = (b[0] * b[1] + d * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                b[2] += 1
                found_bin = True
                break
            elif abs(d*2 - b[0]) < max_group_size:
                b[0] = (b[0] * b[1] + d*2 * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                found_bin = True
                b[2] += 1
                break
            elif abs(d/2 - b[0]) < max_group_size:
                b[0] = (b[0] * b[1] + d/2 * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                found_bin = True
                b[2] += 1
                break
        if (found_bin == False):
            bins.append([d, confidence[idx], 1])

    for b in bins:
        b[1] *= (1)

    return sorted(bins, key=lambda x: float(x[1]), reverse=True)


def get_quarter_note(period):
    while (period > 0.750):
        period /= 2
    while (period < 0.375):
        period *= 2
    return period

def match_periods(pd1,pd2,window):
    return abs(get_quarter_note(pd1)-get_quarter_note(pd2)) < window


def index_to_time(idx):

    return idx * time_step

def time_to_index(time):

    return int(round(time/time_step))

def correlate_onsets(a, b, peak_pick_method, peak_pick_index, corr_range, cur_time,plot_num,debug_plot):
    global DEBUG_SET


    min_idx = corr_range[0]
    max_idx = corr_range[1]

    all_periods_found = []


    # Performs an O(n*logn) correlation
    corr = scipy.signal.fftconvolve(a, b[::-1], mode='full')
    mid = math.floor(len(corr) / 2) #The correlation is symmetric, start here

    # The overall period search space is approx 150 ms to 4000ms (spans eigth notes at 200bpm to measures at 60bpm)
    # Divide by 1000000 to get numbers that are easier to read
    corr_space = corr[mid + min_idx:mid + max_idx] / 1000000
    corr_space_x = np.linspace(min_idx * time_step, max_idx * time_step, max_idx-min_idx)  # Maps indices to time

    # This value will be higher for "live" songs or songs with weird lags and tempo fluctuations


    #Get a score for each peak search range
    for idx,PARAM in enumerate(peak_pick_method[peak_pick_index]):

        time_low = PARAM[0][0]
        time_high = PARAM[0][1]
        search_low_idx = mid + time_to_index(time_low)
        search_high_idx = mid + time_to_index(time_high)
        search_total_indices = search_high_idx-search_low_idx
        corr_search = corr[search_low_idx:search_high_idx] / 1000000
        corr_search_x = np.linspace(index_to_time(search_low_idx - mid), index_to_time(search_high_idx - mid), search_total_indices)  # Maps indices to time

        corr_min = min(corr_search)
        corr_avg = np.average(corr_search)
        corr_max = max(corr_search)

        num_pds_to_find = PARAM[1]
        num_pds_found = 0
        periods_found_in_range = [] #Fill with time.strength pairs

        if PARAM[4] == False:
            check_for_duplicates = periods_found_in_range
        else:
            check_for_duplicates = all_periods_found

        #These are the indices of the max values in the correlation
        best_indices = (np.argsort(corr_search, axis=0)[::-1])[0:20]

        for i in best_indices:
            # See if this is already a selected value (e.g. it's too close or a multiple of another already found period)


            consider_pd = index_to_time(i+search_low_idx-mid)


            pd_is_unique = True
            for other_pd_in_range in check_for_duplicates:
                if (match_periods(consider_pd,other_pd_in_range[0], P_DISTINCT_PEAK_TOLERANCE)):
                    pd_is_unique = False
            if (pd_is_unique):
                #Add the new period, depending on modification mode parameter this may require looking at 2^n multiples (/2, 1, *2, *4) of the period (to see where it is reinforced)

                do_snap = False
                if (consider_pd < 0.375):
                    do_snap = True

                consider_pd_qn_time = get_quarter_note(consider_pd)
                #It should be at a peak, we want to snap to the peak because resolution can be low when multiplying small freqs

                snap_index = time_to_index(consider_pd_qn_time) - 16

                if (do_snap == True):
                    max_snap = 0
                    max_index = snap_index
                    for trav in range(-5,5):
                        if (corr_space[snap_index+trav] > max_snap):
                            max_snap = corr_space[snap_index+trav]
                            max_index = snap_index+trav

                    consider_pd_qn_time = get_quarter_note(index_to_time(max_index + 16))





                consdier_period_qn_strength = corr_space[time_to_index(consider_pd_qn_time)-16] - corr_avg

                found_pd_strength = corr_space[time_to_index(consider_pd)-16]


                if (PARAM[2][num_pds_found] >= 1):
                    num_peaks_in_sum = 0
                    half_pd_strength = unity_pd_strength = double_pd_strength = quad_pd_strength = triple_pd_strength = six_pd_strength = 0

                    half_pd_idx = time_to_index(consider_pd_qn_time/2)-16
                    unity_pd_idx = time_to_index(consider_pd_qn_time) - 16
                    double_pd_idx = time_to_index(consider_pd_qn_time * 2) - 16
                    quad_pd_idx = time_to_index(consider_pd_qn_time * 4) - 16

                    half_pd_strength = (corr_space[half_pd_idx] - corr_min)  +(corr_space[half_pd_idx-1] - corr_min)*.7+(corr_space[half_pd_idx+1] - corr_min)*.7
                    unity_pd_strength = (corr_space[unity_pd_idx] - corr_min)  +(corr_space[unity_pd_idx-1] - corr_min)*.7+(corr_space[unity_pd_idx+1] - corr_min)*.7
                    double_pd_strength = (corr_space[double_pd_idx] - corr_min)  +(corr_space[double_pd_idx-1] - corr_min)*.7+(corr_space[double_pd_idx+1] - corr_min)*.7
                    quad_pd_strength = (corr_space[quad_pd_idx] - corr_min)  +(corr_space[quad_pd_idx-1] - corr_min)*.7+(corr_space[quad_pd_idx+1] - corr_min)*.7

                    #Extra checks for special kinds of songs (extra checks round 1, within bands)

                    # Does it seem to be 3:4 time signiature?
                    triple_pd_index = time_to_index(consider_pd_qn_time * 3) - 16
                    triple_pd_strength = (corr_space[triple_pd_index] - corr_min)

                    six_pd_index = time_to_index(consider_pd_qn_time * 6) - 16
                    six_pd_strength = (corr_space[six_pd_index] - corr_min)

                    mult_3_over_2_score +=  math.pow(six_pd_strength/(quad_pd_strength+.1),2)/cur_time



                    found_pd_strength = (half_pd_strength*PARAM[5][0] + unity_pd_strength * PARAM[5][1] + double_pd_strength * PARAM[5][2] + quad_pd_strength * PARAM[5][3])


                found_pd_strength *=  PARAM[3][num_pds_found]

                periods_found_in_range.append([consider_pd_qn_time, found_pd_strength])
                all_periods_found.append([consider_pd_qn_time, found_pd_strength])



                num_pds_found+=1
                if (num_pds_found == num_pds_to_find):
                    break

        #Added all of the peaks for this range

    known_pd = .462
    # Plot?
    if (DEBUG_SET == 2 and debug_plot):
        #print(all_periods_found)
        plt.subplot(421 + plot_num)
        plt.ylabel("Freq band" + str(plot_num))
        plt.plot(corr_space_x, corr_space, color='b')
        plt.plot([known_pd, known_pd], [corr_min, corr_max], color='r')
        plt.plot([known_pd * 2, known_pd * 2], [corr_min, corr_max ], color='r')
        plt.plot([known_pd * 4, known_pd * 4], [corr_min, corr_max ], color='r')
        plt.plot([known_pd / 2, known_pd / 2], [corr_min, corr_max ], color='r')

        plt.plot([all_periods_found[0][0], all_periods_found[0][0]], [corr_min, corr_max * 1.2], color='purple', lw = '1.5')

        if (len(all_periods_found) > 2):
            plt.plot([all_periods_found[2][0], all_periods_found[2][0]], [corr_min - corr_max * .5, corr_max], color='pink')
        if (len(all_periods_found) > 1):
            plt.plot([all_periods_found[1][0], all_periods_found[1][0]], [corr_min - corr_max * .2, corr_max], color='g')










    return all_periods_found

#will be threaded later, currently called about every 300ms
def tempo_processing_thread(onset_vecs, cur_time):
    global DEBUG_SET

    if (cur_time > DEBUG_TIME and DEBUG_SET == 1):
        DEBUG_SET = 2

    candidate_periods_l = []
    candidate_periods_m = []
    candidate_periods_s = []
    plot_num = 0
    for band in range(0, 5):

        short_periods = correlate_onsets(onset_vecs[band][-130:], onset_vecs[band][-130:], P_PEAK_PICK_ARRAY_SHORT, band, [16,130], cur_time, plot_num, False)

        med_periods = correlate_onsets(onset_vecs[band][-517:], onset_vecs[band][-517:], P_PEAK_PICK_ARRAY_MED, band, [16,345], cur_time, plot_num, False)

        long_periods = correlate_onsets(onset_vecs[band], onset_vecs[band], P_PEAK_PICK_ARRAY_LONG, band, [16,345], cur_time, plot_num, True)


        if (DEBUG_SET == 2): plot_num += 1
        for guess in long_periods:
            candidate_periods_l.append(guess)
        for guess in med_periods:
            candidate_periods_m.append(guess)
        for guess in short_periods:
            candidate_periods_s.append(guess)


    candidate_periods_l = np.array(candidate_periods_l)
    times_l = candidate_periods_l[:, 0]
    strengths_l = candidate_periods_l[:, 1]
    voting_power_l = []
    for s in strengths_l:
        if (s <0):
            s = 0
        vote = s
        voting_power_l.append(vote)
    consensus_period_l = find_consensus(times_l, voting_power_l, P_CONSENSUS_TOLERANCE)
    best_period_l = consensus_period_l[0][0]

    if (cur_time >= 29.5):
        print("tempo consensus at time", cur_time, consensus_period_l)

    candidate_periods_m = np.array(candidate_periods_m)
    times_m = candidate_periods_m[:, 0]
    strengths_m = candidate_periods_m[:, 1]
    voting_power_m = []
    for s in strengths_m:
        if (s < 0):
            s = 0
        vote = s
        voting_power_m.append(vote)
    consensus_period_m = find_consensus(times_m, voting_power_m, P_CONSENSUS_TOLERANCE)
    best_period_m = consensus_period_m[0][0]

    candidate_periods_s = np.array(candidate_periods_s)
    times_s = candidate_periods_s[:, 0]
    strengths_s = candidate_periods_s[:, 1]
    voting_power_s = []
    for s in strengths_s:
        if (s < 0):
            s = 0
        vote = s
        voting_power_s.append(vote)
    consensus_period_s = find_consensus(times_s, voting_power_s, P_CONSENSUS_TOLERANCE)
    best_period_s = consensus_period_s[0][0]




    if (DEBUG_SET == 2):
        DEBUG_SET = 0
        #plt.show()



    #Potential smoothing here

    return (cur_time, best_period_s, best_period_m, best_period_l)




def chroma(x_psd, y_psd, prev_notes, deb):
    #Ignore every thing below 150Hz
    y_psd = y_psd[7:]
    x_psd = x_psd[7:]

    #0 is A, 3 bins per note for 12 notes = 36 total
    notes_presence = np.zeros(36)

    for freq,pow in zip(x_psd,y_psd):
        note = math.floor((math.log(freq/440,2)*36)%36)
        notes_presence[int(note)] += pow

    #Calculate the difference of the power for each note
    chroma_difference = 0
    for prev_note, note in zip(prev_notes, notes_presence):
        chroma_difference += abs(note-prev_note)

    if (deb):
        pass
        #plt.plot(x_psd, y_psd)
        #plt.show()

    return (chroma_difference, notes_presence)






def to_idx(time):
    return int(time/time_step)-3

def acquisition_thread(wav):
    global time_vec, onset_vecs, comb_pows

    found_beats = []

    #period data contains lists of the form [time, short_term_corr_period, med...period, long_term_corr_period]
    #I am working on putting those together to get the best period estimate. For now period_data[-1][3] is the
    #best guess of the current period cuz it's the most recent period from the long term correlation

    period_data = []


    song_over = False
    cur_sample = 0
    cur_window = 0
    start_window = 0
    cur_time = 0
    onset_vecs = np.array([[], [], [], [], []], dtype=np.int)
    prev_onsets = np.zeros(5, dtype=np.int)
    time_vec = []
    comb_pows = []
    comb_times = []
    prev_notes = np.zeros(36)
    started_placing_beats = False

    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while (len(sample_arr) == 1024):

        if (cur_time > 4 and ((cur_window-start_window) % 30 == 0 or start_window == 0)):
            period_data.append(tempo_processing_thread(onset_vecs,cur_time))
            started_placing_beats = True
            start_window = cur_window



        cur_time = cur_sample/44100
        time_vec.append(cur_time)

        windowed = np.hanning(1024)*sample_arr

        #STEP 1: GENERATE ONSET VECTORS THAT WILL BE USED LATER TO FIND THE BEATS

        x_psd, y_psd = scipy.signal.periodogram(windowed, 22050) #the power spectrum

        y_psd = np.sqrt(y_psd) #sqrt power values. get rid of huge differences and prevent int overflow with correlation

        #sum up the ranges
        onsets = np.array([0,0,0,0,0])

        onsets[0] = np.sum(y_psd[0:7])
        onsets[1] = np.sum(y_psd[7:25])
        onsets[2] = np.sum(y_psd[25:187])
        onsets[3] = np.sum(y_psd[187:510])



        debug_chroma = False
        if (cur_window % 30 == 0):
            pass
            #debug_chroma = True

        (chroma_difference, prev_notes) = chroma(x_psd, y_psd,prev_notes,debug_chroma)


        onsets[4] = chroma_difference

        #Now we have the power in each range. Compute the derivative (simple difference) and get rid of negative power changes to get onset vectors
        new_onset_samples = onsets - prev_onsets
        new_onset_samples = new_onset_samples.clip(min=0)
        prev_onsets = onsets



        #Add the new onset values to the vectors so that we have power_onset_vecs[range_num][time_idx]
        new_onset_samples = new_onset_samples.reshape((5,1))
        onset_vecs = np.hstack([onset_vecs,new_onset_samples])


        if (cur_window %30 == 0):
            pass
            #plt.plot(time_vec, onset_vecs[0], color = 'r')
            #plt.plot(time_vec, onset_vecs[4], color='b')
            #plt.plot(time_vec, onset_vecs[3], color = 'r')
            #plt.show()

        cur_sample = wav.tell()
        cur_window +=1

        #Shift over 256 samples for the next window
        sample_arr = sample_arr[256:]
        sample_arr = np.append(sample_arr,np.fromstring(wav.readframes(512), dtype='int16')[::2])


    found_beats = [1,2,3,4,5,6,7,8,9,10]

    return found_beats, period_data

def beat_detect_simulate_realtime(wav):

    t0 = time.time() #benchmark

    wav.rewind()

    #Will block until the song is done
    (found_beats, period_data) = acquisition_thread(wav)

    t1 = time.time() #benchmark

    return found_beats, period_data

def plot_results(wav, song_name,found_beats, period_data):

    #get known_pds and known_beats so we can evaluate performance against them
    known_pds = []
    known_beats = []
    txtfile_name = PATH + song_name.split('.')[0] + ".txt"
    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
            if 'str' in line:
                break



    for i in range(1, len(known_beats)):
        known_pds.append((known_beats[i],known_beats[i] - known_beats[i - 1]))

    #plot the found periods against known periods
    plt.subplot(311).set_title("Top to bottom: Long (all), medium (6s), short term (1.5s) correlation tempo results")


    known_pds_time = [x[0] for x in known_pds]
    known_pds = [x[1] for x in known_pds]

    found_pds_time = [x[0] for x in period_data]
    found_pds_s = [x[1] for x in period_data]
    found_pds_m = [x[2] for x in period_data]
    found_pds_l = [x[3] for x in period_data]

    plt.plot(known_pds_time, known_pds, color = 'r')
    plt.plot(found_pds_time, found_pds_l, color = 'b')

    plt.subplot(312)


    plt.plot(known_pds_time, known_pds, color='r')
    plt.plot(found_pds_time, found_pds_m, 'r--', color='g')

    plt.subplot(313)

    plt.plot(known_pds_time, known_pds, color='r')
    plt.plot(found_pds_time, found_pds_s, 'r--', color='purple')

    #plot the found beats against known beats with the waveform
    wav.rewind()
    display_div = 50

    plt.show()


    sample_data = wav.readframes(wav.getnframes())
    sample_arr = np.fromstring(sample_data, dtype='int16')
    downsample_arr = sample_arr[::display_div]
    plt.plot(downsample_arr, color='#dddddd')
    plt.grid()


    for b in known_beats:
        xpos = 44100 / display_div * b
        plt.plot([xpos, xpos], [-20000, 35000], 'k-', lw=1.5, color='pink')
    ax = plt.gca()
    ax.xaxis.set_ticks([n * 44100 / display_div for n in range(0, 31)])
    ticks = ax.get_xticks() / (44100 / display_div)
    ax.set_xticklabels(ticks)

    for b in found_beats:
        xpos = 44100 / display_div * b
        plt.plot([xpos, xpos], [-25000, 30000], 'k-', lw=1.5, color='cyan')
    ax = plt.gca()
    ax.xaxis.set_ticks([n * 44100 / display_div for n in range(0, 31)])
    ticks = ax.get_xticks() / (44100 / display_div)
    ax.set_xticklabels(ticks)

    #Add other stuff to the plot here (onset graphs, etc.)

    plot_time = [x * 44100 / display_div for x in time_vec]

    plt.plot(plot_time, onset_vecs[0]*5, color='b')
    plt.show()


    plt.show()

def test_song(song_name):
    global DEBUG_SET
    DEBUG_SET = 1

    wav = wave.open(PATH + song_name + ".wav")
    print("Successfully read the wav file: " + song_name)

    #Run the algorithm
    found_beats, period_data = beat_detect_simulate_realtime(wav)

    #Check the performance
    plot_results(wav, song_name, found_beats, period_data)





def testbed():


    pref = "open_00"
    for i in range(1, 26):
        if i == 10:
            pref = pref[:-1]
        test_song(pref + str(i))

    pref = "challenge_00"
    for i in range(1, 34):
        if i == 10:
            pref = pref[:-1]
        test_song(pref + str(i))

    pref = "closed_00"
    for i in range(1, 26):
        if i == 10:
            pref = pref[:-1]
        test_song(pref + str(i))










def main(argv):
    testbed()

if __name__ == "__main__":
    main(sys.argv[1:])

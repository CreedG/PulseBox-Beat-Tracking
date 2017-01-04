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

import cProfile


#Change to where the wav files are located.
PATH = "C:/Users/Creed/Documents/!LightTower/I3CTestWav-master/Algorithm_11_26/all/" #Change to where your wav files are


time_step = 256.0/22050

total_incorrect = 0
total_score = 0
num_scores = 0


#Parameters that were varied to train the algorithm

'''The peak pick arrays are used when finding the period of the song. They represent the weightings of different frequency ranges in determining the period.'''

#Spans most recent 17 seconds
P_PEAK_PICK_ARRAY_LONG = [[[[0.18, 4], 2, [0, 0], [0.4637504524433074, 0.22809289464349886], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [0.6003130885154166], False, [1, 1, 1, 1]]],
                          [[[0.18, 4], 2, [0, 0], [0.18210559730152082, 0.1637055231406911], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [5.842614764989088], False, [1, 1, 1, 1]]],
                          [[[0.18, 4], 2, [0, 0], [1.0117249240325608, 0.351551594852886], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [1.409717359426398], False, [1, 1, 1, 1]]],
                          [[[0.18, 4], 2, [0, 0], [3.881593456113786, 1.4187772268270373], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [1.2205340254264134], False, [1, 1, 1, 1]]],
                          [[[0.18, 4], 2, [0, 0], [1.2649516398093676, 0.0052008308469825865], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [0.03732238499117998], False, [1, 1, 1, 1]]]]
#Spans most recent 6 seconds
P_PEAK_PICK_ARRAY_MED= [[[[0.18, 1], 1, [0], [0.062099218684597376], False, [1, 1, 1, 1]]],
                         [[[0.18, 1], 1, [0], [1.5753151334669286], False, [1, 1, 1, 1]]],
                         [[[0.18, 1], 1, [0], [1.1637075593024846], False, [1, 1, 1, 1]]],
                         [[[0.18, 1], 1, [0], [19.83503952234608], False, [1, 1, 1, 1]]],
                         [[[0.18, 1], 1, [0], [0.017411179300045182], False, [1, 1, 1, 1]]]]

#Spans most recent 3 seconds
P_PEAK_PICK_ARRAY_SHORT = [[[[0.18, 4], 2, [0, 0], [0.4637504524433074, 0.23183151247396472], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [1.0265975602231432], False, [1, 1, 1, 1]]],
                           [[[0.18, 4], 2, [0, 0], [0.10600325904505178, 0.11025863779222567], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [1.987351968103246], False, [1, 1, 1, 1]]],
                           [[[0.18, 4], 2, [0, 0], [1.0746105972730395, 0.3642503639087244], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [1.2603563519577021], False, [1, 1, 1, 1]]],
                           [[[0.18, 4], 2, [0, 0], [5.384319540310969, 1.1746500524488648], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [0.5021989052809293], False, [1, 1, 1, 1]]],
                           [[[0.18, 4], 2, [0, 0], [0.7270916362492784, 0.005458336170681191], False, [1, 1, 1, 1]], [[1.5, 4], 1, [0], [0.04874661415392632], False, [1, 1, 1, 1]]]]

P_CONSENSUS_TOLERANCE = 0.04
P_DISTINCT_PEAK_TOLERANCE = 0.03

DEBUG_TIME = 15
DEBUG_SET = 1


#Given multiple values, place values near each other into bins
def find_consensus(data, confidence, size_within_group):

    #bins are [value,confidence,delta, num ele]
    bins = []
    # bins[[val,z],[val,z]...]
    for idx, d in enumerate(data):
        found_bin = False
        for b in bins:
            #3 checks for double/half consideration
            if (confidence[idx] == 0): continue
            if abs(d - b[0]) < size_within_group:
                b[0] = (b[0] * b[1] + d * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                b[2] += (d-b[0])
                b[3] += 1
                found_bin = True
                break
            elif abs(d - b[0]/2) < size_within_group:
                b[0] = (b[0] * b[1] + d*2 * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                b[2] += (d-b[0])
                b[3] += 1
                found_bin = True
                break
            elif abs(d/2 - b[0]) < size_within_group:
                b[0] = (b[0] * b[1] + d/2 * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                b[2] += (d-b[0])
                b[3] += 1
                found_bin = True
                break
        if (found_bin == False):
            bins.append([d, confidence[idx], 0, 1])

    for b in bins:
        b[2] /= b[3]

    return sorted(bins, key=lambda x: float(x[1]), reverse=True)


def get_quarter_note(period):
    while (period > 0.750):
        period /= 2
    while (period < 0.375):
        period *= 2
    return period


def period_is_multiple(pd1,pd2,window):
    return abs(get_quarter_note(pd1)-get_quarter_note(pd2)) < window


def index_to_time(idx):

    return idx * time_step

def time_to_index(time):

    return int(round(time/time_step))


#Perform an autocorrelation for a given frequency range and pick out the peaks using the weightings from the P_PEAK_PICK_ARRAY parameters
def correlate_onsets(a, b, peak_pick_method, peak_pick_index, corr_range):

    min_idx = corr_range[0]
    max_idx = corr_range[1]
    all_periods_found = []

    # Perform an O(n*logn) correlation
    corr = scipy.signal.fftconvolve(a, b[::-1], mode='full')
    mid = math.floor(len(corr) / 2) #The correlation is symmetric, start here

    # Define the range of period values to examine
    corr_space = corr[mid + min_idx:mid + max_idx] / 1000000

    #Find peaks and determine their voting power, which will eventually combine to find the best overall period
    for idx,PARAM in enumerate(peak_pick_method[peak_pick_index]):

        time_low = PARAM[0][0]
        time_high = PARAM[0][1]
        search_low_idx = mid + time_to_index(time_low)
        search_high_idx = mid + time_to_index(time_high)

        corr_search = corr[search_low_idx:search_high_idx] / 1000000
        corr_avg = np.average(corr_search)

        num_pds_to_find = PARAM[1]
        num_pds_found = 0
        periods_found_in_range = []


        check_for_duplicates = periods_found_in_range

        #These are the indices of the max values in the correlation
        best_indices = (np.argsort(corr_search, axis=0)[::-1])[0:20]

        for i in best_indices:
            # See if this is already a selected value (e.g. it's too close or a multiple of another already found period)
            consider_pd = index_to_time(i+search_low_idx-mid)

            pd_is_unique = True
            for other_pd_in_range in check_for_duplicates:
                if (period_is_multiple(consider_pd,other_pd_in_range[0], P_DISTINCT_PEAK_TOLERANCE)):
                    pd_is_unique = False
            if (pd_is_unique):

                #Ge the quarter note time that this peak corresponds to, may require snapping to a local max
                consider_pd_qn_time = get_quarter_note(consider_pd)

                do_snap = (consider_pd < 0.375)
                snap_index = time_to_index(consider_pd_qn_time) - 16
                if (do_snap):
                    max_snap = 0
                    max_index = snap_index
                    for trav in range(-5, 5):
                        if (corr_space[snap_index + trav] > max_snap):
                            max_snap = corr_space[snap_index + trav]
                            max_index = snap_index + trav

                    consider_pd_qn_time = get_quarter_note(index_to_time(max_index + 16))


                #The strength is the power multiplied by the weighting from the P_PEAK_PICK_ARRAY parameter
                found_pd_strength = corr_space[time_to_index(consider_pd)-16] * PARAM[3][num_pds_found]

                periods_found_in_range.append([consider_pd_qn_time, found_pd_strength])
                all_periods_found.append([consider_pd_qn_time, found_pd_strength])

                num_pds_found+=1
                if (num_pds_found == num_pds_to_find):
                    break

    #Return all of the periods for this frequency range and their voting power
    return all_periods_found



def are_periods_related(p1,p2):
    return ((abs(p1 / p2 - 1) < .03) or
            (abs(p1 / p2 - .5) < .03) or
            (abs(p1 / p2 - 2) < .03) or
            (abs(p1 / p2 - 1.333) < .03) or
            (abs(p1 / p2 - 0.75) < .03))


#Called about every 300ms, updates the tempo
def tempo_processing_thread(onset_vecs, cur_time):
    global period_data, tempo_derivative, tempo_instability, found_med, uncertainty, tag_use_med_correlation, tag_use_short_correlation

    candidate_periods_l, candidate_periods_m, candidate_periods_s = [], [], []

    #Get the tempo votes from the short, medium, and long term correlations
    for freq_band in range(0, 5):

        long_periods = correlate_onsets(onset_vecs[freq_band][-1500:], onset_vecs[freq_band][-1500:], P_PEAK_PICK_ARRAY_LONG, freq_band, [16, 345])
        med_periods = correlate_onsets(onset_vecs[freq_band][-500:], onset_vecs[freq_band][-500:], P_PEAK_PICK_ARRAY_MED, freq_band, [16, 345])
        short_periods = correlate_onsets(onset_vecs[freq_band][-250:], onset_vecs[freq_band][-250:], P_PEAK_PICK_ARRAY_SHORT, freq_band, [16,300])

        for guess in long_periods:
            candidate_periods_l.append(guess)
        for guess in med_periods:
            candidate_periods_m.append(guess)
        for guess in short_periods:
            candidate_periods_s.append(guess)


    candidate_periods_array = [candidate_periods_l, candidate_periods_m, candidate_periods_s]
    best_period_votes = []

    #Extract the best tempo estimate from each of the short, medium, and long term correlations

    for candidate_periods in candidate_periods_array:
        candidate_periods =  np.array(candidate_periods)
        times = candidate_periods[:, 0]
        strengths = candidate_periods[:, 1]
        voting_power = []
        for s in strengths:
            if (s < 0):
                s = 0
            vote = s
            voting_power.append(vote)
        consensus_period = find_consensus(times, voting_power, P_CONSENSUS_TOLERANCE)
        best_period_votes.append(consensus_period[0][0])

    best_period_l = best_period_votes[0]
    best_period_m = best_period_votes[1]
    best_period_s = best_period_votes[2]


    #Now integrate these short, medium, and long term period guesses by smoothing and combining them
    weights = np.linspace(1, 1, 3)
    groups =  find_consensus([best_period_l,best_period_m,best_period_s], weights, .05)
    best_period_combined =  groups[0][0]

    if (len(period_data[3]) > 0):

        if (cur_time < 6.5):
            best_period_combined = best_period_l
        else:
            weights = [1,.45,.45]
            groups = find_consensus([best_period_l, best_period_m, best_period_s], weights, .05)
            best_period_combined = groups[0][0]

            long_groups = find_consensus(period_data[3][-10:], np.linspace(1, 1, 10), .015)
            short_and_med_groups = find_consensus(period_data[1][-10:]+period_data[2][-10:], np.linspace(1, 1, 24), .015)

            if (cur_time < 20 and len(short_and_med_groups) == 1 and len(long_groups) >= 2):
                if (are_periods_related(long_groups[0][0],(long_groups[1][0])) == False and
                        len(find_consensus([long_groups[1][0], short_and_med_groups[0][0]], np.linspace(1, 1, 2), .015)) == 2):
                    #If the medium and short term correlations agree, maybe the song is changing too much so use medium correlation
                    tag_use_med_correlation = True


        #Look for the tempo moving up or down, if this happens add to tempo_instability
        tempo_derivative.append((best_period_s-(period_data[1][-1])))
        dpos = len(tempo_derivative)-1
        stretch_len = 1
        total_change = tempo_derivative[dpos]
        while (cur_time <= 15 and abs(tempo_derivative[dpos]) < .1 and abs(total_change/stretch_len) > .01):
            stretch_len += 1
            if (dpos >= 1):
                dpos -= 1
            else:
                break
            if (stretch_len > 3):
                if (abs(tempo_derivative[dpos]+tempo_derivative[dpos+1]+tempo_derivative[dpos+2]+tempo_derivative[dpos+3]) < .02):
                    break
            total_change += tempo_derivative[dpos]
        if (stretch_len > tempo_instability):
            tempo_instability = stretch_len

    #If the long term tempo is uncertain near the beginning of the song, add to instability
    if (cur_time > 10 and cur_time < 15):
        if (are_periods_related(best_period_l,period_data[3][-1]) == False):
            tempo_instability += 3

    if (tempo_instability > 10):
        tag_use_short_correlation = True
    else:
        tag_use_short_correlation = False

    #Storage for all the vectors

    period_data[0].append(cur_time)
    period_data[1].append(best_period_s)
    period_data[2].append(best_period_m)
    period_data[3].append(best_period_l)
    period_data[4].append(best_period_combined)


    #A few different strategies for smoothing and combining to get the tempo, depending on how unstable the tempo was
    if (tag_use_short_correlation == False):
        if (tag_use_med_correlation == False):
            if (cur_time < 6.5):
                ultimate_period = period_data[4][-1]
            else:
                weights = np.linspace(1, 1, len(period_data[4]))
                groups = find_consensus(period_data[4], weights, .05)
                ultimate_period = groups[0][0]
        else:
            weights = np.linspace(1, 1, len(period_data[2]))
            groups = find_consensus(period_data[2], weights, .05)
            ultimate_period = groups[0][0]
    else:
        weights = np.linspace(1, 1, 4)
        groups = find_consensus(period_data[1][-4:], weights, .05)
        ultimate_period = groups[0][0]

    #Prevent jumping around between double and half
    if (len(period_data[5]) > 0):
        if (period_data[5][-1] < 0.4 and abs(ultimate_period / period_data[5][-1] - 2) < .1):
            ultimate_period /= 2
        if (period_data[5][-1] > 0.7 and abs(period_data[5][-1] / ultimate_period - 2) < .1):
            ultimate_period *= 2

    #The final result of the period processing
    period_data[5].append(ultimate_period)



def to_idx(time):
    return int(time/time_step)-3

def to_time(idx):
    return (idx+3)*time_step

def acquisition_thread(wav):
    global time_vec, onset_vecs, comb_times, comb_pows, period_data, tempo_derivative, tempo_instability, found_med, uncertainty, tag_use_short_correlation, tag_use_med_correlation

    tempo_derivative = []
    known_beats = []
    txtfile_name = PATH + name_of_song.split('.')[0] + ".txt"
    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
            if 'str' in line:
                break


    found_beats = []
    period_data = [[],[],[],[],[],[],[]]

    tempo_instability = 0

    cur_sample = 0
    cur_window = 0
    start_window = 0
    cur_time = 0
    onset_vecs = np.array([[], [], [], [], []], dtype=np.int)
    prev_onsets = np.zeros(5, dtype=np.int)
    time_vec = []
    comb_pows = []
    comb_times = []
    found_med = False

    tag_use_short_correlation = False
    tag_use_med_correlation = False




    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while (len(sample_arr) == 1024):


        if (cur_time > 4 and ((cur_window-start_window) % 30 == 0 or start_window == 0)):
            tempo_processing_thread(onset_vecs,cur_time)
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

        #Let the fifth onset just be the overall power (abandoned chroma since it was taking too long)
        onsets[4] = onsets[3]+onsets[2]+onsets[1]+onsets[0]

        #Now we have the power in each range. Compute the derivative (simple difference) and get rid of negative power changes to get onset vectors
        new_onset_samples = onsets - prev_onsets

        #Get rid of the big spike at the beginning from silewhich can make results less reliable
        if (len(onset_vecs[4]) == 0):
            new_onset_samples[0] /= 3
            new_onset_samples[1] /= 3
            new_onset_samples[2] /= 3
            new_onset_samples[3] /= 3
            new_onset_samples[4] /= 3


        new_onset_samples = new_onset_samples.clip(min=0)
        prev_onsets = onsets

        #Add the new onset values to the vectors so that we have power_onset_vecs[range_num][time_idx]
        new_onset_samples = new_onset_samples.reshape((5,1))
        onset_vecs = np.hstack([onset_vecs,new_onset_samples])

        cur_sample = wav.tell()
        cur_window +=1

        #Shift over 256 samples for the next window
        sample_arr = sample_arr[256:]
        sample_arr = np.append(sample_arr,np.fromstring(wav.readframes(512), dtype='int16')[::2])

    return found_beats, period_data


def beat_detect_simulate_realtime(wav):

    t0 = time.time() #benchmark
    wav.rewind()

    (found_beats, period_data) = acquisition_thread(wav)

    t1 = time.time() #benchmark

    return found_beats, period_data


def process_results(wav, song_name,found_beats, period_data):
    global total_incorrect, total_score, num_scores

    # get known_pds and known_beats so we can evaluate performance against them
    known_pds = []
    known_beats = []
    smoothed_known = []
    txtfile_name = PATH + song_name.split('.')[0] + ".txt"
    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))

            sum = 0
            tally = 0
            for i in range(len(known_beats) - 5, len(known_beats)):
                if (i >= 0):
                    sum += known_beats[i]
                    tally += 1
            sum = float(sum) / tally
            smoothed_known.append(sum)
            if 'str' in line:
                break

    #known_beats = smoothed_known

    # get the official score
    if (True):
        n = 0
        main, backup = beatEvaluator(found_beats, known_beats)
        print("Beats detected:", found_beats)
        print("Score:",main)

        total_score += main
        num_scores += 1

    #get the period score
    if (True):


        found_pds = period_data[5]
        found_pds_time = period_data[0]

        for i in range(1, len(known_beats)):
            known_pds.append((known_beats[i], known_beats[i] - known_beats[i - 1]))

        known_pds_time = [x[0] for x in known_pds]
        known_pds = [x[1] for x in known_pds]

        incorrect_count = 0

        known_pds_index = 0

        for i in range(0, len(found_pds)):
            time = found_pds_time[i]
            while (known_pds_time[known_pds_index] < time and known_pds_index < len(known_pds) - 1):
                known_pds_index += 1
            # print (known_pds_time[known_pds_index],known_pds[known_pds_index],"...",time,found_pds[i])
            if (time >= 5):
                if (abs(known_pds[known_pds_index] - found_pds[i]) > 0.04 and abs(
                                known_pds[known_pds_index] / 2 - found_pds[i]) > 0.04 and abs(
                        known_pds[known_pds_index] - found_pds[i] / 2) > 0.04):
                    # print("inc at ",time)
                    incorrect_count += 1

        print("incorrect periods: ", incorrect_count, "/", len(found_pds))
        #print(tempo_instability)

        total_incorrect += incorrect_count


def test_song(song_name):
    global name_of_song, incorrect_count, total_incorrect
    DEBUG_SET = 1

    name_of_song = song_name

    wav = wave.open(PATH + song_name + ".wav")

    print("Successfully read the wav file: " + song_name)

    #Run the algorithm
    found_beats, period_data = beat_detect_simulate_realtime(wav)

    #Placeholder
    found_beats = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

    process_results(wav, song_name, found_beats, period_data)




def testbed():

    pref = "challenge_00"
    for i in range(1, 34):
        if i == 10:
            pref = pref[:-1]
        if (i != 12):
            test_song(pref + str(i))

    pref = "closed_00"
    for i in range(1, 26):
        if i == 10:
            pref = pref[:-1]
        test_song(pref + str(i))


    print("TOTAL average score for all songs:",total_score/num_scores)



def main(argv):
    testbed()

if __name__ == "__main__":
    main(sys.argv[1:])

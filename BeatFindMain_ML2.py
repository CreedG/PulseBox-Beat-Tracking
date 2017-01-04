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

#When set to true, runs all songs and reports values
SIMULATE_REALTIME = True

#False
DEBUG_PLOTS = False

time_step = 256.0/22050


'''Parameters that were varied to train the algorithm.'''

#Tempo related parameters

#The peak pick arrays are used when finding the period of the song. They represent the weightings of different frequency ranges in determining the period.
#Spans most recent 17 seconds
P_PEAK_PICK_ARRAY_LONG = [[[[0.18, 4], 2, [0, 0], [0.4637504524433074, 0.22809289464349886]], [[1.5, 4], 1, [0], [0.6003130885154166]]],
                          [[[0.18, 4], 2, [0, 0], [0.18210559730152082, 0.1637055231406911]], [[1.5, 4], 1, [0], [5.842614764989088]]],
                          [[[0.18, 4], 2, [0, 0], [1.0117249240325608, 0.351551594852886]], [[1.5, 4], 1, [0], [1.409717359426398]]],
                          [[[0.18, 4], 2, [0, 0], [3.881593456113786, 1.4187772268270373]], [[1.5, 4], 1, [0], [1.2205340254264134]]],
                          [[[0.18, 4], 2, [0, 0], [1.2649516398093676, 0.0052008308469825865]], [[1.5, 4], 1, [0], [0.03732238499117998]]]]
#Spans most recent 6 seconds
P_PEAK_PICK_ARRAY_MED= [[[[0.18, 1], 1, [0], [0.062099218684597376]]],
                         [[[0.18, 1], 1, [0], [1.5753151334669286]]],
                         [[[0.18, 1], 1, [0], [1.1637075593024846]]],
                         [[[0.18, 1], 1, [0], [19.83503952234608]]],
                         [[[0.18, 1], 1, [0], [0.017411179300045182]]]]

#Spans most recent 3 seconds
P_PEAK_PICK_ARRAY_SHORT = [[[[0.18, 4], 2, [0, 0], [0.4637504524433074, 0.23183151247396472]], [[1.5, 4], 1, [0], [1.0265975602231432]]],
                           [[[0.18, 4], 2, [0, 0], [0.10600325904505178, 0.11025863779222567]], [[1.5, 4], 1, [0], [1.987351968103246]]],
                           [[[0.18, 4], 2, [0, 0], [1.0746105972730395, 0.3642503639087244]], [[1.5, 4], 1, [0], [1.2603563519577021]]],
                           [[[0.18, 4], 2, [0, 0], [5.384319540310969, 1.1746500524488648]], [[1.5, 4], 1, [0], [0.5021989052809293]]],
                           [[[0.18, 4], 2, [0, 0], [0.7270916362492784, 0.005458336170681191]], [[1.5, 4], 1, [0], [0.04874661415392632]]]]

P_PEAK_CONSENSUS_TOLERANCE = 0.04
P_DISTINCT_PEAK_TOLERANCE = 0.03
P_TEMPO_START = 6.5
P_MED_WEIGHT = 0.45
P_SHORT_WEIGHT = 0.45
P_COMBINE_TEMPOS_THRESH = 0.05
P_MAX_TIME_MED_CORR = 20
P_LONG_TEMPO_CHANGE_PENALTY = 3
P_SHORT_CORR_THRESH = 10
P_TEMPO_CONSENSUS_TOLERANCE = 0.05
P_TEMPO_WEIGHT_START = 1



#Acquisition related parameters
P_FREQ_BAND_1 = 7
P_FREQ_BAND_2 = 25
P_FREQ_BAND_3 = 187


#Beat placement parameters
P_SNAP_THRESH_1 = 11
P_SNAP_THRESH_2 = 7
P_SNAP_THRESH_3 = 1

P_BEAT_START_MULT = 1.2
P_BEAT_SHIFT = 0.02
P_BEAT_THRESH_START = 0.85
P_BEAT_THRESH_DECAY = 4
P_BEAT_THRESH_UNSTABLE = 12

P_BEAT_START_PERCENT_STABLE = 0.5
P_BEAT_START_PERCENT_UNSTABLE = 0.2
P_BEAT_END_PERCENT_STABLE = 1.5
P_BEAT_END_PERCENT_UNSTABLE = 1.6


ALL_PARAMETERS = [P_PEAK_CONSENSUS_TOLERANCE, P_DISTINCT_PEAK_TOLERANCE, P_TEMPO_START, P_MED_WEIGHT, P_SHORT_WEIGHT,
                  P_COMBINE_TEMPOS_THRESH, P_MAX_TIME_MED_CORR, P_LONG_TEMPO_CHANGE_PENALTY, P_SHORT_CORR_THRESH, P_TEMPO_CONSENSUS_TOLERANCE,
                  P_TEMPO_WEIGHT_START, P_FREQ_BAND_1, P_FREQ_BAND_2, P_FREQ_BAND_3, P_SNAP_THRESH_1,
                  P_SNAP_THRESH_2, P_SNAP_THRESH_3, P_BEAT_START_MULT, P_BEAT_SHIFT, P_BEAT_THRESH_START,
                  P_BEAT_THRESH_DECAY, P_BEAT_THRESH_UNSTABLE, P_BEAT_START_PERCENT_STABLE, P_BEAT_START_PERCENT_UNSTABLE, P_BEAT_END_PERCENT_STABLE,
                  P_BEAT_END_PERCENT_UNSTABLE]


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
    return ((abs(p1 / p2 - 1) < P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - .5) < P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - 2) < P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - 1.333) < P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - 0.75) < P_DISTINCT_PEAK_TOLERANCE))


#Called about every 300ms, updates the tempo
def tempo_processing_thread(onset_vecs, cur_time):
    global period_data, tempo_derivative, tempo_instability, tag_use_short_correlation, tag_use_med_correlation

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
        consensus_period = find_consensus(times, voting_power, P_PEAK_CONSENSUS_TOLERANCE)
        best_period_votes.append(consensus_period[0][0])

    best_period_l = best_period_votes[0]
    best_period_m = best_period_votes[1]
    best_period_s = best_period_votes[2]

    #Now integrate these short, medium, and long term period guesses by smoothing and combining them
    weights = np.linspace(1, 1, 3)
    groups =  find_consensus([best_period_l,best_period_m,best_period_s], weights, .05)
    best_period_combined =  groups[0][0]

    if (len(period_data[3]) > 0):

        if (cur_time < P_TEMPO_START):
            best_period_combined = best_period_l
        else:
            weights = [1,P_MED_WEIGHT,P_SHORT_WEIGHT]
            groups = find_consensus([best_period_l, best_period_m, best_period_s], weights, P_COMBINE_TEMPOS_THRESH)
            best_period_combined = groups[0][0]

            long_groups = find_consensus(period_data[3][-10:], np.linspace(1, 1, 10), .015)
            short_and_med_groups = find_consensus(period_data[1][-10:]+period_data[2][-10:], np.linspace(1, 1, 24), .015)

            if (cur_time < P_MAX_TIME_MED_CORR and len(short_and_med_groups) == 1 and len(long_groups) >= 2):
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
            tempo_instability += P_LONG_TEMPO_CHANGE_PENALTY

    if (tempo_instability > P_SHORT_CORR_THRESH):
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
            if (cur_time < P_TEMPO_START):
                ultimate_period = period_data[4][-1]
            else:
                weights = np.linspace(P_TEMPO_WEIGHT_START, 1, len(period_data[4]))
                groups = find_consensus(period_data[4], weights, P_TEMPO_CONSENSUS_TOLERANCE)
                ultimate_period = groups[0][0]
        else:
            weights = np.linspace(P_TEMPO_WEIGHT_START, 1, len(period_data[2]))
            groups = find_consensus(period_data[2], weights, P_TEMPO_CONSENSUS_TOLERANCE)
            ultimate_period = groups[0][0]
    else:
        weights = np.linspace(1, 1, 4)
        groups = find_consensus(period_data[1][-4:], weights, P_TEMPO_CONSENSUS_TOLERANCE)
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

def main_thread(wav):
    global period_data, tempo_derivative, tempo_instability, tag_use_short_correlation, tag_use_med_correlation

    #INITIALIZE VARS

    found_beats = []
    period_data = [[],[],[],[],[],[],[]]

    #1. Audio acquisition
    cur_sample = 0
    cur_window = 0
    start_window = 0
    cur_time = 0
    onset_vecs = np.array([[], [], [], [], []], dtype=np.int)
    prev_onsets = np.zeros(5, dtype=np.int)
    time_vec = []

    #2. Period finding
    tempo_instability = 0
    tempo_derivative = []
    tag_use_short_correlation = False
    tag_use_med_correlation = False

    #3. Beat finding
    prev_beat_guess = 0
    tentative_prev_time = 0
    plotting = False
    started_placing_beats = False
    prev_thresh = 0
    first_beat_selected = False
    beat_thresh = 0
    beat_max = 0
    comb_pows = []
    comb_times = []

    # 1. PERFORM AUDIO ACQUISITION
    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while (len(sample_arr) == 1024):

        # 2. CALCULATE TEMPO EVERY ~350ms
        if (cur_time > 4 and ((cur_window - start_window) % 30 == 0 or start_window == 0)):
            tempo_processing_thread(onset_vecs, cur_time)
            started_placing_beats = True
            start_window = cur_window

        cur_time = cur_sample / 44100
        time_vec.append(cur_time)

        windowed = np.hanning(1024) * sample_arr

        #Generate onset vectors

        x_psd, y_psd = scipy.signal.periodogram(windowed, 22050)  # the power spectrum

        y_psd = np.sqrt(y_psd)  # sqrt power values. get rid of huge differences and prevent int overflow with correlation

        # sum up the ranges
        onsets = np.array([0, 0, 0, 0, 0])

        onsets[0] = np.sum(y_psd[0:P_FREQ_BAND_1])
        onsets[1] = np.sum(y_psd[P_FREQ_BAND_1:P_FREQ_BAND_2])
        onsets[2] = np.sum(y_psd[P_FREQ_BAND_2:P_FREQ_BAND_3])
        onsets[3] = np.sum(y_psd[P_FREQ_BAND_3:510])

        onsets[4] = onsets[3] + onsets[2] + onsets[1] + onsets[0]

        # Now we have the power in each range. Compute the derivative (simple difference) and get rid of negative power changes to get onset vectors
        new_onset_samples = onsets - prev_onsets

        if (len(onset_vecs[4]) == 0):
            new_onset_samples[0] /= 3
            new_onset_samples[1] /= 3
            new_onset_samples[2] /= 3
            new_onset_samples[3] /= 3
            new_onset_samples[4] /= 3
        new_onset_samples = new_onset_samples.clip(min=0)
        prev_onsets = onsets

        # Add the new onset values to the vectors so that we have power_onset_vecs[range_num][time_idx]
        new_onset_samples = new_onset_samples.reshape((5, 1))
        onset_vecs = np.hstack([onset_vecs, new_onset_samples])


        # 3. BEAT OFFSET FINDING USING TEMPO

        # Beat finding uses a comb filter approach. Add up powers as you backtrack to the beginning of the song.
        # Snap to the nearest x samples (each sample is 11ms) to deal with tempo error.

        #This is the maximum distance to snap to the nearest peak when backtracking. Set it higher if the tempo is unstable
        if (tempo_instability >= P_SNAP_THRESH_1):
            snap_range = 7
        elif (tempo_instability >= P_SNAP_THRESH_2):
            snap_range = 6
        elif (tempo_instability >= P_SNAP_THRESH_3):
            snap_range = 5
        else:
            snap_range = 4

        if (started_placing_beats):
            #Grab the period from the period estimator
            best_pd = period_data[5][-1]

            examine_time = cur_time
            comb_pow = 0

            #If snapping distance is farther, you don't have to run it as often. do snap_range-2 for a little overlap
            if ((cur_window - 346) % snap_range-2 == 0):

                if (plotting):
                    plt.subplot(4, 2, 1 + ((cur_window - 346) % (snap_range * 8)) / snap_range)
                    plt.plot(time_vec, onset_vecs[4] / 100, '--', color='red')

                #back_num is how many steps we've taken back
                back_num = 0
                #snapped_time is where we are starting this iteration
                snapped_time = 0


                while (examine_time > 0):
                    index = to_idx(examine_time)

                    best_index_val = 0
                    best_index = index - snap_range + 1

                    check_max = snap_range
                    if (index >= len(onset_vecs[4]) - snap_range):
                        check_max = 1

                    for i in range(-snap_range + 1, check_max):
                        if (onset_vecs[4][index + i] > best_index_val):
                            best_index_val = onset_vecs[4][index + i]
                            best_index = index + i

                    examine_time = to_time(best_index) - best_pd

                    if (back_num == 0):
                        snapped_time = to_time(best_index)

                    comb_pow += (onset_vecs[4][best_index])

                    back_num += 1

                add_pow = comb_pow / to_idx(cur_time)
                comb_pows.append(add_pow)
                comb_times.append(snapped_time)

                #Select the peak of this comb vector that represents the beat by finding the largest value in a given range

                #Below are just different ways to select the beat time depending on the time in the song and the tempo instability

                if (first_beat_selected == False):

                    if (cur_time < 4+best_pd*P_BEAT_START_MULT):
                        if (add_pow > beat_max):
                            beat_max = add_pow
                            prev_beat_guess = cur_time
                            tentative_prev_time = prev_beat_guess
                    else:
                        next_beat_time = prev_beat_guess + best_pd
                        if (next_beat_time < 31 and next_beat_time >= 5):
                            found_beats.append(next_beat_time - P_BEAT_SHIFT)
                        first_beat_selected = True
                        beat_max = 0

                else:

                    offset_in_beat = cur_time - prev_beat_guess
                    percent_in_beat = offset_in_beat / best_pd

                    if (percent_in_beat < P_BEAT_THRESH_START):
                        beat_thresh = prev_thresh * 1.3
                    if (percent_in_beat >= P_BEAT_THRESH_START):
                        beat_thresh = prev_thresh * (1.2 - (percent_in_beat) / P_BEAT_THRESH_DECAY)

                    if (tempo_instability >= P_BEAT_THRESH_UNSTABLE):
                        #If instability is high enough, then use a "reactive" approach. Just register beats immediately after a volume peak.
                        if ((comb_pows[-2] >= comb_pows[-1]) and (comb_pows[-2] >= comb_pows[-3]) and (
                            comb_pows[-2] > beat_thresh)):
                            prev_thresh = comb_pows[-2]
                            prev_beat_guess = comb_times[-2]
                            if (comb_times[-2] - found_beats[-1] > best_pd * .3):
                                found_beats.append(comb_times[-2] - P_BEAT_SHIFT)
                            percent_in_beat = 0
                    else:
                        #Otherwise, select the highest value from the recent past and extrapolate the beat to that value plus the estimated period.
                        if ((tempo_instability <= P_SHORT_CORR_THRESH and percent_in_beat >= P_BEAT_START_PERCENT_STABLE) or (
                                tempo_instability > P_SHORT_CORR_THRESH and percent_in_beat >= P_BEAT_START_PERCENT_UNSTABLE)):
                            if (comb_pows[-2] > beat_max):
                                beat_max = comb_pows[-2]
                                tentative_prev_time = comb_times[-2]

                        if ((tempo_instability <= P_SHORT_CORR_THRESH and percent_in_beat > P_BEAT_END_PERCENT_STABLE) or (
                                tempo_instability > P_SHORT_CORR_THRESH and percent_in_beat > P_BEAT_END_PERCENT_UNSTABLE)):
                            prev_beat_guess = tentative_prev_time
                            next_beat_time = prev_beat_guess + best_pd
                            if (next_beat_time < 31 and next_beat_time >= 5):
                                found_beats.append(next_beat_time - P_BEAT_SHIFT)
                            beat_max = 0



        cur_sample = wav.tell()
        cur_window +=1

        #Shift over 256 samples for the next window
        sample_arr = sample_arr[256:]
        sample_arr = np.append(sample_arr,np.fromstring(wav.readframes(512), dtype='int16')[::2])

    return found_beats, period_data


def beat_detect_simulate_realtime(wav):

    t0 = time.time() #benchmark
    wav.rewind()

    (found_beats, period_data) = main_thread(wav)

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
        #print("Beats detected:", found_beats)
        #print("Score:",main)

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

        #print("incorrect periods: ", incorrect_count, "/", len(found_pds))
        #print(tempo_instability)

        total_incorrect += incorrect_count


def test_song(song_name):
    global name_of_song, incorrect_count, total_incorrect
    DEBUG_SET = 1

    name_of_song = song_name

    wav = wave.open(PATH + song_name + ".wav")

    #print("Successfully read the wav file: " + song_name)

    #Run the algorithm
    found_beats, period_data = beat_detect_simulate_realtime(wav)


    process_results(wav, song_name, found_beats, period_data)



def testbed():
    global total_incorrect, total_score, num_scores

    total_incorrect = 0
    total_score = 0
    num_scores = 0

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




def session():
    global total_incorrect
    total_incorrect = 0

    testbed()
    average_score = total_score/num_scores
    #print("Average score:",average_score)
    print("")
    return average_score





def set_param(num, val):
    global ALL_PARAMETERS
    global P_PEAK_CONSENSUS_TOLERANCE, P_DISTINCT_PEAK_TOLERANCE, P_TEMPO_START, P_MED_WEIGHT, P_SHORT_WEIGHT, \
        P_COMBINE_TEMPOS_THRESH, P_MAX_TIME_MED_CORR, P_LONG_TEMPO_CHANGE_PENALTY, P_SHORT_CORR_THRESH, P_TEMPO_CONSENSUS_TOLERANCE, P_TEMPO_WEIGHT_START, \
        P_FREQ_BAND_1, P_FREQ_BAND_2, P_FREQ_BAND_3, P_SNAP_THRESH_1, P_SNAP_THRESH_2, P_SNAP_THRESH_3, P_BEAT_START_MULT, P_BEAT_SHIFT, P_BEAT_THRESH_START, \
        P_BEAT_THRESH_DECAY, P_BEAT_THRESH_UNSTABLE, P_BEAT_START_PERCENT_STABLE, P_BEAT_START_PERCENT_UNSTABLE, P_BEAT_END_PERCENT_STABLE, P_BEAT_END_PERCENT_UNSTABLE

    if (num == 0): P_PEAK_CONSENSUS_TOLERANCE = val
    if (num == 1): P_DISTINCT_PEAK_TOLERANCE = val
    if (num == 2): P_TEMPO_START = val
    if (num == 3): P_MED_WEIGHT = val
    if (num == 4): P_SHORT_WEIGHT = val
    if (num == 5): P_COMBINE_TEMPOS_THRESH = val
    if (num == 6): P_MAX_TIME_MED_CORR = val
    if (num == 7): P_LONG_TEMPO_CHANGE_PENALTY = val
    if (num == 8): P_SHORT_CORR_THRESH = val
    if (num == 9): P_TEMPO_CONSENSUS_TOLERANCE = val
    if (num == 10): P_TEMPO_WEIGHT_START = val
    if (num == 11): P_FREQ_BAND_1 = val
    if (num == 12): P_FREQ_BAND_2 = val
    if (num == 13): P_FREQ_BAND_3= val
    if (num == 14): P_SNAP_THRESH_1 = val
    if (num == 15): P_SNAP_THRESH_2 = val
    if (num == 16): P_SNAP_THRESH_3 = val
    if (num == 17): P_BEAT_START_MULT = val
    if (num == 18): P_BEAT_SHIFT = val
    if (num == 19): P_BEAT_THRESH_START = val
    if (num == 20): P_BEAT_THRESH_DECAY = val
    if (num == 21): P_BEAT_THRESH_UNSTABLE = round(P_BEAT_THRESH_UNSTABLE * mult)
    if (num == 22): P_BEAT_START_PERCENT_STABLE = val
    if (num == 23): P_BEAT_START_PERCENT_UNSTABLE = val
    if (num == 24): P_BEAT_END_PERCENT_STABLE = val
    if (num == 25): P_BEAT_END_PERCENT_UNSTABLE = val

    ALL_PARAMETERS = [P_PEAK_CONSENSUS_TOLERANCE, P_DISTINCT_PEAK_TOLERANCE, P_TEMPO_START, P_MED_WEIGHT,
                      P_SHORT_WEIGHT,
                      P_COMBINE_TEMPOS_THRESH, P_MAX_TIME_MED_CORR, P_LONG_TEMPO_CHANGE_PENALTY, P_SHORT_CORR_THRESH,
                      P_TEMPO_CONSENSUS_TOLERANCE,
                      P_TEMPO_WEIGHT_START, P_FREQ_BAND_1, P_FREQ_BAND_2, P_FREQ_BAND_3, P_SNAP_THRESH_1,
                      P_SNAP_THRESH_2, P_SNAP_THRESH_3, P_BEAT_START_MULT, P_BEAT_SHIFT, P_BEAT_THRESH_START,
                      P_BEAT_THRESH_DECAY, P_BEAT_THRESH_UNSTABLE, P_BEAT_START_PERCENT_STABLE,
                      P_BEAT_START_PERCENT_UNSTABLE, P_BEAT_END_PERCENT_STABLE,
                      P_BEAT_END_PERCENT_UNSTABLE]

def change_param(num, mult):
    global ALL_PARAMETERS
    global P_PEAK_CONSENSUS_TOLERANCE, P_DISTINCT_PEAK_TOLERANCE, P_TEMPO_START, P_MED_WEIGHT, P_SHORT_WEIGHT, \
        P_COMBINE_TEMPOS_THRESH, P_MAX_TIME_MED_CORR, P_LONG_TEMPO_CHANGE_PENALTY, P_SHORT_CORR_THRESH, P_TEMPO_CONSENSUS_TOLERANCE, P_TEMPO_WEIGHT_START, \
        P_FREQ_BAND_1, P_FREQ_BAND_2, P_FREQ_BAND_3, P_SNAP_THRESH_1, P_SNAP_THRESH_2, P_SNAP_THRESH_3, P_BEAT_START_MULT, P_BEAT_SHIFT, P_BEAT_THRESH_START, \
        P_BEAT_THRESH_DECAY, P_BEAT_THRESH_UNSTABLE, P_BEAT_START_PERCENT_STABLE, P_BEAT_START_PERCENT_UNSTABLE, P_BEAT_END_PERCENT_STABLE, P_BEAT_END_PERCENT_UNSTABLE

    if (num == 0): P_PEAK_CONSENSUS_TOLERANCE *= mult
    if (num == 1): P_DISTINCT_PEAK_TOLERANCE *= mult
    if (num == 2): P_TEMPO_START *= mult
    if (num == 3): P_MED_WEIGHT *= mult
    if (num == 4): P_SHORT_WEIGHT *= mult
    if (num == 5): P_COMBINE_TEMPOS_THRESH *= mult
    if (num == 6): P_MAX_TIME_MED_CORR *= mult
    if (num == 7): P_LONG_TEMPO_CHANGE_PENALTY *= mult
    if (num == 8): P_SHORT_CORR_THRESH *= mult
    if (num == 9): P_TEMPO_CONSENSUS_TOLERANCE *= mult
    if (num == 10): P_TEMPO_WEIGHT_START *= mult
    if (num == 11): P_FREQ_BAND_1 = round(P_FREQ_BAND_1 * mult)
    if (num == 12): P_FREQ_BAND_2 = round(P_FREQ_BAND_2 * mult)
    if (num == 13): P_FREQ_BAND_3 = round(P_FREQ_BAND_3 * mult)
    if (num == 14): P_SNAP_THRESH_1 = round(P_SNAP_THRESH_1 * mult)
    if (num == 15): P_SNAP_THRESH_2 = round(P_SNAP_THRESH_2 * mult)
    if (num == 16): P_SNAP_THRESH_3 = round(P_SNAP_THRESH_3 * mult)
    if (num == 17): P_BEAT_START_MULT *= mult
    if (num == 18): P_BEAT_SHIFT *= mult
    if (num == 19): P_BEAT_THRESH_START *= mult
    if (num == 20): P_BEAT_THRESH_DECAY *= mult
    if (num == 21): P_BEAT_THRESH_UNSTABLE = round(P_BEAT_THRESH_UNSTABLE * mult)
    if (num == 22): P_BEAT_START_PERCENT_STABLE *= mult
    if (num == 23): P_BEAT_START_PERCENT_UNSTABLE *= mult
    if (num == 24): P_BEAT_END_PERCENT_STABLE *= mult
    if (num == 25): P_BEAT_END_PERCENT_UNSTABLE *= mult

    ALL_PARAMETERS = [P_PEAK_CONSENSUS_TOLERANCE, P_DISTINCT_PEAK_TOLERANCE, P_TEMPO_START, P_MED_WEIGHT,
                      P_SHORT_WEIGHT,
                      P_COMBINE_TEMPOS_THRESH, P_MAX_TIME_MED_CORR, P_LONG_TEMPO_CHANGE_PENALTY, P_SHORT_CORR_THRESH,
                      P_TEMPO_CONSENSUS_TOLERANCE,
                      P_TEMPO_WEIGHT_START, P_FREQ_BAND_1, P_FREQ_BAND_2, P_FREQ_BAND_3, P_SNAP_THRESH_1,
                      P_SNAP_THRESH_2, P_SNAP_THRESH_3, P_BEAT_START_MULT, P_BEAT_SHIFT, P_BEAT_THRESH_START,
                      P_BEAT_THRESH_DECAY, P_BEAT_THRESH_UNSTABLE, P_BEAT_START_PERCENT_STABLE,
                      P_BEAT_START_PERCENT_UNSTABLE, P_BEAT_END_PERCENT_STABLE,
                      P_BEAT_END_PERCENT_UNSTABLE]


def ml():
    global ALL_PARAMETERS

    global P_PEAK_CONSENSUS_TOLERANCE, P_DISTINCT_PEAK_TOLERANCE, P_TEMPO_START, P_MED_WEIGHT, P_SHORT_WEIGHT, \
        P_COMBINE_TEMPOS_THRESH, P_MAX_TIME_MED_CORR, P_LONG_TEMPO_CHANGE_PENALTY, P_SHORT_CORR_THRESH, P_TEMPO_CONSENSUS_TOLERANCE, P_TEMPO_WEIGHT_START, \
        P_FREQ_BAND_1, P_FREQ_BAND_2, P_FREQ_BAND_3, P_SNAP_THRESH_1, P_SNAP_THRESH_2, P_SNAP_THRESH_3, P_BEAT_START_MULT, P_BEAT_SHIFT, P_BEAT_THRESH_START, \
        P_BEAT_THRESH_DECAY, P_BEAT_THRESH_UNSTABLE, P_BEAT_START_PERCENT_STABLE, P_BEAT_START_PERCENT_UNSTABLE, P_BEAT_END_PERCENT_STABLE, P_BEAT_END_PERCENT_UNSTABLE


    cur_param = 0
    cur_max = 0

    print("Running first session")
    cur_max = session()

    while (True):

        #Just add and subtract to find a local max for each parameter

        baseline =  ALL_PARAMETERS[cur_param]

        inc_best = 0
        inc_val = ALL_PARAMETERS[cur_param]
        dec_best = 0
        dec_val = ALL_PARAMETERS[cur_param]

        increasing = True
        while (increasing):
            #How much to multiply it by
            change_param(cur_param,1.2)

            print("Running session with param",cur_param,"increased to val",ALL_PARAMETERS[cur_param], "All params:", ALL_PARAMETERS)
            score = session()
            if (score > cur_max):
                print("Increasing", cur_param, "gave new best", score)
                cur_max = score
                inc_best = score
                inc_val = ALL_PARAMETERS[cur_param]
            else:
                increasing = False

        set_param(cur_param,baseline)

        decreasing = True
        while (decreasing):
            change_param(cur_param, 1.0/1.2)
            print("Running session with param", cur_param, "decreased to val", ALL_PARAMETERS[cur_param]," All params:", ALL_PARAMETERS)
            score = session()
            if (score > cur_max):
                print("Decreasing",cur_param,"gave new best", score)
                cur_max = score
                dec_best = score
                dec_val = ALL_PARAMETERS[cur_param]
            else:
                decreasing = False


        if (inc_best == 0 and dec_best == 0):
            set_param(cur_param, baseline)

        if (inc_best > dec_best):
            set_param(cur_param, inc_val)
        else:
            set_param(cur_param, dec_val)

        cur_param += 1
        if (cur_param > len(ALL_PARAMETERS)):
            cur_param = 0

        print("Finished examining param",cur_param)




def main(argv):
    ml()

if __name__ == "__main__":
    main(sys.argv[1:])

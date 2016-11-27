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


time_step = 512.0/44100

debugNum = 0


def find_consensus(data, confidence, max_group_size):
    global time_step
    bins = []
    # bins[[val,z],[val,z]...]
    for idx, d in enumerate(data):
        found_bin = False
        for b in bins:
            #3 checks for double/half consideration
            if abs(d - b[0]) < max_group_size:
                b[0] = (b[0] * b[1] + d * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                found_bin = True
                break
            elif abs(d*2 - b[0]) < max_group_size:
                b[0] = (b[0] * b[1] + d*2 * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                found_bin = True
                break
            elif abs(d/2 - b[0]) < max_group_size:
                b[0] = (b[0] * b[1] + d/2 * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                found_bin = True
                break
        if (found_bin == False):
            bins.append([d, confidence[idx]])

    return sorted(bins, key=lambda x: int(x[1]), reverse=True)


def get_quarter_note(period):
    while (period > 0.750):
        period /= 2
    while (period < 0.375):
        period *= 2
    return period

def match_periods(pd1,pd2,window):
    return abs(get_quarter_note(pd1)-get_quarter_note(pd2)) < window


def index_to_time(idx):
    global time_step
    return (16+idx) * time_step

def time_to_index(time):
    global time_step
    return int(round(time/time_step) - 16)


def correlate_onsets(a, b):
    global time_step

    period_guesses_to_return = []

    # Performs an O(n*logn) correlation
    corr = scipy.signal.fftconvolve(a, b[::-1], mode='full')
    mid = len(corr) // 2

    # The period search space is approx 187.5 ms to 3000ms (spans eigth notes at 160bpm to measures at 80bpm)
    # Divide by 1000000 to get numbers that are easier to read
    corr_search = corr[mid + 16:mid + 256]/1000000
    measure_search = corr[mid + 128:mid + 256]/1000000
    corr_search_x = np.linspace(16*time_step,256*time_step,240) #Maps indices to time

    corr_baseline_noise = min(corr_search)
    measure_baseline_noise = min(measure_search)
    corr_avg = np.average(corr_search)
    measure_avg = np.average(measure_search)

    overall_best_idx = (np.argsort(corr_search, axis=0)[::-1])[0]
    measure_best_idx = (np.argsort(measure_search, axis=0)[::-1])[0]

    best_overall = overall_best_idx * time_step + 16 * time_step
    best_measure =  measure_best_idx* time_step + 128 * time_step

    best_overall_strength = corr_search[overall_best_idx]-corr_baseline_noise
    best_measure_strength = (measure_search[measure_best_idx]-measure_baseline_noise)

    best_overall_period = get_quarter_note(best_overall)
    best_measure_period = get_quarter_note(best_measure)

    #Now that we are within the quarter note range, check to see that we have a reasonably high value, if not, we
    #may need to go to the second option (qn = quarter note)

    best_overall_strength = corr_search[time_to_index(best_overall_period)]
    best_measure_strength = corr_search[time_to_index(best_measure_period)]

    #We took the overall best period within the eigth note to measure range and mapped it to the quarter note range
    #If there's no peak for the quarter note here (correlation is low), we will add in the second best peak

    corr_max = max(corr_search)

    overall_percent_above = (best_overall_strength-corr_avg)/(corr_max-corr_avg)
    measure_percent_above = (best_measure_strength-corr_avg)/(corr_max-corr_avg)

    #print("overall percent above:",overall_percent_above)
    #print("measure percent above:",measure_percent_above)

    replaced = False
    if (overall_percent_above < .3):
        #print("Bringing in second best peak")
        next_best_indices = (np.argsort(corr_search, axis=0)[::-1])[1:20]
        for i in next_best_indices:
            consider_time = corr_search_x[i]
            consider_time = get_quarter_note(consider_time)
            if (abs(consider_time-best_overall_period) > 0.03):
                period_guesses_to_return.append([consider_time,corr_search[i]*2])
                break

    period_guesses_to_return.append([best_overall_period,best_overall_strength])
    period_guesses_to_return.append([best_measure_period,best_measure_strength])

    return period_guesses_to_return

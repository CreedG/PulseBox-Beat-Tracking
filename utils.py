import struct
import wave, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import sleep
import scipy.fftpack
import scipy.signal
import time
from ring import Ring

import math

time_step = 512.0/44100

debugNum = 0

def find_consensus(data, confidence, max_group_size):
    time_step
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
    return (16+idx) * time_step

def time_to_index(time):
    return int(round(time/time_step) - 16)


def correlate_onsets(a, b):
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

    #debug("overall percent above:",overall_percent_above)
    #debug("measure percent above:",measure_percent_above)

    replaced = False
    if (overall_percent_above < .3):
        #debug("Bringing in second best peak")
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

class KalmanFilter:
    def __init__(self):
        self.Q = 5 * 0.1**5 # process variance, pls tweak
        self.R = 1 * 0.1**3 # measurement variance, pls tweak
        self.P=1
        self.xhat=0
        self.K=0

    def iter(self, z):
        self.K = self.P/( self.P+self.R )
        self.xhat+=self.K*(z-self.xhat)
        self.P = (1-self.K)*self.P + self.Q
        return self.xhat

#Initializes my ring class with the known period currently
def init_phase_globals(period, known_beats):
    global r, prev_len
    prev_len = [0]*7
    r = Ring(period, known_beats)

#Returns the next beat by interfacing with the ring class
def detect_phase(period, times, strengths, time):
    global r, prev_len

    r.iter(period, time)

    for b in range(0,len(times)):
        for i in range(prev_len[b], len(times[b])):
            r.insert(times[b][i], strengths[b][i], b)
        prev_len[b] = len(times[b])

    r.generate_next_beat(time)
    return r.get_beats()

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


def find_consensus(data, confidence, max_group_size):
    global time_step
    bins = []
    # bins[[val,z],[val,z]...]
    for idx, d in enumerate(data):
        found_bin = False
        for b in bins:
            if abs(d - b[0]) < max_group_size:
                b[0] = (b[0] * b[1] + d * confidence[idx]) / (b[1] + confidence[idx])
                b[1] += confidence[idx]
                found_bin = True
                break
        if (found_bin == False):
            bins.append([d, confidence[idx]])

    return sorted(bins, key=lambda x: int(x[1]), reverse=True)

def correlate_onsets(a, b):
    global time_step

    # Performs an O(n*logn) correlation
    corr = scipy.signal.fftconvolve(a, b[::-1], mode='full')
    mid = len(corr) / 2
    mid_to_end = math.ceil(mid)

    # The search space is approx 187.5 ms to 3000ms (spans eigth notes at 160bpm to measures at 80bom)
    corr_search = corr[mid + 16:mid + 256]
    corr_search_x = np.linspace(16*time_step,256*time_step,240)

    # Grab the best peak so we can give it more influence in the weighting
    best_peak_time = time_step * np.argmax(corr_search) + 16 * time_step


    # Grab the top 8 peaks
    K = 8
    if (len(corr_search) < 8):
        K = len(corr_search)
    peak_idxs = np.argpartition(corr_search, -K)[-K:]
    sorted_peak_idxs = np.sort(peak_idxs)


    # Two peaks right next to each other are probably not distinct autocorrelation results, they are just noise
    # from the same "actual peak". Therefore, average peaks that are really close (within 2 indexes = 22ms(
    sorted_combined_peak_idxs = []

    trav = 0
    while (trav < len(sorted_peak_idxs)):
        group_av = sorted_peak_idxs[trav]
        group_cnt = 1.0
        for trav2 in range(trav, trav + 5):
            if (trav2 < 0): continue
            if (trav == trav2): continue
            if (trav2 >= len(sorted_peak_idxs)): break
            # print(trav-trav2)
            if (abs(sorted_peak_idxs[trav] - sorted_peak_idxs[trav2]) <= 2):
                group_av += sorted_peak_idxs[trav2]
                group_cnt += 1
                trav = trav2
        group_av /= group_cnt
        sorted_combined_peak_idxs.append(group_av)
        trav += 1



    # After potential combining of peaks we can expect to have somewhere around 4-8 peaks

    # These respresnt potential periods (and therefore tempos)

    # Multiply and divide them to find out which quarter note they each correspond to (375-750ms)
    # If smaller, multiply by 2, if larger divide by 2 to get them in this range
    quarter_note_periods = []

    for p in sorted_combined_peak_idxs:
        peak_time = p * time_step + 16 * time_step
        while (peak_time > 0.750): peak_time /= 2
        while (peak_time < 0.375): peak_time *= 2
        quarter_note_periods.append(peak_time)

    # Give the best peak time it's own entry (again) to reinforce it
    while (best_peak_time > 0.750): best_peak_time /= 2
    while (best_peak_time < 0.375): best_peak_time *= 2
    quarter_note_periods.append(best_peak_time)
    quarter_note_periods.append(best_peak_time)


    # Consider them all equally likely in the consensus process
    quarter_note_z = np.ones(len(quarter_note_periods))

    #print("qnote periods",quarter_note_periods)
    return find_consensus(quarter_note_periods, quarter_note_z, 0.005)



# IGNORE THIS, not currently working
def corr_peaks(a, b):
    x1vec = a[0]
    x2vec = b[0]
    y1vec = a[1]
    y2vec = b[1]

    print("aaa")
    print(len(x1vec))

    tolerance = 0.03

    shifts = []

    # STAGE 1, fill the shifts array to get the possible tempos

    for ix1, x1 in enumerate(x1vec):
        for ix2, x2 in enumerate(x2vec):

            print("comparing", x1, x2)

            dist = abs(x1 - x2)
            if (dist < tolerance):
                break

            mult = y1vec[ix1] * y2vec[ix2]

            matched = False
            # add the shift component which may be slightly off from another OR a multiple, correct for these
            # target range is 80-160bpm which is 375 to 750ms so squash all multiples to fit
            for s in shifts:
                if (abs(dist - s[0]) < tolerance):
                    s[0] = (s[0] * s[1] + dist * mult) / (s[1] + mult)  # Shift it over
                    s[1] += mult  # Add the new component
                    matched = True
                    print("matched the bin", s[0], ", new strength is", s[1])
                    break

            if (matched == False):
                print("creating a new bin for", dist)
                shifts.append([dist, mult])

            print("shifts is now", shifts)
            print()

    # Find period from the maximal correlation shift
    max = [0, 0]
    for s in shifts:
        if s[1] > max[1]:
            max[1] = s[1]
            max[0] = s[0]

    T = max[0]

    # Two close contenders lowers the z (confidence) score

    # STAGE 2 get the actual beat times using the period... assuming it is correct.

    # Find phase from combing through the data at the period interval. Do this for each input to the correlation (or once if auto)

    # Use the n (3 for now) most recent peaks to test

    xlen = len(x1vec)

    best_phase_pow = 0
    best_phase = 0

    print(xlen)

    for start_idx in reversed(range(xlen - 10, xlen)):
        # print("starting from",start_idx,"which is",x1vec[start_idx])
        total_pow = 0
        comb_idx = start_idx
        comb_sec = x1vec[start_idx]
        while (comb_idx >= 0 and comb_sec >= 0):
            # print("comb_sec =",comb_sec)
            comb_sec -= T
            while (x1vec[comb_idx] > comb_sec - .05 and comb_idx >= 0):
                if (abs(comb_sec - x1vec[comb_idx]) < .05):
                    # print("found match at comb_sec,x1vec[comb_idx]",comb_sec,x1vec[comb_idx])
                    total_pow += y1vec[comb_idx]
                    comb_sec = x1vec[
                        comb_idx]  # Snap the value to get rid of accumulated period error (over a long time, may deviate)
                comb_idx -= 1
        # print("total pow for the phase is", total_pow)
        if (total_pow > best_phase_pow):
            best_phase_pow = total_pow
            best_phase = x1vec[start_idx]

    P = best_phase

    return T, P, 0


if __name__ == "__main__":
    a = [3, 3, 5, 5.1, 6, 6.1]
    b = [1, 2, 1, 2, 3, 2]

    t, p, z = find_consensus(a, b, 0.3)
    print(a)

    print(t, p, z)

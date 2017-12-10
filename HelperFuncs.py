import scipy.fftpack
import scipy.signal
import math
import numpy as np

import BeatFind_Parameters as P


def get_quarter_note(period):
    while (period > 0.750):
        period /= 2
    while (period < 0.375):
        period *= 2
    return period


def index_to_time(idx, time_step):

    return idx * time_step

def time_to_index(time, time_step):

    return int(round(time/time_step))


def period_is_multiple(pd1,pd2,window):
    return abs(get_quarter_note(pd1)-get_quarter_note(pd2)) < window



def are_periods_related(p1,p2):
    return ((abs(p1 / p2 - 1) < P.P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - .5) < P.P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - 2) < P.P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - 1.333) < P.P_DISTINCT_PEAK_TOLERANCE) or
            (abs(p1 / p2 - 0.75) < P.P_DISTINCT_PEAK_TOLERANCE))


#Perform an autocorrelation for a given frequency range and pick out the peaks using the weightings from the P_PEAK_PICK_ARRAY parameters
def correlate_onsets(a, b, time_step, peak_pick_method, peak_pick_index, corr_range):

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
        search_low_idx = mid + time_to_index(time_low, time_step)
        search_high_idx = mid + time_to_index(time_high, time_step)

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
            consider_pd = index_to_time(i+search_low_idx-mid, time_step)

            pd_is_unique = True
            for other_pd_in_range in check_for_duplicates:
                if (period_is_multiple(consider_pd,other_pd_in_range[0], P.P_DISTINCT_PEAK_TOLERANCE)):
                    pd_is_unique = False
            if (pd_is_unique):

                #Ge the quarter note time that this peak corresponds to, may require snapping to a local max
                consider_pd_qn_time = get_quarter_note(consider_pd)

                do_snap = (consider_pd < 0.375)
                snap_index = time_to_index(consider_pd_qn_time, time_step) - 16
                if (do_snap):
                    max_snap = 0
                    max_index = snap_index
                    for trav in range(-5, 5):
                        if (corr_space[snap_index + trav] > max_snap):
                            max_snap = corr_space[snap_index + trav]
                            max_index = snap_index + trav

                    consider_pd_qn_time = get_quarter_note(index_to_time(max_index + 16, time_step))


                #The strength is the power multiplied by the weighting from the P_PEAK_PICK_ARRAY parameter
                found_pd_strength = corr_space[time_to_index(consider_pd, time_step)-16] * PARAM[3][num_pds_found]

                periods_found_in_range.append([consider_pd_qn_time, found_pd_strength])
                all_periods_found.append([consider_pd_qn_time, found_pd_strength])

                num_pds_found+=1
                if (num_pds_found == num_pds_to_find):
                    break

    #Return all of the periods for this frequency range and their voting power
    return all_periods_found


#Given multiple values, place values near each other into bins
def find_consensus(data, confidence, size_within_group):

    #bins are [value,confidence,delta,num ele]
    bins = []
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

    #Result is a sorted list of bins containing the value and score. sorted[0][0] gives the bin of highest score
    return sorted(bins, key=lambda x: float(x[1]), reverse=True)

def feature1(pts):
    data = pts[-500:]
    ret =  np.log(np.max(data)/np.mean(data)) - 2
    return ret
def feature2(pts):
    data = pts[-500:]
    ret = np.mean(data) / np.mean(np.sort(data)[:-3])
    return ret

def feature3(data):
    points = data[-500:]
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.mean(diff)

    modified_z_score = P.Z_MULT * diff / med_abs_deviation
    for z in modified_z_score:
        if z > P.MAD_THRESH:
            return 1
    return 0

def feature4(pts):
    data = pts[-500:]
    diff = (100 - P.OUTLIER_THRESH) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    for d in data:
        if d < minval or d > maxval:
            return 1
    return 0

def weighted(onset_vecs, confidence, data, recalc):

    ret = 0
    for band in range(4):
        if recalc:
            modifier = (feature1(onset_vecs[band]) * P.F[0] + feature2(onset_vecs[band]) * P.F[1] + feature3(onset_vecs[band]) * P.F[2] + feature4(onset_vecs[band]) * P.F[3])
            confidence[band] *= modifier
        #ret += data[band] * (confidence[band]/ sum(confidence)) * P.ONSET_WEIGHTS[band]
        ret += data[band] * P.ONSET_WEIGHTS[band]
    return confidence, ret

def set_params(params):
    for i in range(4):
        P.ONSET_WEIGHTS[i] = params[i]
    P.MAD_THRESH = params[4]
    P.OUTLIER_THRESH = min(params[5] + 85, 100)
    for i in range(4):
        P.F[i] = params[6 + i]
    P.Z_MULT = params[10]

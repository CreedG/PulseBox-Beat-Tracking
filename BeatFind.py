import wave, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal


import BeatFind_Parameters as P
import HelperFuncs as H


#Each window of data (either captured from the mic or read from the wav file) moves 256 samples at 22050Hz
time_step = 256.0/22050
instabilities = []
PATH = "closed/"

#When SIMULATE is set to true, input is from a wav file and output it a list of beat times
#When SIMULATE is set to false, input is from the microphone and output is a serial write at the beat times to the LED controller
SIMULATE = True

#If SIMULATE is set to true, plots may be optionally turned on to show some data for the song
DEBUG_PLOTS = False


#Tempo finding is called about every 350ms. Its function is to estimate the period of the song and generate a rating for instability (how much tempo changes)
#This is achieved by splitting the onset vectors into different frequency bands and correlating them for different amounts of time (most recent 17, 6, and 3 seconds)
#Then this data is recombined with appropriate weightings as determined by the parameters in BeatFind_Parameters
#This granular approach (though complex) is more effective than just doing a single autocorrelation and allows us to extract more information

def tempo_processing_thread(onset_vecs, cur_time):
    global period_data, tempo_derivative, tempo_instability, tag_use_short_correlation, tag_use_med_correlation

    candidate_periods_l, candidate_periods_m, candidate_periods_s = [], [], []

    #Get the tempo votes from the short, medium, and long term correlations
    for freq_band in range(0, 5):

        long_periods = H.correlate_onsets(onset_vecs[freq_band][-1500:], onset_vecs[freq_band][-1500:], time_step, P.P_PEAK_PICK_ARRAY_LONG, freq_band, [16, 345])
        med_periods = H.correlate_onsets(onset_vecs[freq_band][-500:], onset_vecs[freq_band][-500:], time_step, P.P_PEAK_PICK_ARRAY_MED, freq_band, [16, 345])
        short_periods = H.correlate_onsets(onset_vecs[freq_band][-250:], onset_vecs[freq_band][-250:], time_step, P.P_PEAK_PICK_ARRAY_SHORT, freq_band, [16,300])

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
            if s < 0:
                s = 0
            vote = s
            voting_power.append(vote)
        consensus_period = H.find_consensus(times, voting_power, P.P_PEAK_CONSENSUS_TOLERANCE)
        best_period_votes.append(consensus_period[0][0])

    best_period_l = best_period_votes[0]
    best_period_m = best_period_votes[1]
    best_period_s = best_period_votes[2]

    #Now find the best tempo using these short, medium, and long term period guesses by smoothing and combining them
    weights = np.linspace(1, 1, 3)
    groups =  H.find_consensus([best_period_l,best_period_m,best_period_s], weights, .05)
    best_period_combined =  groups[0][0]

    if len(period_data[3]) > 0:
        if cur_time < P.P_TEMPO_START:
            best_period_combined = best_period_l
        else:
            #Combine the short, medium, and long term guesses if they agree on a certain period value
            weights = [1,P.P_MED_WEIGHT,P.P_SHORT_WEIGHT]
            groups = H.find_consensus([best_period_l, best_period_m, best_period_s], weights, P.P_COMBINE_TEMPOS_THRESH)
            best_period_combined = groups[0][0]

            long_groups = H.find_consensus(period_data[3][-10:], np.linspace(1, 1, 10), .015)
            short_and_med_groups = H.find_consensus(period_data[1][-10:]+period_data[2][-10:], np.linspace(1, 1, 24), .015)

            if cur_time < P.P_MAX_TIME_MED_CORR and len(short_and_med_groups) == 1 and len(long_groups) >= 2:
                if (H.are_periods_related(long_groups[0][0],(long_groups[1][0])) == False and
                        len(H.find_consensus([long_groups[1][0], short_and_med_groups[0][0]], np.linspace(1, 1, 2), .015)) == 2):
                    #If the medium and short term correlations agree but not the long term correlation, maybe the song
                    # is changing too much so use medium correlation as period
                    tag_use_med_correlation = True

        #Look for the tempo gradually moving in the short term correlation, if this happens add to tempo_instability
        tempo_derivative.append((best_period_s-(period_data[1][-1])))
        dpos = len(tempo_derivative)-1
        stretch_len = 1
        total_change = tempo_derivative[dpos]
        while cur_time <= 15 and abs(tempo_derivative[dpos]) < .1 and abs(total_change/stretch_len) > .01:
            stretch_len += 1
            if dpos >= 1:
                dpos -= 1
            else:
                break
            if stretch_len > 3:
                if abs(tempo_derivative[dpos]+tempo_derivative[dpos+1]+tempo_derivative[dpos+2]+tempo_derivative[dpos+3]) < .02:
                    break
            total_change += tempo_derivative[dpos]
        if stretch_len > tempo_instability:
            tempo_instability = stretch_len

    #If the long term tempo is uncertain near the beginning/middle of a song, add to tempo_instability
    if P.P_PENALTY_RANGE_LOW < cur_time < P.P_PENALTY_RANGE_HIGH:
        if not H.are_periods_related(best_period_l, period_data[3][-1]):
            tempo_instability += P.P_LONG_TEMPO_CHANGE_PENALTY

    #If instability is high enough, we want to use the short term correlation to track changes better
    tag_use_short_correlation = (tempo_instability > P.P_SHORT_CORR_THRESH)

    #Store all the vectors (not too much data since it's just one sample every 350ms)
    period_data[0].append(cur_time)
    period_data[1].append(best_period_s)
    period_data[2].append(best_period_m)
    period_data[3].append(best_period_l)
    period_data[4].append(best_period_combined)

    #A few different strategies for smoothing and combining to get the tempo, depending on how unstable the tempo was
    if not tag_use_short_correlation:
        if not tag_use_med_correlation:
            if cur_time < P.P_TEMPO_START:
                ultimate_period = period_data[4][-1]
            else:
                #Normal operation, find the most frequent tempo of the combined short, med, and long term correlation results
                weights = np.linspace(P.P_TEMPO_WEIGHT_START, 1, len(period_data[4]))
                groups = H.find_consensus(period_data[4], weights, P.P_TEMPO_CONSENSUS_TOLERANCE)
                ultimate_period = groups[0][0]
        else:
            #Tempo seems to have been changing somewhat, so find the most frequent tempo of the medium term correlation results
            weights = np.linspace(P.P_TEMPO_WEIGHT_START, 1, len(period_data[2]))
            groups = H.find_consensus(period_data[2], weights, P.P_TEMPO_CONSENSUS_TOLERANCE)
            ultimate_period = groups[0][0]
    else:
        #Tempo seems to have been changing a lot, so only look at the short term correlation results
        weights = np.linspace(1, 1, 4)
        groups = H.find_consensus(period_data[1][-4:], weights, P.P_TEMPO_CONSENSUS_TOLERANCE)
        ultimate_period = groups[0][0]

    #Prevent jumping around between double and half
    if len(period_data[5]) > 0:
        if period_data[5][-1] < 0.4 and abs(ultimate_period / period_data[5][-1] - 2) < .1:
            ultimate_period /= 2
        if period_data[5][-1] > 0.7 and abs(period_data[5][-1] / ultimate_period - 2) < .1:
            ultimate_period *= 2

    #The final result of the period processing
    period_data[5].append(ultimate_period)


def time_to_window_num(time):
    return int(time/time_step)-3

def window_num_to_time(idx):
    return (idx+3)*time_step


# Beat finding uses a comb filter approach. Add up powers and backtrack (at intervals of the estimated period) to the beginning of the song.
# Peaks in total power at a given offset indicate the beat positions.
# As you backtrack, snap to the nearest peak within a range of x samples (each sample is 11ms) to deal with tempo error and variations.
# The amount to snap is dictated by how unstable the tempo was.

def place_beats(onset_vecs, cur_time, cur_window):
    global prev_beat_guess, tentative_prev_time, prev_thresh, beat_thresh, first_beat_selected, started_placing_beats, beat_max, comb_pows, comb_times, music_playing, found_beats, instabilities

    # This is the maximum distance to snap to the nearest peak when backtracking. Set it higher if the tempo is unstable
    if tempo_instability >= P.P_SNAP_THRESH_1:
        snap_range = 7
    elif tempo_instability >= P.P_SNAP_THRESH_2:
        snap_range = 6
    elif tempo_instability >= P.P_SNAP_THRESH_3:
        snap_range = 5
    else:
        snap_range = 4
    #snap_range = 6
    instabilities.append(tempo_instability)

    if started_placing_beats:
        # Grab the period from the period estimator
        best_pd = period_data[5][-1]
        best_pd = .4
        examine_time = cur_time
        comb_pow = 0

        # If snapping distance is farther, you don't have to run it as often. do snap_range-2 for a little overlap
        if (cur_window - 346) % snap_range - 2 == 0:
            # back_num is how many steps we've taken back
            back_num = 0
            # snapped_time is where we are starting this iteration
            snapped_time = 0

            while examine_time > 0:
                index = time_to_window_num(examine_time)
                best_index_val = 0
                best_index = index - snap_range + 1
                check_max = snap_range
                if index >= len(onset_vecs[5]) - snap_range:
                    check_max = 1

                for i in range(-snap_range + 1, check_max):
                    if onset_vecs[5][index + i] > best_index_val:
                        best_index_val = onset_vecs[5][index + i]
                        best_index = index + i

                examine_time = window_num_to_time(best_index) - best_pd
                if back_num == 0:
                    snapped_time = window_num_to_time(best_index)
                comb_pow += (onset_vecs[5][best_index])
                back_num += 1

            add_pow = (comb_pow) / time_to_window_num(cur_time)
            comb_pows.append(add_pow)
            comb_times.append(snapped_time)

            # Select the peak of this comb vector that represents the beat by finding the largest value in a given range

            # Below are just different ways to select the beat time depending on the time in the song and the tempo instability
            if not first_beat_selected:

                # Handle selecting the first beat
                if cur_time < 4 + best_pd * P.P_BEAT_START_MULT:
                    if add_pow > beat_max:
                        beat_max = add_pow
                        prev_beat_guess = cur_time
                        tentative_prev_time = prev_beat_guess
                else:
                    next_beat_time = prev_beat_guess + best_pd
                    if 31 > next_beat_time >= 5 and music_playing == True:
                        found_beats.append(next_beat_time - P.P_BEAT_SHIFT)
                    first_beat_selected = True
                    beat_max = 0

            else:

                offset_in_beat = cur_time - prev_beat_guess
                percent_in_beat = offset_in_beat / best_pd

                if percent_in_beat < P.P_BEAT_THRESH_START:
                    beat_thresh = prev_thresh * 1.3
                if percent_in_beat >= P.P_BEAT_THRESH_START:
                    beat_thresh = prev_thresh * (1.2 - percent_in_beat / P.P_BEAT_THRESH_DECAY)

                if tempo_instability >= P.P_BEAT_THRESH_UNSTABLE:
                    # If instability is high enough, then use a "reactive" approach. Just register beats immediately after a volume peak above the threshold
                    if comb_pows[-1] > beat_thresh:
                        print(1)

                        prev_thresh = comb_pows[-1]
                        prev_beat_guess = comb_times[-1]
                        if comb_times[-1] - found_beats[-1] > best_pd * .3 and music_playing == True:
                            # Add the current time as a beat
                            found_beats.append(cur_time)
                else:
                    # Normal beat finding operation: select the highest value from the recent past and extrapolate the next beat to that value plus the estimated period
                    if ((
                                tempo_instability <= P.P_SHORT_CORR_THRESH and percent_in_beat >= P.P_BEAT_START_PERCENT_STABLE) or (
                                    tempo_instability > P.P_SHORT_CORR_THRESH and percent_in_beat >= P.P_BEAT_START_PERCENT_UNSTABLE)):
                        if comb_pows[-2] > beat_max:
                            print(2)

                            beat_max = comb_pows[-2]
                            tentative_prev_time = comb_times[-2]

                    if ((
                                tempo_instability <= P.P_SHORT_CORR_THRESH and percent_in_beat > P.P_BEAT_END_PERCENT_STABLE) or (
                                    tempo_instability > P.P_SHORT_CORR_THRESH and percent_in_beat > P.P_BEAT_END_PERCENT_UNSTABLE)):
                        prev_beat_guess = tentative_prev_time
                        next_beat_time = prev_beat_guess + best_pd
                        if 31 > next_beat_time >= 5 and music_playing == True:
                            print(4)
                            found_beats.append(next_beat_time - P.P_BEAT_SHIFT)
                        beat_max = 0


# The main thread that gathers the audio data and calls the tempo finding and beat placement code

def main_thread(wav):
    #Share this data with the period finding code
    global period_data, tempo_derivative, tempo_instability, tag_use_short_correlation, tag_use_med_correlation, instabilities, onset_vecs

    instabilities = []
    #Share this data with the beat finding code
    global prev_beat_guess, tentative_prev_time, prev_thresh, beat_thresh, first_beat_selected, started_placing_beats, beat_max, comb_pows, comb_times, music_playing, found_beats

    #Initialize all vars for the song

    found_beats = []
    period_data = [[],[],[],[],[],[],[]]
    recalc = False
    #1. Init audio acquisition vars
    cur_sample, cur_window, start_window, cur_time, total_onset_power = 0,0,0,0,0
    onset_vecs = np.array([[], [], [], [], [], []], dtype=np.int)
    prev_onsets = np.zeros(6, dtype=np.int)
    time_vec = []
    band_confidence = [1] * 4
    #2. Init period finding vars
    tempo_instability = 0
    tempo_derivative = []
    tag_use_short_correlation, tag_use_med_correlation = False,False

    #3. Init beat finding vars
    prev_beat_guess, tentative_prev_time, prev_thresh, beat_thresh, beat_max = 0,0,0,0,0
    started_placing_beats, first_beat_selected, music_playing = False,False,True
    comb_pows, comb_times = [], []

    # 1. PERFORM AUDIO ACQUISITION
    sample_arr = np.fromstring(wav.readframes(2048), dtype='int16')[::2]

    while len(sample_arr) == 1024:

        # 2. CALCULATE TEMPO EVERY ~350ms
        recalc = False
        if cur_time > 4 and ((cur_window - start_window) % 30 == 0 or start_window == 0):
            recalc = True
            tempo_processing_thread(onset_vecs, cur_time)
            started_placing_beats = True
            start_window = cur_window

        cur_time = cur_sample / 44100
        time_vec.append(cur_time)

        windowed = np.hanning(1024) * sample_arr

        #Get the power spectrum
        x_psd, y_psd = scipy.signal.periodogram(windowed, 22050)
        y_psd = np.sqrt(y_psd)

        # Sum up the ranges
        onsets = np.array([0, 0, 0, 0, 0, 0])

        onsets[0] = np.sum(y_psd[0:P.P_FREQ_BAND_1])
        onsets[1] = np.sum(y_psd[P.P_FREQ_BAND_1:P.P_FREQ_BAND_2])
        onsets[2] = np.sum(y_psd[P.P_FREQ_BAND_2:P.P_FREQ_BAND_3])
        onsets[3] = np.sum(y_psd[P.P_FREQ_BAND_3:510])
        onsets[4] = onsets[3] + onsets[2] + onsets[1] + onsets[0]
        band_confidence, onsets[5] = H.weighted(onset_vecs, band_confidence, onsets, recalc)


        # Add up the total power for low and mid frequencies (ignores noise from hf). If this value is sufficiently low, assume there are no instruments (no music) and register no beats
        if cur_time > 1:
            total_onset_power += onsets[1]+onsets[2]
            if cur_window % 100 == 0:
                avg_onset_power = total_onset_power/cur_window
                music_playing = (avg_onset_power > 150)

        # Now we have the power in each range. Compute the derivative (simple difference) and get rid of negative power changes to get onset vectors
        new_onset_samples = onsets - prev_onsets

        # Decrease the big spike at the beginning (the first sample), usually is not a "real" onset
        if len(onset_vecs[4]) == 0:
            for i in range (0,5):
                new_onset_samples[i] /= 3

        # Only take positive increases to get onsets
        new_onset_samples = new_onset_samples.clip(min=0)
        prev_onsets = onsets

        # Add the new onset values to the vectors so that we have them stored as power_onset_vecs[range_num][time_idx]
        new_onset_samples = new_onset_samples.reshape((6, 1))
        onset_vecs = np.hstack([onset_vecs, new_onset_samples])

        # 3. BEAT OFFSET FINDING USING TEMPO
        place_beats(onset_vecs, cur_time, cur_window)

        #Increment position in song (by 256 samples at 22050Hz i.e. 11.6ms)
        cur_sample = wav.tell()
        cur_window +=1
        sample_arr = sample_arr[256:]
        sample_arr = np.append(sample_arr,np.fromstring(wav.readframes(512), dtype='int16')[::2])
    srtd = np.sort(onset_vecs[4])[:-300]
    savg = sum(srtd)/len(srtd)
    avg = sum(onset_vecs[4]) / len(onset_vecs[4])
    #sys.stdout.write("[" + str(sum(instabilities) / len(instabilities)) + ",")
    return found_beats, period_data

def grab_known(song_name):
    known_beats = []
    txtfile_name = PATH + song_name + ".txt"
    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
    known_pds = []
    for i in range(1, len(known_beats)):
        known_pds.append(known_beats[i] - known_beats[i-1])

    return known_beats, known_pds

def run_song(song_name, params=None):
    if params:
        H.set_params(params)
        P.FOURTH = params[-1]
    wav = wave.open(PATH + song_name + ".wav")
    wav.rewind()

    known_beats, known_pds = grab_known(song_name)
    #Run the algorithm
    found_beats, period_data = main_thread(wav)
    plot_results(found_beats, known_beats)
    found_pds = []
    for i in range(1, len(found_beats)):
        found_pds.append(found_beats[i] - found_beats[i-1])
    plt.plot([x * 2 for x in period_data[5]])
    plt.plot(known_pds)
    plt.show()
    plt.plot(comb_times, comb_pows)
    for b in known_beats:
        xpos = b
        plt.plot([xpos, xpos], [min(comb_pows), max(comb_pows)], 'k-', lw=1.5, color='pink')
    plt.show()
    return found_beats, known_beats


def plot_results(found_beats, known_beats):
    display_div = 50
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

    for i in range(5):
        if i == 4:
            plt.plot(scipy.signal.resample(onset_vecs[i + 1]*10000 // max(onset_vecs[5]) + 24000 - i * 10000, 44100//50 * 30), lw=.8)
        else:
            plt.plot(scipy.signal.resample(onset_vecs[i]*10000 // max(onset_vecs[i]) + 24000 - i * 10000, 44100//50 * 30), lw=.8)
    plt.show()

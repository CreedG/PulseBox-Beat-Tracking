import numpy as np
import os
import sys
import BeatFind

def beatEvaluator2(detections,annotations, metrical_options = None):
    """
    Evaluate the accuracy of the algorithm
    :param detections: list of floats
    :param annotations: list of floats
    :return:
    """
    mainscore = 0  # amlT
    backup_scores = []  # amlC, cmlT, cmlC
    # Parameters
    startup_time = 5
    phase_tolerance = 0.175
    tempo_tolerance = 0.175
    if metrical_options is None:
        metrical_options = np.array([[1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 0.5, 0], [2, 0.5, 0]])
    # Run
    (cmlC, cmlT, amlC, amlT) = continuity(detections, annotations, tempo_tolerance, phase_tolerance, startup_time, metrical_options)
    mainscore = amlT
    backup_scores = [amlC, cmlT, cmlC]
    return mainscore, backup_scores


def continuity(detections, annotations, tempo_tolerance, phase_tolerance, startup_time, metrical_options):
    # Convert to numpy array
    detections = np.array(detections)
    annotations = np.array(annotations)

    min_ann_time, min_beat_time = verifyStartUpTime(startup_time, annotations, phase_tolerance)
    valid = np.argwhere(detections > min_beat_time)
    detections = detections[valid]
    detections = detections.reshape(detections.size)
    valid = np.argwhere(annotations > min_ann_time)
    annotations = annotations[valid]
    annotations = annotations.reshape(annotations.size)


    # Check parameters
    if tempo_tolerance < 0 or phase_tolerance < 0:
        raise ValueError('Tempo and phase tolerances must be positive.')
    if len(detections) < 2 or len(annotations) < 2:
        raise ValueError('At least two beats are required past the minimum beat time.')

    """# Interpolate annoations
    double_annotations = np.interp(np.arange(0, len(annotations), 0.5), np.arange(0,len(annotations)), annotations)
    # Make different variations of annotations
    off_beats = double_annotations[1::2]  # Off-beats
    double_tempo = double_annotations  # Double tempo
    half_tempo_e = annotations[0::2]  # Half tempo (even)
    half_tempo_o = annotations[1::2]  # Half tempo (odd)
    all_variations = [annotations, off_beats, double_tempo, half_tempo_e, half_tempo_o]"""

    all_variations = []
    for k in range(metrical_options.shape[0]):
        start_ann = int(metrical_options[k, 0]) - 1  # -1 for MATLAB indexing
        factor = metrical_options[k, 1]
        offbeat = metrical_options[k, 2]
        variation = makeVariations(annotations, start_ann, factor, offbeat)
        all_variations.append(variation)

    cmlCVec = []
    cmlTVec = []
    for variation in all_variations:
        C, T = ContinuityEval(detections, variation, tempo_tolerance, phase_tolerance)
        cmlCVec.append(C)
        cmlTVec.append(T)
    cmlC = cmlCVec[0]
    cmlT = cmlTVec[0]
    amlC = max(cmlCVec)
    amlT = max(cmlTVec)
    return cmlC, cmlT, amlC, amlT


def ContinuityEval(detections,annotations,tempo_tolerance,phase_tolerance):
    """Calculate continuity-based accuracy"""
    correct_phase = np.zeros(max(detections.size, annotations.size))
    correct_tempo = np.zeros(max(detections.size, annotations.size))
    for i in range(detections.size):
        # find the closest annotation and the signed offset
        closest = np.argmin(np.abs(annotations-detections[i]))
        signed_offset = detections[i] - annotations[closest]
        # first deal with the phase condition
        tolerance_window = np.zeros(2)
        if closest == 0:
            # first annotation, so use the forward interval
            annotation_interval = annotations[closest + 1] - annotations[closest]
            tolerance_window[0] = -phase_tolerance * annotation_interval
            tolerance_window[1] = phase_tolerance * annotation_interval
        else:
            # use backward interval
            annotation_interval = annotations[closest] - annotations[closest - 1]
            tolerance_window[0] = -phase_tolerance * annotation_interval
            tolerance_window[1] = phase_tolerance * annotation_interval
        # if the signed_offset is within the tolerance window range, then the phase is ok.
        my_eps = 1e-12
        correct_phase[i] = np.logical_and(signed_offset >= (tolerance_window[0] - my_eps), signed_offset <= (tolerance_window[1] + my_eps))
        # now look at the tempo condition calculate the detection interval back to the previous detection
        # (if we can)
        if i == 0:
            # first detection, so use the interval ahead
            detection_interval = detections[i + 1] - detections[i]
        else:
            # we can always look backwards, which is where we should look for the period interval
            detection_interval = detections[i] - detections[i - 1]
        # find out if the relative intervals of detections to annotations are less than the tolerance
        correct_tempo[i] = np.abs(1 - (detection_interval / annotation_interval)) <= tempo_tolerance
        # now want to take the logical AND between correct_phase and correct_tempo
        correct_beats = np.logical_and(correct_phase, correct_tempo)
        # we'll look for the longest continuously correct segment to do so, we'll add zeros on the front and end in case
        # the sequence is all ones
        correct_beats = np.concatenate([[0], correct_beats, [0]])
        # now find the boundaries
        d2 = np.argwhere(correct_beats == 0)
        d2 = d2.reshape(d2.size)
        correct_beats = correct_beats[1:-1]
        # in best case, d2 = 1 & length(checkbeats)
        contAcc = (np.amax(np.diff(d2)) - 1) / correct_beats.size
        totAcc = np.sum(correct_beats) / correct_beats.size
    return contAcc, totAcc


def makeVariations(annotations, start_ann, factor, offbeat):
    annotations = annotations[start_ann:]
    # Interpolate annoations
    interpolated_annotations = np.interp(np.arange(0, len(annotations), 1/factor), np.arange(0, len(annotations)), annotations)

    if offbeat:
        double_annotations = np.interp(
            np.arange(0, len(interpolated_annotations), 0.5),
            np.arange(0, len(interpolated_annotations)),
            interpolated_annotations)
        variations = double_annotations[1::2]
    else:
        variations = interpolated_annotations
    return variations


def verifyStartUpTime(start_up_time, annotations, phase_tolerance):
    closest = np.argmin(np.abs(annotations - start_up_time))
    annotation_interval = annotations[closest + 1] - annotations[closest]
    tolerance_window = np.array([-phase_tolerance*annotation_interval, phase_tolerance*annotation_interval])
    # now check if these tolerance windows straddle startUpTime
    if (annotations[closest] + tolerance_window[0] < start_up_time) and (annotations[closest] + tolerance_window[1] >= start_up_time):
        min_ann_time = annotations[closest]
        min_beat_time = annotations[closest] + tolerance_window[0]
    else:
        min_ann_time = start_up_time
        min_beat_time = start_up_time
    # finally double check that min_beat_time and min_ann_time can't go below 0
    min_ann_time = max(0, min_ann_time)
    min_beat_time = max(0, min_beat_time)
    return min_ann_time, min_beat_time


def run_test(wav, params=None):
    ours, theirs = BeatFind.run_song(wav, params)
    mainscore, backupscores = beatEvaluator2(ours, theirs)
    print("%s,%f]" % (wav, mainscore))
    return mainscore

def test_beats():
    path = "closed/"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    wavs = [x[:-4] for x in files if x[-4:] == ".wav"]
    print('Beginning tests for %d files' % len(wavs))
    results = []
    mainscores = []
    for i in wavs:
        mainscore = run_test(i)
        mainscores.append(mainscore)
    print('Done testing all files')
    print('Average score of %f' % (np.mean(mainscores)))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) != 1:
        #test_beats()
        run_test('competition')
    else:
        wavfile = argv[0]
        run_test(wavfile)

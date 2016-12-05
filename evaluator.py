import numpy as np
import os
import sys
import main


def beatEvaluator(detections,annotations):
    """
    Evaluate the accuracy of the algorithm
    :param detections: list of floats
    :param annotations: list of floats
    :return:
    """
    mainscore = 0  # amlT
    backup_scores = []  # amlC, cmlT, cmlC
    # Parameters
    min_beat_time = 5
    phase_tolerance = 0.175
    tempo_tolerance = 0.175
    # Run
    (cmlC, cmlT, amlC, amlT) = continuity(detections, annotations, tempo_tolerance, phase_tolerance, min_beat_time)
    mainscore = amlT
    backup_scores = [amlC, cmlT, cmlC]
    return mainscore, backup_scores


def continuity(detections, annotations, tempo_tolerance, phase_tolerance, min_beat_time):
    # Throw away first 5 seconds of beats
    detections = sorted(i for i in detections if i >= min_beat_time)
    annotations = sorted(i for i in annotations if i >= min_beat_time)
    # Convert to numpy array
    detections = np.array(detections)
    annotations = np.array(annotations)
    # Check parameters
    if tempo_tolerance < 0 or phase_tolerance < 0:
        raise ValueError('Tempo and phase tolerances must be positive.')
    if len(detections) < 2 or len(annotations) < 2:
        raise ValueError('At least two beats are required past the minimum beat time.')
    # Interpolate annoations
    double_annotations = np.interp(np.arange(0, len(annotations), 0.5), np.arange(0,len(annotations)), annotations)
    # Make different variations of annotations
    off_beats = double_annotations[1::2]  # Off-beats
    double_tempo = double_annotations  # Double tempo
    half_tempo_e = annotations[0::2]  # Half tempo (even)
    half_tempo_o = annotations[1::2]  # Half tempo (odd)
    all_variations = [annotations, off_beats, double_tempo, half_tempo_e, half_tempo_o]

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
        correct_phase[i] = np.logical_and(signed_offset >= tolerance_window[0], signed_offset <= tolerance_window[1])
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

def run_test(wav):
    ours, theirs = main.run_algorithm(wav, True)
    mainscore, backupscores = beatEvaluator(ours, theirs)
    print("'%s' scored %f" % (wav, mainscore))
    return mainscore

def test_beats():
    path = "all/"
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
        test_beats()
    else:
        wavfile = argv[0]
        run_test(wavfile)

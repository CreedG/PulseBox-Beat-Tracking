import numpy as np
import os
import sys
#import main

def beatEvaluator2(detections,annotations, metrical_options):
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


def run_test(wav):
    ours, theirs = main.run_algorithm(wav, True)
    mainscore, backupscores = beatEvaluator2(ours, theirs)
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
    """annotations = open(r'all/open_001.txt').read().splitlines()
    for i in range(len(annotations)):
        annotations[i] = float(annotations[i])
    # Test vectors
    #detections = np.array([1.2657294267021642, 1.568023949874671, 1.9546569347513079, 3.0570684868281743, 3.0668133442972927, 3.546497643559616, 4.697968437516527, 5.67826200675658, 5.825876108296273, 6.759687270539722, 6.632509093108422, 7.80215572560814, 8.37886131014419, 9.20835533625288, 9.611862278459222, 9.80075297239434, 10.694498290810145, 11.530791726045795, 11.76907933509561, 12.5557160433159, 12.859122394965839, 13.264508996054902, 13.801828506105956, 14.606767612190616, 15.353095496948063, 16.425388378860525, 16.555096247382995, 17.151710026411504, 17.72757810258656, 18.916160922908144, 19.230039674858915, 20.068904375040173, 19.940793293703454, 21.259822840138455, 21.55170529610863, 22.02343900723623, 22.761959538649656, 22.917098030387294, 23.359042667075947, 24.045922528412806, 24.831931460552706, 26.047135037312557, 26.633091612523106, 26.963850182348644, 27.791135196099113, 27.87226507527578, 28.58150066351335, 29.58200236627728, 29.90505760749002, 30.28718246204631])
    #detections = np.array([0.8038516062701249, 1.2726279057074361, 1.786065354065935, 3.1478641082438377, 3.6194219372787098, 3.6618928134216557, 4.121272711232493, 4.803360767173317, 5.687967918243281, 6.88553960666437, 7.452246746851404, 7.571299179479291, 7.800090127832114, 8.321369096936117, 9.311317234403678, 9.93954077681223, 11.10277387852636, 11.165613828518747, 11.59606452832368, 12.74765636070147, 12.617212045377661, 13.860875845533323, 13.81920164001412, 14.584838193547924, 15.55251898065957, 16.52800334727093, 16.494547841856377, 17.531268941963546, 17.74072531230666, 18.496432066030774, 18.84546937302626, 19.98836926931529, 20.708577784596173, 20.996682625639114, 21.444529107319784, 22.40732420064309, 22.633875912077407, 23.410475099030535, 24.28163506595328, 24.91846891834848, 24.61579774949844, 25.246007607074255, 25.77838978041116, 27.242154267368907, 27.81660695744746, 28.090168764595667, 28.248415450730747, 29.530369591303607, 29.806364312859106, 30.649908404138397])
    #detections = np.array([0.5214484141719338, 1.1522636393921075, 1.7091719324132044, 2.3874741594596176, 2.954378699844038, 3.5214204329480987, 4.184950723659208, 4.756212632778842, 5.3712725424884304, 5.989720791194427, 6.6097596819318545, 7.226122551475426, 7.80904082687365, 8.387462386892434, 8.95816951002742, 9.538729536959641, 10.113504790330207, 10.761041379103563, 11.380811906584281, 11.937709584999162, 12.536691658974904, 13.17131993932312, 13.786879130364467, 14.363165974324307, 14.933221069743018, 15.575874190354883, 16.150857316262687, 16.816246112573417, 17.41244412947066, 17.98669198830264, 18.631993708448892, 19.19821559662707, 19.832673522097192, 20.400476689887505, 21.02636700119012, 21.575389545816755, 22.21694194508407, 22.78483557505346, 23.409811268188633, 23.946764274968164, 24.572163497154136, 25.2318671570195, 25.803414908974585, 26.42817259811168, 26.959286830483336, 27.62133135533747, 28.16538569907559, 28.833744924338962, 29.429296800620545, 29.967963470264074])
    #detections = np.array([0.5805817935572541, 1.2387984746223668, 1.9751389891344766, 2.613194243463319, 2.962653559367622, 3.8108113918904567, 4.393821995221544, 5.167158823328257, 5.690376274265511, 5.960745270134324, 6.64778104785243, 7.288777310436178, 8.051478253099042, 8.640535402668753, 9.207171843662335, 9.594689709988966, 10.378355507346448, 10.936302009237338, 11.397596028738398, 12.268468761510483, 12.682143857120415, 13.169014476556768, 13.790269071387238, 14.600036228078606, 15.27277263437867, 15.738753038672462, 16.28025970306617, 16.74419066550116, 17.77564727664801, 18.094806392249406, 18.89054698731576, 19.52855147462462, 19.938257065672687, 20.753603922975596, 21.232426169795968, 21.709151882127635, 22.198366798758713, 22.833740088798102, 23.421397356772886, 24.117829448622853, 24.594343514925622, 25.426323069260107, 26.190383356302707, 26.558399438039178, 27.012342418777195, 27.663262692731667, 28.231487383594963, 28.871956365697777, 29.36415505350445, 29.939591773893028])
    mainscore, backup_scores = beatEvaluator2(detections, annotations, None)
    print(mainscore)
    print(backup_scores)"""
    argv = sys.argv[1:]
    if len(argv) != 1:
        test_beats()
    else:
        wavfile = argv[0]
        run_test(wavfile)

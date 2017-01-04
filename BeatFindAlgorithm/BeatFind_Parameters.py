'''These are parameters that were varied to train the algorithm. They represent constants and values used in computing the tempo and phase. Instead of arbitrarily selecting
 these parameters, we used machine learning to find optimal values'''

#1. DATA ACQUISITION PARAMETERS

#The cutoffs for the different frequency ranges (freq resolution is 21.53Hz)
P_FREQ_BAND_1 = 7
P_FREQ_BAND_2 = 26
P_FREQ_BAND_3 = 188


#2. TEMPO PARAMETERS

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
                         [[[0.18, 1], 1, [0], [8.83503952234608]]],
                         [[[0.18, 1], 1, [0], [0.017411179300045182]]]]

#Spans most recent 3 seconds
P_PEAK_PICK_ARRAY_SHORT = [[[[0.18, 4], 2, [0, 0], [0.4637504524433074, 0.23183151247396472]], [[1.5, 4], 1, [0], [1.0265975602231432]]],
                           [[[0.18, 4], 2, [0, 0], [0.11600325904505178, 0.10025863779222567]], [[1.5, 4], 1, [0], [1.987351968103246]]],
                           [[[0.18, 4], 2, [0, 0], [1.0746105972730395, 0.3642503639087244]], [[1.5, 4], 1, [0], [1.2603563519577021]]],
                           [[[0.18, 4], 2, [0, 0], [5.384319540310969, 1.1746500524488648]], [[1.5, 4], 1, [0], [0.5021989052809293]]],
                           [[[0.18, 4], 2, [0, 0], [0.7270916362492784, 0.005458336170681191]], [[1.5, 4], 1, [0], [0.04874661415392632]]]]

#Various parameters used to guide the selection of the best tempo from all of the autocorrelation data
P_PEAK_CONSENSUS_TOLERANCE = 0.03628117913832199
P_DISTINCT_PEAK_TOLERANCE = 0.027189739840437717
P_TEMPO_START =  4.074271285190735
P_MED_WEIGHT = 0.4777492238385939
P_SHORT_WEIGHT = 0.4750856179435148
P_COMBINE_TEMPOS_THRESH = 0.05
P_MAX_TIME_MED_CORR = 16.251637885037976
P_LONG_TEMPO_CHANGE_PENALTY = 2.5484785166187054
P_SHORT_CORR_THRESH = 10
P_TEMPO_CONSENSUS_TOLERANCE = 0.0581057805472111
P_TEMPO_WEIGHT_START = 1.6652186015163242
P_PENALTY_RANGE_LOW = 10.273462868
P_PENALTY_RANGE_HIGH = 15.102342432

#3. BEAT PLACEMENT PARAMETERS

#Various parameters used to find the beat offset
P_SNAP_THRESH_1 = 10
P_SNAP_THRESH_2 = 9
P_SNAP_THRESH_3 = 1

P_BEAT_START_MULT = 1.2
P_BEAT_SHIFT = 0.02425174259739585
P_BEAT_THRESH_START = 0.85
P_BEAT_THRESH_DECAY = 4
P_BEAT_THRESH_UNSTABLE = 12

P_BEAT_START_PERCENT_STABLE = 0.6023182410982814
P_BEAT_START_PERCENT_UNSTABLE = 0.2
P_BEAT_END_PERCENT_STABLE = 1.5
P_BEAT_END_PERCENT_UNSTABLE = 1.161022051183548



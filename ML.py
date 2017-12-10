from evaluator2 import run_test
import numpy as np


def testbed_sp(params):
    scores = []
    wavs = ['challenge_008',
            'closed_005',
            'challenge_018',
            'challenge_015',
            'closed_001',
            'closed_019',
            'challenge_009',
            'closed_010',
            'closed_025',
            'closed_006',
            'challenge_011',
            'closed_009',
            'challenge_031',
            'closed_022',
            'challenge_003',
            'closed_020',
            'challenge_023',
            'challenge_010',
            'challenge_026',
            'closed_002',
            'closed_018',
            'closed_016',
            'challenge_027']
    for i in wavs:
        if i == 'challenge_011':
            scores.append(run_test(i, params))
            print(i + " " + str(scores[-1]))

    score = np.mean(scores)
    print("TOTAL average score for all songs:", score)
    return score


def testbed(params):
    scores = []
    pre = "closed_00"
    for i in range(1, 26):
        if i == 10:
            pre = pre[:-1]
        scores.append(run_test(pre + str(i), params))
        # print(i + " " + str(scores[-1]))
    pre = "challenge_00"
    for i in range(1, 34):
        if i == 10:
            pre = pre[:-1]
        if i != 12:
            scores.append(run_test(pre + str(i), params))

    score = np.mean(scores)
    print("TOTAL average score for all songs:", score)
    return score


def ml():
    ALL_PARAMETERS = [1.2, 1.2, .8, 1, 5, 10, 1,0,1,0, .675,1]

    cur_param = 0

    print("Running first session")
    cur_max = testbed(ALL_PARAMETERS)
    while True:
        baseline = ALL_PARAMETERS[cur_param]
        ALL_PARAMETERS[cur_param] *= np.random.normal(1, 0.15)
        if np.random.uniform(0,6) > 5:
            ALL_PARAMETERS[cur_param] = .1
        print("Running session with param", cur_param, "increased to val", ALL_PARAMETERS[cur_param], "All params:",
              ALL_PARAMETERS)
        score = testbed(ALL_PARAMETERS)
        if score > cur_max:
            print("Changing", cur_param, "gave new best", score)
            cur_max = score
        else:
            ALL_PARAMETERS[cur_param] = baseline
        cur_param += 1
        if cur_param >= len(ALL_PARAMETERS):
            cur_param = 0


if __name__ == "__main__":
    ml()

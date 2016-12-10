import struct
import wave, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import sleep

import scipy.fftpack
import scipy.signal
import scipy.io

import time
from ring import Ring

import math


time_step = 512.0/44100

debugNum = 0



#Initializes my ring class with the known period currently
def init_phase_globals(period):
    global backlink, cumscore

def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def detect_phase2(period, onsetenv):
    tightness = 6
    alpha = .7
    oesr = 250#86
    pd = int(period * oesr)
    templt = [math.exp(-0.5*((x/(pd/32))**2)) for x in range(-pd, pd)]
    localscore = np.convolve(templt,onsetenv)
    temp = [round(len(templt)/2) + x for x in range(len(onsetenv))]
    #print(temp)
    # Our vector lengths should be half of theirs but its not.
    localscore = localscore[temp]
    backlink = [0] * len(localscore)
    cumscore = [0] * len(localscore)
    prange = range(round(-2*pd),-round(pd/2))
    txwt = [-tightness*abs((math.log(x/-pd))**2) for x in prange]
    starting = 1
    for i in range(100,len(localscore)):
        timerange = [i + x for x in prange]
        zpad = max([0, min([-timerange[0],len(prange)])])
        scorecands = [sum(x) for x in zip( txwt, [0] * zpad + cumscore[timerange[zpad]: timerange[-1]])]
        vv = max(scorecands)
        xx = scorecands.index(vv)
        cumscore[i] = vv + localscore[i] - alpha
        if starting == 1 and localscore[i] < .01 * max(localscore):
            backlink[i] = -1
        else:
            backlink[i] = timerange[xx]
            starting = 0
    maxm = max(cumscore)
    bestendposs = find(cumscore, lambda x: x > 0.5*maxm)
    bestendx = max(bestendposs)
    b = [bestendx]
    while backlink[b[-1]] > 0:
        b = b + [backlink[b[-1]]]
    b.reverse()
    boe = localscore[b]
    bwinlen = 5
    sboe = np.convolve(np.hanning(bwinlen), boe)
    sboe = sboe[int(np.floor(bwinlen/2)) + 1: len(boe)]
    thsboe = 0.5*math.sqrt(np.mean(sboe**2))
    b = b[np.min(np.argwhere(sboe>thsboe)): np.max(np.argwhere(sboe>thsboe))]
    return [x / oesr for x in b]


if __name__ == "__main__":
    # open_001.wav test
    onsetenv = scipy.io.loadmat(r'C:\Users\William Heimsoth\Documents\Fall 2016\ENEE408A\onsetenv.mat')['onsetenv']
    onsetenv = onsetenv.reshape(onsetenv.size)
    period = 0.60081625
    beats = detect_phase2(period, onsetenv)
    print(len(beats))
    print(beats)
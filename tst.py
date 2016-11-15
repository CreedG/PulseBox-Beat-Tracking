import matplotlib.pyplot as plt
import wave, sys
import numpy as np
from scikits.talkbox.features import mfcc
import scipy.signal as sig

def bpm(fi, le, sec):
    f = open('/home/josh/Documents/beats/training_set/open/' + fi + '.txt')
    c = f.read().split()
    a = [float(x) for x in c]
    b = []
    for i in range(1,len(a)-1):
       b.append(60/(a[i] - a[i-1]))
    tim = 60 / (sum(b)/len(b))
    print "Official average BPM = " + str(sum(b)/len(b))
    c = [0] * le
    for i in range(0,int(sec/tim)+1):
        c[int(i * tim * (le / sec))] = 50
    return c
def text_bpm(fi, le, sec):
    f = open('/home/josh/Documents/beats/training_set/open/' + fi + '.txt')
    c = f.read().split()
    a = [float(x) for x in c]
    b = []
    for i in range(1,len(a)-1):
       b.append(60/(a[i] - a[i-1]))
    #plt.plot(b)
    #plt.show()
    print "Official average BPM = " + str(sum(b)/len(b))
    #print max(b) - min(b)

    c = [0] * le
    for i in range(0,len(a)):
        c[int(a[i]*le/sec)] = 50
    return c

def calculate_bpm(nceps, time, peaks):
    na = [float(x) * time / nceps for x in peaks]
    diff = []
    for i in range(1,len(na)-1):
        diff.append(na[i] - na[i-1])
    avg = sum(diff)/len(diff)
    print("Calculated Bpm = " + str(60./avg) + "or " + str(30./avg) + " or " + str(15./avg))

def wot(fi):
    wav = wave.open('/home/josh/Documents/beats/training_set/open/' + fi + '.wav')
    wav.setpos(wav.getnframes()/3)
    frames_to_read = wav.getnframes()/10
    Y = wav.readframes(frames_to_read)
    time = frames_to_read / wav.getframerate()

    frames = np.fromstring(Y,dtype='int16') * 1.

    ceps, mspec, spec = mfcc(frames)

    w = (ceps[:,1] - ceps[:,0])**2 + (ceps[:,2] - ceps[:,1])**2 + (ceps[:,3] - ceps[:,2])**2
    w -= np.mean(w)
    w /= max(w)
    a = np.correlate(w,w,mode='full')
    a = a[len(a)/2 - 1:]
    b = sig.find_peaks_cwt(a,np.arange(1,20 * time))
    calculate_bpm(len(a), time, b[0:len(b)-3])
    back = bpm(fi, len(a), time)
    c = range(0,len(a))
    c = [100 if x in b else 0 for x in c]
    plt.plot(a)
    plt.plot(c,color='#cccccc')
    plt.plot(back,color='#cc0000')
    plt.show()
    exit(0)
    m = []
    r = [0]
    for i in range(0,len(a)):
        m.append(a[i] *len(a)/ (800 * (len(a) - i)))
    for i in range(0,len(a)):
        if m[i] > 1 and i > r[-1] + 20:
            r.append(i)
    s = []
    for i in range(1,len(r)):
        s.append(r[i] - r[i-1])
    s = s[:-1]
    avg = float(sum(s))/len(s)
    t = avg / 4 * 30 / len(a)
    print 60 / t

def main(argv):
    if len(argv) != 1:
        print "Read in file plx"
        return
    wot(argv[0])

if __name__ == "__main__":
    main(sys.argv[1:])

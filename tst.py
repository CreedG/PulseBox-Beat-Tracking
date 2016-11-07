import matplotlib.pyplot as plt
import wave, sys
import numpy as np
from scikits.talkbox.features import mfcc
wav = wave.open('aphex.wav')
Y = wav.readframes(wav.getnframes())
frames = np.fromstring(Y,dtype='int16') * 1.
ceps, mspec, spec = mfcc(frames)
w = (ceps[:,1] - ceps[:,0])**2 + (ceps[:,2] - ceps[:,1])**2 + (ceps[:,3] - ceps[:,2])**2
w -= np.mean(w)
w /= max(w)
a = np.correlate(w,w,mode='full')
a = a[len(a)/2 - 1:]
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
plt.plot(m)
plt.show()

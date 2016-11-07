import numpy
import matplotlib.pyplot as plt
from scipy.fftpack import dct
def hz2mel(hz):
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt, nfft, samplerate,lowfreq, highfreq):
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra, L):
    if L > 0:
        ncoeff = numpy.shape(cepstra)[0]
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def mfcc(fftd,samplerate=44100, numcep=13, ceplifter=22, nfilt=26,nfft=1024,lowfreq=0,highfreq=None):
    highfreq= highfreq or samplerate/2
    pspec = 1.0/nfft * numpy.square(fftd)
    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log
    feat = numpy.log(feat)
    feat = dct(feat, type=2, norm='ortho')
    #feat = lifter(feat,ceplifter)
    return feat

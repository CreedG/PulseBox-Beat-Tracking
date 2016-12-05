import matplotlib.pyplot as plt
import numpy
class Ring:
    def __init__(self, period, ticks = 1000, window_size = 13, downshift = .8):
        self.downshift = downshift
        self.data = [0] * ticks
        self.l = ticks
        self.beat = period / 2
        self.period = period
        self.to_put = []
        assert(window_size % 2 == 1)
        self.window_size = window_size
        self.hanning = numpy.hanning(window_size)
        self.phase = 0

    def __repr__(self):
        return repr(self.data)
    def plot(self):
        tix = numpy.arange(self.beat - self.period/2, self.beat + self.period/2, self.period / self.l )[0:1000]
        plt.plot([self.beat, self.beat],[0,1.5 * max(self.data)], 'r')
        plt.plot(tix, self.data)
        plt.xlim([tix[0], tix[-1]])
        plt.ylim([0, 1.5 * max(self.data)])
        plt.show()
    def __len__(self):
        return self.l

    #Makes sure phase shift works correctly
    def check_yo(self):
        m = max(self.data)
        p = min([float(i)/self.l * self.period -self.period/2 for i, j in enumerate(self.data) if j == m], key=abs)
        if(abs(p) > .01):
            print(p)
            exit(1)

    def shift_phase(self, period):
        shift = int(self.phase * self.l / self.period) + 1
        if self.phase < 0:
            last = self.data[shift:]
            rest = self.data[:shift]
            self.data = last + rest
        else:
            first = self.data[:shift]
            rest = self.data[shift:]
            self.data = rest + first
        self.check_yo()

    #Moves window forward
    def iter(self, period, time):
        self.period = period
        self.shift_phase(period)
        tmp = self.to_put
        self.to_put = []
        for t, s in tmp:
            self.insert(t, s)
        if len(self.to_put) > 0:
            self.generate_next_beat(time)
            self.iter(period, time)

    def insert(self, time, strength):
        if time > self.beat + self.period/2:
           self.to_put.append((time, strength))
           return
        add_to = int((time - self.beat) / self.period * self.l ) + self.l // 2
#Ignore ticks outside current ticks. Fairly graceless but makes sense for large forward phase shifts
        if(add_to < 0 or add_to >= self.l):
            return
        j = 0
        for i in range( add_to - self.window_size//2, add_to + self.window_size//2):
            if i >= 0 and i < self.l:
                self.data[i] += strength * self.hanning[j]
            j += 1

    def generate_next_beat(self, time):
        self.phase = min([float(i)/self.l * self.period -self.period/2 for i, j in enumerate(self.data) if j == max(self.data)], key=abs)
        if time < self.beat + self.period / 2 :
            return self.beat
        self.data = [x * self.downshift for x in self.data]
        self.beat = self.beat + self.phase + self.period
        return self.beat

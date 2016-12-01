import matplotlib.pyplot as plt
import numpy
class Ring:
    def __init__(self, ticks, period, window_size):
        self.data = [0] * ticks
        self.l = ticks
        self.beat = period / 2
        self.period = period
        self.to_put = []
        assert(window_size % 2 == 1)
        self.windo = window_size
        self.hanning = numpy.hanning(window_size)
        self.phase = 0
    def __repr__(self):
        return repr(self.data)
    def plot(self):
        plt.plot(self.data)
        plt.show()
    def __len__(self):
        return self.l
    def iter(self, period, time):
        #print((time, self.period, self.beat))
        if time < self.beat + self.period/2:
            return 0
        self.period = period
        shift = int(self.phase * self.l / self.period) + 1
        if self.phase > 0:
            last = self.data[-shift:]
            rest = self.data[:-shift]
            self.data = last + rest
        else:
            first = self.data[:shift]
            rest = self.data[shift:]
            self.data = rest + first
        self.beat = self.n_beat()
        #print(self.beat)
        tmp = self.to_put
        self.to_put = []
        for t, s in tmp:
            self.insert(t, s)
        return len(self.to_put)
    def insert(self, time, strength):
        if time > self.beat + self.period/2:
           self.to_put.append((time, strength))
           return
        add_to = int((time - self.beat) * self.period * self.l ) + self.l // 2
        assert(add_to >= 0)
        j = 0
        for i in range( add_to - self.windo//2, add_to + self.windo//2):
            if i >= 0 and i < self.l:
                self.data[i] += strength * self.hanning[j]
            j += 1
    def n_beat(self):
        m = max(self.data)
        mi = 1/m
        #self.data = [x * mi for x in self.data]
        self.phase = min([float(i)/self.l * self.period -self.period/2 for i, j in enumerate(self.data) if j == m], key=abs)
        self.next_beat = self.beat + self.phase + self.period
        return self.next_beat
    def ne_beat(self):
        return self.next_beat

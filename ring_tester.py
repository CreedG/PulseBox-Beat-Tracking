from ring import Ring
def detect_phase(period, times, strengths, time):
    global r, prev_len
    if 'r' not in globals():
        prev_len = 0
        r = Ring(1000, period, 23)

    r.iter(period, time)
    for i in range(prev_len, len(times)):
        r.insert(times[i], strengths[i])
    prev_len = len(times)

    r.plot()
    return r.generate_next_beat()

l = [.4]
print(detect_phase(.6, l, [1] * len(l), 5))

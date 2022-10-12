import numpy as np
import matplotlib.pyplot as plt
import net_estimator as ne

#Speed of light, meters per second
c = 299792458

class Clock:
    def __init__(self, lam, br, wh):
        self.time = 0
        self.frequency_error = 0
        self.frequency_lambda = lam
        self.brown_sigma = br
        self.white_sigma = wh

    def advance(self, time_int):
        self.frequency_error += np.random.normal(0, self.frequency_lambda * time_int)
        self.time += (1 + self.frequency_error) * time_int + np.random.normal(0, self.brown_sigma * time_int)
        return self.time + np.random.normal(0, self.white_sigma)

class CorrectedClock(Clock):
    def __init__(self, lam, br, wh, alpha, h):
        super().__init__(lam, br, wh)
        self.alpha = alpha
        self.h = h
        self.correction = 0

    def run(self, time_int, rx_times, tx_times):
        raw_time = self.advance(time_int)
        return self.correct(raw_time, tx_times)

    def correct(self, rx_times, tx_times):
        eps = np.average(tx_times - rx_times)
        self.correction = self.alpha * self.correction + self.h * eps
        self.time += self.correction
        return self.time

def clocksquare(a, b, dist):
    ret = np.zeros((a * b, 2))
    for aa in range(a):
        for bb in range(b):
            ret[b * aa + bb, :] = np.array([aa * dist, bb * dist])
    return ret

def sort_tx_indices(delays, ind):
    indices = []
    while True:
        min_ind = -1
        for j, de in enumerate(delays[:,ind]):
            if de > 0 and j not in indices:
                if min_ind == -1:
                    min_ind = j
                elif de < delays[min_ind, ind]:
                    min_ind = j
        if min_ind != -1:
            indices.append(min_ind)
        else:
            break
    return indices

def create_tree(detect, dist, master_ix):
    masters = np.zeros(detect.shape[0]) - 1
    layer = []
    newlayer = [master_ix]
    while len(newlayer) > 0:
        layer = newlayer
        newlayer = []
        for n in range(detect.shape[0]):
            min_ix = -1
            min = np.inf
            if masters[n] == -1:
                for m in layer:
                    if detect[n, m] and dist[n,m] < min:
                        min_ix = m
                        min = dist[n,m]
                if min_ix != -1:
                    masters[n] = min_ix
                    newlayer.append(n)
    return masters

if __name__ == '__main__':
    clockpositions = clocksquare(6, 6, 500)
    master_ix = 14
    distances = ne.distances(clockpositions)
    detects, _ = ne.detect(distances, np.array([]), 750)
    masters = create_tree(detects, distances, master_ix)
    delays = detects * distances / c
    clocks = []
    for i in range(36):
        clocks.append(CorrectedClock(0.1, 0.1, 0.00001, 0.15, 0.75))
    times = 5000
    tx_times = np.zeros(36)
    timediffs = np.zeros((times, 36))
    for i in range(times):
        for j, clock in enumerate(clocks):
            tx_times[j] = clock.advance(0.001)
        time_errors = tx_times - (i + 1)
        for j, clock in enumerate(clocks):
            tx_indices = sort_tx_indices(delays, j)
            tx_times_j = tx_times[tx_indices]
            rx_times = np.zeros(len(tx_indices))
            prevdelay = 0
            for k, ind in enumerate(tx_indices):
                rx_times[k] = (clock.advance(delays[ind, j] - prevdelay))
                prevdelay = delays[ind, j]
            # Clock nr 0 is master
            #if j == master_ix:
            if False:
                time = clock.time
            else:
                time = clock.correct(rx_times, tx_times_j)
            timediffs[i, j] = time - clocks[master_ix].time
    plt.plot(timediffs[0:times:100,:])
    plt.show()
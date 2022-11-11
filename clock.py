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
        self.frequency_error += np.random.normal(0, np.sqrt(self.frequency_lambda * 16.0*(np.pi **2)*time_int**2))
        self.time += (1 + self.frequency_error) * time_int + np.random.normal(0, np.sqrt(4.0 * np.pi * self.brown_sigma * time_int))
        return self.get_time()

    def get_time(self):
        return self.time + np.random.normal(0, np.sqrt(self.white_sigma))
class CorrectedClock(Clock):
    def __init__(self, lam, br, wh, alpha, h):
        super().__init__(lam, br, wh)
        self.alpha = alpha
        self.h = h
        self.correction = 0
        self.delay_compensation = 0

    def run(self, time_int, rx_times, tx_times):
        raw_time = self.advance(time_int)
        return self.correct(raw_time, tx_times)

    def correct(self, rx_time, tx_time):
        eps = tx_time - rx_time
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
    masters = np.zeros(detect.shape[0], dtype=int) - 1
    nr_of_slaves = np.zeros(detect.shape[0], dtype=int)
    delay_request_stamps = np.zeros(detect.shape[0])
    layer = []
    newlayer = [master_ix]
    while len(newlayer) > 0:
        layer = newlayer
        newlayer = []
        for n in range(detect.shape[0]):
            if n == master_ix:
                continue
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
                    nr_of_slaves[min_ix] += 1
                    delay_request_stamps[n] = nr_of_slaves[min_ix]
    return masters, delay_request_stamps

def estimate_propdelay(delay1, delay2):
    propdelay = (delay1 + delay2)/2
    eps = (delay1 - delay2)/2
    return eps, propdelay

if __name__ == '__main__':
    nr_of_clocks = 36
    clockpositions = clocksquare(6, 6, 500)
    master_ix = 0
    distances = ne.distances(clockpositions)
    detects, _ = ne.detect(distances, np.array([]), 750)
    fs = 1000
    stamp_intval = 0.0001
    times = 5000

    whitesigmasq = 0.00001
    brownsigmasq = 0.1
    freqlambdasq = 0.1
    whitesigmasq_master = 0.00001
    brownsigmasq_master = 0.001
    freqlambdasq_master = 0.001

    masters, delay_request_stamps = create_tree(detects, distances, master_ix)
    max_stamp = max(delay_request_stamps)
    delays = detects * distances / c
    clocks = []
    for i in range(nr_of_clocks):
        if i == master_ix:
            clocks.append(CorrectedClock(0.000001, 0.000001, 0.00000001, 0.15, 0.75))
            #clocks.append(Clock(0.0001, 0.0, 0.0))
        else:
            clocks.append(CorrectedClock(0.0001, 0.0001, 0.00000001, 0.15, 0.75))
            #clocks.append(Clock(0.0001, 0.000, 0.0))
    tx_times = np.zeros(nr_of_clocks)
    prop_delay_ests = np.zeros(nr_of_clocks)
    delay_1 = np.zeros(nr_of_clocks)
    delay_2 = np.zeros(nr_of_clocks)
    timediffs = np.zeros((times, nr_of_clocks))
    time_errors = np.zeros(times)
    for j, clock in enumerate(clocks):
        tx_times[j] = clock.advance(max_stamp * stamp_intval)
    for i in range(times):
        for j, clock in enumerate(clocks):
            tx_times[j] = clock.advance(0.001 - max_stamp * stamp_intval)
        for j, clock in enumerate(clocks):
            m_ix = masters[j]
            tx_time = tx_times[m_ix]
            rx_time = clock.advance(delays[m_ix, j])
            # Clock nr master_ix is master
            delay_1[j] = rx_time - tx_time
           # delreq_txtime = clock.advance(0.0001 - delays[m_ix, j])
        for t in np.arange(1, max_stamp + 1):
            delay_advances = np.zeros(nr_of_clocks)
            for j, clock in enumerate(clocks):
                tx_times[j] = clock.advance(stamp_intval - delay_advances[j])
            for j, clock in enumerate(clocks):
                if delay_request_stamps[j] == t:
                    m_ix = masters[j]
                    m_clock = clocks[m_ix]
                    tx_time = tx_times[j]
                    rx_time = m_clock.advance(delays[m_ix, j])
                    delay_advances[m_ix] += delays[m_ix, j]
                    delay_2[j] = rx_time - tx_time
                    eps, prop_delay_ests[j] = estimate_propdelay(delay_1[j], delay_2[j])
                    if j == master_ix or not isinstance(clock, CorrectedClock):
                        time = clock.get_time
                    else:
                        clock.delay_compensation = prop_delay_ests[j]
                        time = clock.correct(eps, 0)
        for j, clock in enumerate(clocks):
            if j == master_ix or not isinstance(clock, CorrectedClock):
                time_errors[i] = clock.get_time() - (i + 1) * (1 / fs)
            timediffs[i, j] = clock.get_time() - clocks[master_ix].time
    plt.figure(0)
    plt.plot(timediffs[0:times:100,:])
    plt.figure(1)
    plt.plot(time_errors)
    plt.show()
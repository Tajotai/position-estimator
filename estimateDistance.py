#!/usr/bin/env python3

import os, sys, signal, argparse
import numpy as np
import matplotlib.pyplot as plt
import net_estimator as nest
import position_estimator as pe

def calculate_distance(rx_pow, tx_pow):
    '''
    gives the distance by transmission and receiving power of the signal,
    using the Friis model for path loss in free space. The constant
    173.491005787 is the wavelength of signal with frequency 1.728 MHz

    :param rx_pow: receiver power in dB
    :param tx_pow: transmitter power in dB
    :return: distance in meters
    '''
    return 10 ** ((tx_pow - rx_pow)/20 + np.log10(0.0999308193333333 /(4 * np.pi)))

def bilaterate(pos1, pos2, d1, d2):
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]
    sqd3 = (x2-x1)**2+(y2-y1)**2
    d3 = np.sqrt(sqd3)
    cosalpha = (d1**2 + d3**2 - d2**2)/(2*d1*d3)
    if (cosalpha > -1 and cosalpha < 1):
        # proj = cosalpha*d1
        proj = (d1**2 - d2**2 + sqd3)/(2 * d3)
        ortho = np.sqrt(1 - cosalpha**2) * d1
        result1 = (x1 + (x2 - x1) * proj/d3 + (y2 - y1) * ortho/d3, y1 + (y2 - y1) * proj/d3 + (x1 - x2) * ortho/d3)
        result2 = (x1 + (x2 - x1) * proj/d3 - (y2 - y1) * ortho/d3, y1 + (y2 - y1) * proj/d3 - (x1 - x2) * ortho/d3)
        return (result1, result2)
    else:
        d = (d1 + d2 + d3)/2
        if d1 > d2:
            result = (x1 + d * (x2 - x1)/d3, y1 + d * (y2 - y1)/d3)
        else:
            result = (x2 + d * (x1 - x2)/d3, y2 + d * (y1 - y2)/d3)
        return (result, result)

def find_ix(pair1, pair2, pair3):
    minvar = None
    ix = None
    for i in [0, 1]:
        e1 = pair1[i]
        for j in [0, 1]:
            e2 = pair2[j]
            for k in [0, 1]:
                e3 = pair3[k]
                avrx = (e1[0] + e2[0] + e3[0]) / 3
                avry = (e1[1] + e2[1] + e3[1]) / 3
                var = (e1[0] - avrx) ** 2 + (e2[0] - avrx) ** 2 + (e3[0] - avrx) ** 2 + \
                      (e1[1] - avry) ** 2 + (e2[1] - avry) ** 2 + (e3[1] - avry) ** 2
                if minvar is None or var < minvar:
                    minvar = var
                    ix = (i, j, k)
    return ix

def trilaterate_simple(pos1, pos2, pos3, d1, d2, d3):
    '''
    Gives a simple method to estimate a node's position given the positions of three
    other nodes (pos1, pos2, pos3) and distances from them.
    Executes the task by finding pairwise estimates from three triangles formed by the
    nodes and current node to locate, then finding the correct one from each pair by
    calculating variance (find_ix), then averaging over three estimates.
    :param pos1: position of node1
    :param pos2: position of node2
    :param pos3: position of node3
    :param d1: distance from node1
    :param d2: distance from node2
    :param d3: distance from node3
    :return: calculated position of the node to be located
    '''
    pair1 = bilaterate(pos1, pos2, d1, d2)
    pair2 = bilaterate(pos1, pos3, d1, d3)
    pair3 = bilaterate(pos2, pos3, d2, d3)
    ix = find_ix(pair1, pair2, pair3)
    est1 = pair1[ix[0]]
    est2 = pair2[ix[1]]
    est3 = pair3[ix[2]]
    return ((est1[0] + est2[0] + est3[0]) / 3, (est1[1] + est2[1] + est3[1]) / 3)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--fileIn', nargs=1, help='Input file', required=True)
    args = parser.parse_args()
    signalgenmode = 'random'
    x = np.array([])
    if signalgenmode == 'file':
        if (os.path.exists(args.fileIn[0])):
            x = np.genfromtxt(args.fileIn[0], delimiter = ' ', dtype = None, encoding='utf-8');
        else:
            print(filePath, "not found!")
    else:
        x = genx()

    refFound = False
    pos = dict()
    signal = dict ()
    devs=set()


    for s in x:
        dev1 = s[0]
        dev2 = s[1]
        devs.add(dev1)
        devs.add(dev2)
        if dev1 not in signal:
            signal[dev1] = dict ()
        if dev2 not in signal:
            signal[dev2] = dict ()
        signal[dev1][dev2] = s[2]
        signal[dev2][dev1] = s[2]

        if len(pos) == 0:
            for dev3 in signal:
                if dev3 != dev1 and dev3 != dev2:
                    if dev3 in signal[dev1] and dev3 in signal[dev2]:
                        print(dev3,dev1,dev2)
                        pos[dev3] = (0, 0)

                        dist31 = calculate_distance(signal[dev3][dev1], 23)
                        pos[dev1] = (0, dist31)

                        dist32 = calculate_distance(signal[dev3][dev2], 23)
                        dist12 = calculate_distance(signal[dev1][dev2], 23)

                        dev2Pos = bilaterate(pos[dev3], pos[dev1], dist32, dist12)
                        if dev2Pos[0][0] >= 0 and dev2Pos[0][1] >= 0:
                            pos[dev2] = dev2Pos[0]
                        else:
                            pos[dev2] = dev2Pos[1]

    while len(pos) < len(devs):
        for dev in signal:
            for dev1 in signal:
                for dev2 in signal:
                    for dev3 in signal:
                        if dev != dev1 and dev != dev2 and dev != dev3 and dev1 != dev2 and dev1 != dev3 and dev2 != dev3 and dev in signal[dev1] and dev in signal[dev2] and dev in signal[dev3] and dev1 in pos and dev2 in pos and dev3 in pos and dev not in pos:
                            print(dev,dev1,dev2,dev3)
                            dist14 = calculate_distance(signal[dev1][dev], 23)
                            dist24 = calculate_distance(signal[dev2][dev], 23)
                            dist34 = calculate_distance(signal[dev3][dev], 23)
                            pos[dev] = trilaterate_simple(pos[dev1], pos[dev2], pos[dev3], dist14, dist24, dist34)


    fig, ax = plt.subplots(figsize=(6,6),num="Node positions")

    for p in pos.keys():
        # print(pos[p])
        plt.plot(pos[p][0], pos[p][1], label='', linestyle="None", marker='o', markersize=1, color='black', fillstyle='none')
        ax.annotate(p.replace("Dev-",""), size=8, xy=[pos[p][0], pos[p][1]], xytext=(1, 1), textcoords='offset points')
    ax.axis('equal')
    plt.pause(0)

def genx():
    sink = (0, 0)
    coordfixer = True
    nodes = nest.generate_nodes(100, 2500)
    if coordfixer:
        min = np.min([x[0] ** 2 + x[1] ** 2 for x in nodes])
        nodes = np.concatenate((nodes, np.array([((1 / 2) * np.sqrt(min), 0),
                                                 ((1 / 2) * np.sqrt(min), (1 / 2) * np.sqrt(min))], )), axis=0)
    maxrange = 1500
    sigma = 0.5
    tx_pow = 100
    dist = nest.distances(nodes)
    sinkdist = nest.sinkdistances(nodes, sink)
    x_nodes = nest.signals_errorize(dist, sigma, tx_pow)
    x_sink = nest.signals_errorize(sinkdist, sigma, tx_pow)
    return generate(x_sink, x_nodes)

def generate(x_sink, x_nodes):
    x = np.array([])
    for i, y in enumerate(x_sink):
        x = np.concatenate((x, [('sink', 'Dev'+str(i), y)]))
    for i, y in np.ndenumerate(x_nodes):
        x = np.concatenate((x, [('Dev'+str(i[0]), 'Dev'+str(i[1]), y)]))
    return x

def sigint_handler(signal, frame):
    print ('KeyboardInterrupt')
    sys.exit(0)

if __name__ == '__main__':
   signal.signal(signal.SIGINT, sigint_handler)
   main(sys.argv[1:])
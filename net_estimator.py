import numpy as np
import random as rd
import position_estimator as pe
import expect as exp

def main():
    sink = (0, 0)
    nodes = generate_nodes(100, 2500)
    maxrange = 750
    sigma = 1
    mu = 1
    beta = 1
    tx_pow = 100
    dist = distances(nodes)
    sinkdist = sinkdistances(nodes, sink)
    dist_err = errorize(dist, sigma, mu, beta, tx_pow)
    sinkdist_err = errorize(sinkdist, sigma, mu, beta, tx_pow)
    net = []
    netOrganize(net, [], nodes.shape[0], dist, maxrange, sinkdist)
    est = net_estimate(dist_err, sinkdist_err, maxrange)
    print(net)
    print(dist)
    print(dist_err)
    print(est)

def generate_nodes(n, radius):
    nodes = np.zeros((n, 2))
    for a in range(nodes.shape[0]):
        for b in range(nodes.shape[1]):
            nodes[a, b] = rd.randrange(-radius, radius)
    return nodes

def distances(nodes):
    distances = np.zeros((nodes.shape[0], nodes.shape[0]))
    for a in range(nodes.shape[0]):
        for b in range(nodes.shape[0]):
            r, theta = exp.cart_to_polar(nodes[a, 0] - nodes[b, 0], nodes[a, 1] - nodes[b, 1])
            distances[a, b] = r
    return distances

def sinkdistances(nodes, sink):
    sdist = np.zeros(nodes.shape[0])
    for a in range(nodes.shape[0]):
        r, theta = exp.cart_to_polar(nodes[a, 0] - sink[0], nodes[a, 1] - sink[1])
        sdist[a] = r
    return sdist

def errorize(dist, sigma, mu, beta, tx_pow):
    errored = np.zeros(dist.shape)
    for i, d in np.ndenumerate(dist):
        errored[i] = pe.distance_error_generate(d, sigma, tx_pow, mu, beta)
    return errored

def bestDistanceNode(bc, nodenr, dist, R):
    best = np.infty
    bestnode = -1
    for m in range(dist.shape[0]):
        if (m in bc) and (dist[nodenr, m] < min(best, R)):
            best = dist[nodenr, m]
            bestnode = m
    return bestnode

def netOrganize(net, beaconing, nrofnodes, distances, rge, sinkdistances):
    while not netFull(net, nrofnodes):
        net, beaconing = netRound(net, beaconing, distances, rge, sinkdistances)
        if beaconing == []:
            return
    return

def netRound(net, bc, dist, R, sinkdist):
    newbc = []
    for nodenr in range(dist.shape[0]):
        if not nodeIsInNet(net, nodenr) :
            bestnode = bestDistanceNode(bc, nodenr, dist, R)
            if bestnode != -1:
                net.append((nodenr, bestnode))
                newbc.append(nodenr)
            else:
                if sinkdist[nodenr] < R:
                    net.append((nodenr, -1))
                    newbc.append(nodenr)

    return net, newbc

def nodeIsInNet(net, n):
    found = False
    for e in net:
        if e[0] == n or e[1] == n:
            found = True
    return found

def netFull(net, nodes):
    found = np.zeros(nodes)
    for e in net:
        found[e[0]] = 1
        found[e[1]] = 1
    return (found == 1).all()

def netEstimateRound(est, ready, dist, sinkdist, detect, sinkdet):
    newest = copy.deepcopy(est)
    newready = copy.deepcopy(ready)
    for i in range(sinkdist.shape[0]):
        dist_i = []
        est_i = []
        for j in range(sinkdist.shape[0]):
            if (j in ready) and detect[j, i]:
                dist_i.append(dist[j, i])
                est_j_i.append(est[j])
        if sinkdet[i]:
            dist_i.append(sinkdist[i])
            est_i.append((0, 0))
        if len(dist_i_ix) >= 3:
            newest[i] = pe.position_estimate(est_i, dist_i)
            newready.append(i)
    est = newest
    ready = newready
    return

def detect(dist, sinkdist, rge):
    det = np.zeros(dist.shape)
    sinkdet = np.zeros(sinkdist.shape)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[0]):
            if dist[i, j] <= rge:
                det[i, j] = True
    for i in range(sinkdist.shape[0]):
        if sinkdist[i] <= rge:
            sinkdet[i] = True

def firstreference(sinkdist, sinkdet):
    min_ix = 0
    for i in range(sinkdist.shape[0]):
        if sinkdet[i] and sinkdist[i] < sinkdist[min_ix]:
            min_ix = i
    return min_ix

def secondreference(sinkdist, sinkdet, ref1):
    min_ix = 0
    for i in range(sinkdist.shape[0]):
        if sinkdet[i] and sinkdist[i] < sinkdist[min_ix] and i != ref1:
            min_ix = i
    return min_ix

def allReady(ready, num):
    for i in range(num):
        if i not in ready:
            return False
    return True

def net_estimate(dist_err, sinkdist_err, rge):
    det, sinkdet = detect(dist_err, sinkdist_err, rge)
    ref1, ref1_est = firstreference(sinkdist_err, sinkdet)
    ref2, ref2_est = secondreference(sinkdist_err, sinkdet, ref1)
    est = np.zeros(dist_err.shape)
    ready = []
    update(est, ready, [ref1, ref2], [ref1_est, ref2_est])
    while not allReady(ready, dist_err.shape[0]):
        netEstimateRound(est, ready, dist_err, sinkdist_err, det, sinkdet)
    return est


if __name__ == '__main__':
    main()

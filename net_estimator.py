import numpy as np
import random as rd
import position_estimator as pe
import expect as exp
import copy
import matplotlib.pyplot as plt

def main():
    sink = (0, 0)
    coordfixer = False
    half_netwidth = 1000
    nodes = generate_nodes(25, half_netwidth)
    if coordfixer:
        min = np.min([x[0]**2 + x[1]**2 for x in nodes])
        nodes = np.concatenate((nodes, np.array([((1/2)*np.sqrt(min), 0),
                                                 ((1/2)*np.sqrt(min), (1/2)*np.sqrt(min))], )), axis=0)
    maxrange = 5000
    sigma = 0.8
    tx_pow = 100
    iters = 5
    dist = distances(nodes)
    sinkdist = sinkdistances(nodes, sink)
    dist_err = errorize(dist, sigma, tx_pow)
    sinkdist_err = errorize(sinkdist, sigma, tx_pow)
    net = []
    netOrganize(net, [], nodes.shape[0], dist, maxrange, sinkdist)
    det, est, ref1, ref2 = net_estimate(dist_err, sinkdist_err, maxrange, iters=iters)
    fixed_est = fix_est(est, ref1, ref2, nodes[ref1], nodes[ref2])
    location_errors = get_location_errors(nodes, fixed_est)
    mean_location_error = np.average(location_errors)
    print("est :"+str(est))
    print("fixed_est: "+str(fixed_est) )
    print("original: "+str(nodes))
    print("error: "+str(fixed_est - nodes))

    plot_pos(nodes, fixed_est, mean_location_error, half_netwidth, detect=None)


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

def errorize(dist, sigma, tx_pow):
    errored = np.zeros(dist.shape)
    for i, d in np.ndenumerate(dist):
        errored[i] = pe.distance_error_generate(d, sigma, tx_pow)
    return errored

def signals_errorize(dist, sigma, tx_pow):
    sig = np.zeros(dist.shape)
    for i, d in np.ndenumerate(dist):
        sig[i] = pe.errored_signal_generate(d, sigma, tx_pow)
    return sig

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
        if i not in ready:
            dist_i = []
            est_i = []
            for j in range(sinkdist.shape[0]):
                if (j in ready) and detect[j, i]:
                    dist_i.append(dist[j, i])
                    est_i.append(est[j])
            if sinkdet[i]:
                dist_i.append(sinkdist[i])
                est_i.append((0, 0))
            dist_i = np.array(dist_i)
            est_i = np.array(est_i)
            if len(dist_i) >= 3:
                newest[i] = pe.position_estimate_like(est_i, dist_i)
                newready.append(i)
    est = newest
    nr_added = copy.deepcopy(newready)
    for x in ready:
        nr_added.remove(x)
    changed = False if nr_added == [] else True
    ready = newready
    return est, ready, changed

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
    return det, sinkdet


def firstreference(sinkdist, sinkdet):
    min_ix = 0
    for i in range(sinkdist.shape[0]):
        if sinkdet[i] and sinkdist[i] < sinkdist[min_ix]:
            min_ix = i
    return min_ix, (sinkdist[min_ix], 0)

def secondreference(dist, det, sinkdist, sinkdet, ref1, ref1_est):
    min_ix = -1
    nok = [ref1]
    found = False
    while len(nok) < len(sinkdist) and not found:
        for i in range(sinkdist.shape[0]):
            if sinkdet[i] and (sinkdist[i] < sinkdist[min_ix] or min_ix == -1) and i not in nok:
                min_ix = i
        if det[ref1, min_ix]:
            found = True
        else:
            nok.append(min_ix)
            min_ix = -1
    est = pe.bilaterate((0, 0), ref1_est, sinkdist[min_ix], dist[ref1, min_ix])
    est_final = est[0]
    if est[1][1] > 0:
        est_final = est[1]
    return min_ix, est_final

def update(est, ready, ixs, newest):
    for n, i in enumerate(ixs):
        ready.append(i)
        est[i] = newest[n]
    return


def allReady(ready, num):
    for i in range(num):
        if i not in ready:
            return False
    return True

def net_estimate(dist_err, sinkdist_err, rge, iters = 20):
    det, sinkdet = detect(dist_err, sinkdist_err, rge)
    ref1, ref1_est = firstreference(sinkdist_err, sinkdet)
    ref2, ref2_est = secondreference(dist_err, det, sinkdist_err, sinkdet, ref1, ref1_est)
    est = np.zeros((sinkdist_err.shape[0], 2))
    ready = []
    update(est, ready, [ref1, ref2], [ref1_est, ref2_est])
    changed = True
    #initialization
    while not allReady(ready, dist_err.shape[0]) and changed:
        est, ready, changed = netEstimateRound(est, ready, dist_err, sinkdist_err, det, sinkdet)
        print( "ready: "+str(ready))
    for i in range(iters):
        ready = []
        while not allReady(ready, dist_err.shape[0]) and changed:
            est, ready, changed = netEstimateRound(est, ready, dist_err, sinkdist_err, det, sinkdet)

    return det, est, ref1, ref2

def fix_est(est, ref1, ref2, node1, node2):
    _, rot = exp.cart_to_polar(node1[0], node1[1])
    righthand = np.zeros(est.shape)
    # fix the handedness
    theta = exp.cart_to_polar(node2[0], node2[1])[1] - exp.cart_to_polar(node1[0], node1[1])[1]
    if theta > np.pi:
        for i, x in enumerate(est):
            righthand[i][0] = x[0]
            righthand[i][1] = -x[1]
    else:
        for i, x in enumerate(est):
            righthand[i][0] = x[0]
            righthand[i][1] = x[1]
    # rotate to real ref1 position
    derot = np.zeros(est.shape)
    for i, x in enumerate(righthand):
        r, theta = exp.cart_to_polar(x[0], x[1])
        newx, newy = exp.polar_to_cart(r, theta + rot)
        derot[i] = (newx, newy)
    return derot

def get_location_errors(nodes, est):
    return np.sqrt((nodes[:,0]- est[:,0]) ** 2 + (nodes[:,1]- est[:,1]) ** 2)

def plot_pos(nodes, est, mean_location_error, halfwidth = 1000, detect = None):
    fig, ax = plt.subplots(figsize=(6,6),num="Node positions")
    colors = ['black','blue','red','green','brown','orange','gold','pink','cyan','lime']
    for ix, n in enumerate(nodes):
        # print(pos[p])
        plt.plot(n[0], n[1], label='', linestyle="None", marker='o', markersize=1, color='black', fillstyle='none')
        ax.annotate(text =str(ix), xy = [n[0], n[1]], color=colors[ix % len(colors)])


    fig_est, ax_est = plt.subplots(figsize=(6, 6), num="Estimated positions")
    for ix, n in enumerate(est):
        # print(pos[p])
        plt.plot(n[0], n[1], label='', linestyle="None", marker='o', markersize=1, color='black', fillstyle='none')
        ax_est.annotate(text =str(ix), xy = [n[0], n[1]], color=colors[ix % len(colors)])
        if detect is not None:
            for ix2, n2 in enumerate(est):
                if detect[ix, ix2]:
                    plt.plot([n[0], n2[0]], [n[1], n2[1]], label='', linestyle="--", linewidth=0.1 , marker='o', markersize=1,
                             color='black', fillstyle='none')
    ax.axis('equal')
    ax_est.axis('equal')

    text_str = "Mean location error: "+str(mean_location_error)+" meters"
    ax_est.set_xlabel(text_str)
    plt.pause(0)

if __name__ == '__main__':
    main()

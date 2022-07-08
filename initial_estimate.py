import numpy as np
import random as rd
import coordinates as co
import position_estimator as pe
import copy

import position_estimator


def initial_estimate_roundest, ready, dist, detect):
    newest = copy.deepcopy(est)
    newready = copy.deepcopy(ready)
    for i in range(dist.shape[0]):
        if i not in ready:
            dist_i = []
            est_i = []
            miss_i = []
            for j in range(dist.shape[0]):
                if (j in ready) and (j != i):
                    if detect[j, i]:
                        dist_i.append(dist[j, i])
                        est_i.append(est[j])
                    else:
                        miss_i.append(est[j])
            # if sinkdet[i]:
            #    dist_i.append(sinkdist[i])
            #    est_i.append((0, 0))
            dist_i = np.array(dist_i)
            est_i = np.array(est_i)
            miss_i = np.array(miss_i)
            if len(dist_i) >= 3:
                newest[i] = pe.position_estimate_like(est_i, dist_i, pos_miss=miss_i)
                newready.append(i)
    est = newest
    nr_added = copy.deepcopy(newready)
    for x in ready:
        nr_added.remove(x)
    changed = False if nr_added == [] else True
    ready = newready
    return est, ready, changed

def initial_estimator(dist, anchor_ix):
    coords_with_ref_j = np.zeros((dist.shape[0], dist.shape[0]))
    return

def initial_estimate_hopcount(dist, detect, anchor_locs, anchor_ixs):
    # number of hops to node i from anchor j in hopcounts[i, j]
    hopcounts = calculate_hopcounts(detect, anchor_ixs)
    est = np.zeros((detect.shape[0], 2), dtype=complex)
    for i in range(detect.shape[0]):
        min_anchors = anchor_ixs[util.find_min(hopcounts[i,:], 3)]
        anc1 = min_anchors[0]
        anc2 = min_anchors[1]
        anc3 = min_anchors[2]
        aix1 = anchor_ixs[anc1]
        aix2 = anchor_ixs[anc2]
        aix3 = anchor_ixs[anc3]
        loc1 = anchor_locs[anc1]
        loc2 = anchor_locs[anc2]
        loc3 = anchor_locs[anc3]
        ones = (hopcounts[i, anc1] == 1) + (hopcounts[i, anc2] == 1) + (hopcounts[i, anc3] == 1)
        if ones == 3:
            est[i] = pe.trilaterate_simple(loc1, loc2, loc3, dist[i, aix1], dist[i, aix2], dist[i, aix3])
        elif ones == 2:
            est[i] = update_est_2ones(dists, anchor_locs, anchor_ixs, hopcounts, min_anchors, i)
        elif ones == 1:
            est[i] = update_est_1one

def calculate_hopcounts(detect, anchor_ixs):
    hopcounts = np.zeros((detect.shape[0], anchor_ixs.shape[0])) - 1
    for n, ix in enumerate(anchor_ixs):
        hopcounts[ix, n] = 0
    hops = 0
    changed = True
    while (hopcounts == -1).any() and changed:
        changed = False
        for i in range(detect.shape[0]):
            for n in range(anchor_ixs.shape[0]):
                for j in range(detect.shape[0])
                    if hopcounts[i, n] == -1 and hopcounts[j,n] == hops and detect[i, j]:
                        hopcounts[i, n] = hops + 1
                        changed = True
        hops += 1
    return hopcounts

def update_est_2ones(dists, anchor_locs, anchor_ixs, hopcounts, min_anchors, i):
    anc1, anc2, anc3 = min_anchors
    anc_ones = [anc2, anc3]
    notone = anc1
    if hopcounts[i, anc1] == 1:
        anc_ones = [anc1, anc3]
        notone = anc2
        if hopcounts[i, anc2] == 1:
            anc_ones = [anc1, anc2]
            notone = anc3
    loco1, loco2 = anchor_locs[anc_ones]
    aixo1, aixo2 = anchor_ixs[anc_ones]
    ests = pe.bilaterate(loco1, loco2, dists[i, aixo1], dists[aixo2])
    locno = anchor_locs[notone]
    d_sq_1 = (ests[0, 0] - locno[0]) ** 2 + (ests[0, 1] - locno[1]) ** 2
    d_sq_2 = (ests[1, 0] - locno[0]) ** 2 + (ests[1, 1] - locno[1]) ** 2
    if d_sq_1 < d_sq_2:
        return ests[0]
    else:
        return ests[1]

def update_est_0ones(pos1, pos2, pos3, n1, n2, n3):
    arr = [[pos1, n1], [pos2, n2], [pos3, n3]]
    arr.sort(key= lambda x: x[1])
    arr = np.array(arr)
    p1, p2, p3 = arr[:,0]
    n1, n2, n3 = arr[:,1]

def a_nor(x1, x2, x3, n1, n2, n3):
    return 2 * (n2 ** 2 * x3 - n3 ** 2 * x2 - (n3 ** 2 - n2 ** 2)/(n2 ** 2 - n1 ** 2) * (n1 ** 2 * x2 -n2**2 * x1))

def a_exc(x1, x2, n1):
    return (n1**2 * (y - y2)**2 - n2**2*(y - y1)**2)/(2*(n1**2*x2 - n2**2*x1))




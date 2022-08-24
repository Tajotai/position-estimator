import numpy as np
import random as rd
import coordinates as co
import position_estimator as pe
import copy
import util

def initial_estimate_roundest(ready, dist, detect):
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

def initial_estimate_hopcount(dist, detect, anchor_locs, anchor_ixs, maxrange):
    # number of hops to node i from anchor j in hopcounts[i, j]
    hopcounts = calculate_hopcounts(detect)
    est = np.zeros((detect.shape[0], 2), dtype=complex)
    hopcounts_anc = hopcounts[:, anchor_ixs]
    for i in range(detect.shape[0]):
        min_ix = util.find_min(hopcounts_anc[i, :], 3)
        min_anchors = anchor_ixs[[int(min_ix[0]), int(min_ix[1]), int(min_ix[2])]]
        anc1 = min_anchors[0]
        anc2 = min_anchors[1]
        anc3 = min_anchors[2]
        aix1 = anchor_ixs[anc1]
        aix2 = anchor_ixs[anc2]
        aix3 = anchor_ixs[anc3]
        loc1 = anchor_locs[anc1]
        loc2 = anchor_locs[anc2]
        loc3 = anchor_locs[anc3]
        if i == int(min_ix[0]):
            est[i] = loc1
        else:
            dist1 = give_dist(dist, detect, hopcounts, i, aix1, maxrange)
            dist2 = give_dist(dist, detect, hopcounts, i, aix2, maxrange)
            dist3 = give_dist(dist, detect, hopcounts, i, aix3, maxrange)
            #ones = (hopcounts[i, anc1] == 1) + (hopcounts[i, anc2] == 1) + (hopcounts[i, anc3] == 1)
            #if ones == 3:
            #    est[i] = pe.trilaterate_simple(loc1, loc2, loc3, dist[i, aix1], dist[i, aix2], dist[i, aix3])
            #elif ones == 2:
            #    est[i] = update_est_2ones(dists, anchor_locs, anchor_ixs, hopcounts, min_anchors, i)
            #elif ones == 1:
            #    est[i] = update_est_1one
            pos_i = np.array([loc1, loc2, loc3])
            dist_i = np.array([dist1, dist2, dist3])
            est[i] = pe.position_estimate_like(pos_i, dist_i)
    return est, hopcounts

def calculate_hopcounts(detect):
    hopcounts = np.zeros((detect.shape[0], detect.shape[0])) - 1
    for i in range(detect.shape[0]):
        hopcounts[i, i] = 0
    hops = 0
    changed = True
    while (hopcounts == -1).any() and changed:
        changed = False
        for i in range(detect.shape[0]):
            for n in range(detect.shape[0]):
                if hopcounts[i, n] != -1:
                    continue
                for j in range(detect.shape[0]):
                    if hopcounts[j, n] == hops and detect[i, j]:
                        hopcounts[i, n] = hops + 1
                        changed = True
                        break
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

def give_dist(dist, det, hops, i, j, maxrange):
    if det[i, j]:
        return dist[i, j]
    else:
        n = hops[i, j]
        return approx_range(n, maxrange)

def update_est_1ones_aprx(pos1, pos2, pos3, n1, n2, n3, maxrange):
    r1 = approx_range(n1, maxrange)
    r2 = approx_range(n2, maxrange)
    r3 = approx_range(n3, maxrange)


def approx_range(n, maxrange):
    return (n - 0.5) * maxrange

def update_est_0ones_eq(pos1, pos2, pos3, n1, n2, n3):
    # a bit off
    arr = [[pos1, n1], [pos2, n2], [pos3, n3]]
    arr.sort(key= lambda x: x[1])
    arr = np.array(arr)
    (x1, y1), (x2, y2), (x3, y3) = arr[:,0]
    n1, n2, n3 = arr[:,1]
    nx12 = n2x_diff(x1, x2, n1, n2)
    ny21 = n2x_diff(y2, y1, n2, n1)
    nd12 = n2dist2_diff(x1, x2, y1, y2, n1, n2)
    nx23 = n2x_diff(x2, x3, n2, n3)
    ny32 = n2x_diff(y3, y2, n3, n2)
    nd32 = n2dist2_diff(x3, x2, y3, y2, n3, n2)
    if n1 != n2:
        n_rat = n_ratio(n1, n2, n3)
        a = a_nor(nx23, nx12, n_rat)
        b = b_nor(ny32, ny21, n_rat)
        c = c_nor(nd12, nd32, n_rat)
    else:
        a = 2 * nx12
        b = 2 * ny21
        c = nd12
    bpera = b / a
    cpera = c / a
    n32 = n_sqdiff(n3, n2)
    a_pri = a_prime(n32, bpera)
    b_pri = b_prime(n32, nx23, ny32, bpera, cpera)
    c_pri = c_prime(n32, cpera, nx23, nd32)
    if a_pri == 0:
        y_p = y_m = -c_pri / b_pri
    else:
        sq_disq = np.sqrt(b_pri ** 2 - 4 * a_pri * c_pri)
        y_p = (-b_pri + sq_disq) /(2 * a_pri)
        y_m = (-b_pri - sq_disq)/(2 * a_pri)
    x_p = (b * y_p + c)/ a
    x_m = (b * y_m + c)/ a
    pos_p = np.array([x_p, y_p])
    pos_m = np.array([x_m, y_m])
    d_p = total_sqdist(pos1, pos2, pos3, pos_p)
    d_m = total_sqdist(pos1, pos2, pos3, pos_m)
    if d_p < d_m:
        return pos_p
    else:
        return pos_m

def a_nor(nx23, nx12, n_rat):
    return 2 * (nx23 - n_rat * nx12)

def b_nor(ny32, ny21, n_rat):
    return 2 * (ny32 - n_rat * ny21)

def c_nor(nd12, nd32, n_rat):
    return nd32 + n_rat * nd12

def n_ratio(n_1, n_2, n_3):
    return (n_3 ** 2 - n_2 ** 2)/(n_2 ** 2 - n_1 ** 2)

def n2x_diff(x_1, x_2, n_1, n_2):
    return n_1 ** 2 * x_2 - n_2 ** 2 * x_1

def n2_x2ply2(x, y, n):
    return n**2 * (x ** 2 + y ** 2)

def n2dist2_diff(x1, x2, y1, y2, n1, n2):
    return n2_x2ply2(x1, y1, n2) - n2_x2ply2(x2, y2, n1)

def n_sqdiff(n1, n2):
    return n1 ** 2 - n2 ** 2

def a_prime(n32, bpera):
    return n32 * (bpera ** 2 + 1)

def b_prime(n32, nx23, ny32, bpera, cpera):
    return 2 * (n32 * bpera * cpera + nx23 * bpera - ny32)

def c_prime(n32, cpera, nx23, nd32):
    return n32 * cpera ** 2 + 2 * nx23 * cpera - nd32

def total_sqdist(p1, p2, p3, p):
    sqd1 = (p1[0] - p[0]) ** 2 + (p1[1] - p[1]) ** 2
    sqd2 = (p2[0] - p[0]) ** 2 + (p2[1] - p[1]) ** 2
    sqd3 = (p3[0] - p[0]) ** 2 + (p3[1] - p[1]) ** 2
    return sqd1 + sqd2 + sqd3
import numpy as np
import random as rd
import position_estimator as pe
import coordinates as co
import fix_est as fe
import copy
import matplotlib.markers as mrks
import matplotlib.pyplot as plt
import initial_estimate as ie

def main(index, sink_ix, coordfixer, static, glob, half_netwidth, nr_of_nodes, nr_of_anchors, maxrange, sigma, tx_pow, iters):
    #rd.seed(432789)
    nodes = generate_nodes(nr_of_nodes, half_netwidth)
    if coordfixer:
        min = np.min([x[0]**2 + x[1]**2 for x in nodes])
        nodes = np.concatenate((nodes, np.array([((1/2)*np.sqrt(min), 0),
                                                 ((1/2)*np.sqrt(min), (1/2)*np.sqrt(min))], )), axis=0)
    has_anchors = bool(nr_of_anchors)
    if static:
        #nodes = np.array([[-712., -777.],[ 533 ,  972.],[-699., -774.],[ 702.,  661.],[-335.,  994.],[ 366., -829.],
        #                  [ 930., -533.],[ 382.,  518.],[-435., -566.],[ 423., -930.],[ 827.,  332.],[ 804.,  722.],
        #                  [-328.,  987.],[-388.,  601.],[-524., -551.],[ 978.,  538.],[ 794., -251.],[-498., -907.],
        #                  [ 101.,  658.], [ 586.,  435.],[-281.,  526.],[ 582., -730.],[-547.,   29.],[ 445., -507.],
        #                  [-578.,  501.],[ 273.88409957,    0.],[ 273.88409957,  273.88409957]])
        nodes = np.array([[ 473,-947.],
             [ 142,-835.],[ 875, 189.],[ 820, 748.],[-762, 331.],[ 125,-713.],[ 676, 262.],[ 229, -92.],
             [ 651,-977.],[-698,-551.],[ 868,-809.],[  70, 654.],[-952, 562.],[  29,-149.],
             [ 365, 528.],[ 915,  92.],[ 126, -41.],[-288, 987.],[-488, 577.],[ 172,-894.],
             [ -24,-613.],[ 454,-101.],[ 107,-776.],[-576,-550.],[-811, 605.],[-496,-700.],
             [-225,-431.],[ 946,-190.],[ 132, 885.],[-347, 596.],[ 305,-573.],[ 148, 702.],
             [ 861, 475.],[-123,-368.],[ 309, 711.],[-603,-850.],[  49,-489.],[ 523,-908.],
             [-414,-679.],[-661, 332.],[ 544, 996.],[ 641,-906.],[-450, 308.],[-800,-958.],
             [-661,-444.],[-430, 999.],[-895,-884.],[-162,-203.],[-209, 307.],[-192,-524.]])
    anchors = []
    anchor_locs = []
    if has_anchors:
        anchors = np.arange(nr_of_anchors)
        anchor_locs = nodes[anchors]
    dist = distances(nodes)
    # print(dist[30:40, 30:40])
    # sinkdist = sinkdistances(nodes, sink)
    dist_err = errorize(dist, sigma, tx_pow)
    # print(dist_err[30:40, 30:40])
    # sinkdist_err = errorize(sinkdist, sigma, tx_pow)
    # totaldist = np.concatenate((dist, sinkdist))
    # totaldist_err = np.concatenate((dist_err, sinkdist_err))
    net = []
    # netOrganize(net, [], nodes.shape[0], dist, maxrange, sinkdist)

    if glob:
        det, est, ref1, ref2 = net_estimate_global(dist_err, maxrange, anchor_locs, nr_of_anchors)
    else:
        det, est, ref1, ref2 = net_estimate(dist_err, maxrange, iters, half_netwidth, nodes=nodes, anchors=anchors,
                                            anchor_locs=anchor_locs)
    if est is None:
        return 0, nr_of_nodes - nr_of_anchors
    #det, est = net_estimate_global()
    fixed_est = est
    if not has_anchors:
        fixed_est = fe.fix_est4(est, nodes, ref1, ref2)
    location_errors = fe.get_location_errors(nodes, fixed_est)
    not_classified = np.isnan(location_errors)
    classified = 1 - not_classified
    print(classified)
    nr_of_classified = sum(classified)
    nr_of_notclassified = sum(not_classified)
    class_loc_errors = []
    for n in range(nodes.shape[0]):
        if classified[n] and n not in anchors:
            class_loc_errors.append(location_errors[n])
    class_loc_errors = np.array(class_loc_errors)
    mean_location_error = np.average(class_loc_errors)

    #print final log-likelihood
    det, _ = detect(dist_err, np.zeros(0), maxrange)
    losses = pe.calculate_path_loss(dist_err)
    like_final = pe.globlike(nr_of_anchors, fixed_est[:, 0], fixed_est[:, 1], losses, det)
    print("Likelihood est: "+str(np.real(like_final)))
    like_real = pe.globlike(nr_of_anchors, nodes[:, 0], nodes[:, 1], losses, det)
    print("Likelihood real: "+str(np.real(like_real)))
    #print("est :"+str(est))
    #print("fixed_est: "+str(fixed_est))
    #print("original: "+str(nodes))
    #print("error: "+str(fixed_est - nodes))

    #plot_pos(nodes, fixed_est, mean_location_error, half_netwidth, detect=None, anchors=anchors, mode='save', index=index)
    return mean_location_error, nr_of_notclassified

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
            r, theta = co.cart_to_polar(nodes[a, 0] - nodes[b, 0], nodes[a, 1] - nodes[b, 1])
            distances[a, b] = r
    return distances

def sinkdistances(nodes, sink):
    sdist = np.zeros(nodes.shape[0])
    for a in range(nodes.shape[0]):
        r, theta = co.cart_to_polar(nodes[a, 0] - sink[0], nodes[a, 1] - sink[1])
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

def netEstimateRound(loc_est, params, ready, dist, detect, sinkdist=None, sinkdet=None, initial=False, anchors=[], tempr=5):
    newloc_est = copy.deepcopy(loc_est)
    sigmasq_est, P0_est, gamma_est, sigmasq_sigma, P0_sigma, gamma_sigma = \
        params[0], params[1], params[2], params[3], params[4], params[5]
    newready = copy.deepcopy(ready)
    sigmasq_raw = np.zeros(loc_est.shape[0])
    P0_raw = np.zeros(loc_est.shape[0])
    gamma_raw = np.zeros(loc_est.shape[0])
    for i in range(dist.shape[0]):
        if i not in ready:
            dist_i = []
            loc_est_i = []
            miss_i = []
            for j in range(dist.shape[0]):
                if (j in ready or not initial) and (j != i):
                    if detect[j, i]:
                        dist_i.append(dist[j, i])
                        loc_est_i.append(loc_est[j])
                    else:
                        miss_i.append(loc_est[j])
            # if sinkdet[i]:
            #    dist_i.append(sinkdist[i])
            #    loc_est_i.append((0, 0))
            dist_i = np.array(dist_i)
            loc_est_i = np.array(loc_est_i)
            miss_i = np.array(miss_i)
            if len(dist_i) >= 3:
                if i not in anchors:
                    params_i = np.array([sigmasq_est[i], P0_est[i], gamma_est[i],
                                         sigmasq_sigma[i], P0_sigma[i], gamma_sigma[i]])
                    newloc_est[i], sigmasq_raw[i], P0_raw[i], gamma_raw[i] = \
                        pe.position_estimate_like(loc_est_i, dist_i, pos_miss=miss_i), 0, 0, 0
                    # newloc_est[i], sigmasq_raw[i], P0_raw[i], gamma_raw[i] = \
                    #    pe.position_estimate_like2(loc_est_i, dist_i, params=params_i, pos_miss=miss_i)
                    #newloc_est[i] = pe.position_estimate_like_metro(loc_est_i, dist_i, tempr, est[i])
                newready.append(i)
    loc_est = newloc_est
    sigmasq_est, sigmasq_sigma = update_param(detect, sigmasq_raw, anchors)
    P0_est, P0_sigma = update_param(detect, P0_raw, anchors)
    gamma_est, gamma_sigma = update_param(detect, gamma_raw, anchors)
    params_estsigma = np.array([sigmasq_est, P0_est, gamma_est, sigmasq_sigma, P0_sigma, gamma_sigma])
    nr_added = copy.deepcopy(newready)
    for x in ready:
        nr_added.remove(x)
    changed = False if nr_added == [] else True
    ready = newready
    return loc_est, ready, changed, params_estsigma

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

def firstreference(sinkdist, sinkdet, sink_ix=0):
    min_ix = 0
    if sink_ix == 0:
        min_ix = 1
    for i in range(sinkdist.shape[0]):
        if sinkdet[i] and sinkdist[i] < sinkdist[min_ix] and i != sink_ix:
            min_ix = i
    return min_ix, (sinkdist[min_ix], 0)

def secondreference(dist, det, sinkdist, sinkdet, ref1, ref1_est, sink_ix = 0):
    min_ix = -1
    nok = [ref1, sink_ix]
    found = False
    while len(nok) < len(sinkdist) and not found:
        for i in range(sinkdist.shape[0]):
            if sinkdet[i] and (sinkdist[i] < sinkdist[min_ix] or min_ix == -1) and i not in nok:
                min_ix = i
        if det[ref1, min_ix]:
            found = True
        elif min_ix == -1:
            found = False
            return -1, [0,0]
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

def update_param(detect, raw, anchors):
    est = np.zeros(raw.shape[0])
    est_sigmasq = np.zeros(raw.shape[0])
    for i in range(detect.shape[0]):
        sum = 0
        sum_sq = 0
        counter = 0
        for j in range(detect.shape[0]):
            if (detect[i,j] or i==j) and j not in anchors:
                sum += raw[j]
                sum_sq += raw[j] ** 2
                counter += 1
        est[i] = sum / counter
        est_sigmasq[i] = (sum_sq - (1 / counter) * sum ** 2)/(counter - 1)
    return est, est_sigmasq

def allReady(ready, num, required=None):
    if required == None:
        required = range(num)
    for i in required:
        if i not in ready:
            return False
    return True

def net_estimate(dist_err, rge, iters, half_nw, sinkdist_err = None, nodes=None, sink_ix = 0, anchors=[], anchor_locs=None):
    if sinkdist_err == None:
        sinkdist_err = dist_err[sink_ix, :]

    det, sinkdet = detect(dist_err, sinkdist_err, rge)
    ref1, ref1_est = firstreference(sinkdist_err, sinkdet)
    ref2, ref2_est = secondreference(dist_err, det, sinkdist_err, sinkdet, ref1, ref1_est)
    if ref2 == -1:
        return None, None, None, None
    est = np.zeros((sinkdist_err.shape[0], 2))
    ready = []
    update(est, ready, [sink_ix, ref1, ref2], [(0, 0), ref1_est, ref2_est])
    changed = True
    #initialization
    est, hopcounts = ie.initial_estimate_hopcount(dist_err, det, anchor_locs, anchors, rge)
    dist_all = copy.deepcopy(dist_err)
    for i in range(dist_err.shape[0]):
        for j in range(dist_err.shape[0]):
            if not det[i, j]:
                dist_all[i, j] = ie.approx_range(hopcounts[i, j], rge)
    det_full = np.zeros(det.shape) + 1
    t0 = 50
    tempr = t0
    params_estsigma = np.zeros((6, dist_err.shape[0]))
    params_estsigma[3:,:] = np.inf * np.ones((3, dist_err.shape[0]))
    for i in range(iters - 1):
        prev_est = copy.deepcopy(est)
        iter_ready = []
        while not allReady(iter_ready, dist_err.shape[0], required=ready):
            # est, iter_ready, changed = netEstimateRound(est, iter_ready, dist_err, det, sinkdist_err, sinkdet)
            est, iter_ready, changed, params_estsigma = \
                netEstimateRound(est, params_estsigma, iter_ready, dist_err, det_full, anchors=anchors, tempr=tempr)
        if nodes is not None:
            #fixed_est = fe.fix_est(est, ref1, ref2, nodes[ref1], nodes[ref2])
            location_errors = fe.get_location_errors(nodes, est)
            mean_location_error = np.average(location_errors) * len(nodes)/(len(nodes) + len(anchors))
            #plot_pos(nodes, est, mean_location_error, half_nw, anchors=anchors, mode='save', index=i)
        est_diff = est - prev_est
        mean_diff = np.average(fe.get_location_errors(est_diff, np.zeros(est_diff.shape)))
        #print("Estimation diff "+str(i)+" "+str(mean_diff))
        #tempr *= (1 - 0.0005)
        tempr = t0 * ((iters - i)/iters) ** 4
    #print("sigma_est", params_estsigma[0, :])
    #print("P0_est", params_estsigma[1, :])
    #print("gamma_est", params_estsigma[2, :])
    return det, est, ref1, ref2

def net_estimate_global(dist_err, rge, anchor_locs, n_anc, init_est = None):
    sinkdist_err = dist_err[0, :]
    det, sinkdet = detect(dist_err, sinkdist_err, rge)
    ref1, ref1_est = firstreference(sinkdist_err, sinkdet)
    ref2, ref2_est = secondreference(dist_err, det, sinkdist_err, sinkdet, ref1, ref1_est)
    if init_est is None:
        init_est, hopcounts = ie.initial_estimate_hopcount(dist_err, det, anchor_locs, np.arange(n_anc), rge)
    est = pe.position_estimate_global(det, dist_err, n_anc, anchor_locs)
    return det, est, ref1, ref2

def plot_pos(nodes, est, mean_location_error, halfwidth = 1000, detect = None, anchors = None, mode = 'show', index=0):
    fig_i = plt.figure(index)
    fig, ax = plt.subplots(figsize=(6,6),num="Node positions")
    colors = ['black','blue','red','green','brown','orange','gold','pink','cyan','lime']
    mark = mrks.MarkerStyle('o', 'full')
    for ix, n in enumerate(nodes):
        # print(pos[p])
        #temp
        #if ix < 30 or ix > 40:
        #    continue
        col = 'black'
        if ix in anchors:
            col = 'red'
        plt.plot(n[0], n[1], label='', linestyle="None", marker=mark, markersize=3, color=col, fillstyle='none')
        ax.annotate(text =str(ix), xy = [n[0], n[1]], color=colors[ix % len(colors)])
    filename = 'plots\\node_positions'+str(index)
    plt.savefig(filename)
    plt.close()

    fig_est, ax_est = plt.subplots(figsize=(6, 6), num="Estimated positions")
    for ix, n in enumerate(est):
        # print(pos[p])
        #if ix < 30 or ix > 40:
        #    continue
        col = 'black'
        if ix in anchors:
            col = 'red'
        plt.plot(n[0], n[1], label='', linestyle="None", marker=mark, markersize=3, color=col, fillstyle='none')
        ax_est.annotate(text=str(ix), xy=[n[0], n[1]], color=colors[ix % len(colors)])
        if detect is not None:
            for ix2, n2 in enumerate(est):
                if detect[ix, ix2]:
                    plt.plot([n[0], n2[0]], [n[1], n2[1]], label='', linestyle="--", linewidth=0.1 , marker='o', markersize=1,
                             color='black', fillstyle='none')
    ax.axis('equal')
    ax_est.axis('equal')

    text_str = "Mean location error: "+str(mean_location_error)+" meters"
    ax_est.set_xlabel(text_str)
    if mode == 'show':
        plt.show()
        pass
    elif mode=='save':
        filename = 'plots\\posestplot'+str(index)
        #plt.savefig(filename)
        #plt.close()

if __name__ == '__main__':
    iters = 5
    scale_of_iters = 10
    text_file_name = "position_estimator_results"

    sink_ix = 0
    coordfixer = False
    static = False
    glob = False
    tx_pow = 100

    half_netwidth = [1000]
    nr_of_nodes = [200]
    # terms of % of nodes
    nr_of_anchors = [10]
    maxrange = [1000]
    sigma = [10]

    estimator_iters = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    counter = 0
    bigcounter = 0
    for hn in half_netwidth:
        for nn in nr_of_nodes:
            for na in nr_of_anchors:
                for mr in maxrange:
                    for sig in sigma:
                        for it in estimator_iters:
                            errors = []
                            notclassified = []
                            iters_of_nodenr = 500 // nn
                            iters = iters_of_nodenr * scale_of_iters
                            s = 0
                            n = 0
                            for i in range(iters):
                                err, nc = main(counter, sink_ix, coordfixer, static, glob, hn, nn, nn*na//100, mr, sig, tx_pow, it)
                                errors.append(err)
                                s += err * (nn - nc)
                                n += nn - nc
                                notclassified.append(nc)
                                print("iter ",i," done")
                                counter += 1
                                bigcounter += 1
                            notclassified = np.array(notclassified)
                            avr_error = s / n
                            avr_nc = sum(notclassified)/len(notclassified)
                            print(bigcounter," done ")
                            with open('results.txt', 'a') as f:
                                f.write(" Half netwidth "+str(hn)+" Nodes: "+str(nn) +"Anchors: "+str(na)+" Maximum range "+str(mr)+" sigma "+
                                        str(sig)+" Mean location error "+str(avr_error)+
                                        " Average nonclassified "+str(avr_nc)+' \n ')

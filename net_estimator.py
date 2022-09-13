import numpy as np
import random as rd
import position_estimator as pe
import coordinates as co
import fix_est as fe
import copy
import matplotlib.markers as mrks
import matplotlib.pyplot as plt
import initial_estimate as ie

def main(args=None):
    if args is None:
        sink_ix = 0
        coordfixer = False
        static = False
        glob = True
        half_netwidth = 1000
        nr_of_nodes = 50
        nr_of_anchors = 6
        maxrange = 750
        sigma = 5
        tx_pow = 100
        iters = 10
    else:
        sink_ix = args[0]
        coordfixer = args[1]
        static = args[2]
        glob = True
        half_netwidth = args[3]
        nr_of_nodes = args[4]
        nr_of_anchors = args[5]
        maxrange = args[6]
        sigma = args[7]
        tx_pow = args[8]
        iters = args[9]
        plot_interval = args[10]
    rd.seed(432789)
    nodes = generate_nodes(nr_of_nodes, half_netwidth)
    if coordfixer:
        min = np.min([x[0]**2 + x[1]**2 for x in nodes])
        nodes = np.concatenate((nodes, np.array([((1/2)*np.sqrt(min), 0),
                                                 ((1/2)*np.sqrt(min), (1/2)*np.sqrt(min))], )), axis=0)
    has_anchors = bool(nr_of_anchors)
    anchors = []
    anchor_locs=[]
    if has_anchors:
        anchors = np.arange(nr_of_anchors)
        anchor_locs = nodes[anchors]
    if static:
        #nodes = np.array([[-712., -777.],[ 533.  ,  972.],[-699., -774.],[ 702.,  661.],[-335.,  994.],[ 366., -829.],
        #                  [ 930., -533.],[ 382.,  518.],[-435., -566.],[ 423., -930.],[ 827.,  332.],[ 804.,  722.],
        #                  [-328.,  987.],[-388.,  601.],[-524., -551.],[ 978.,  538.],[ 794., -251.],[-498., -907.],
        #                  [ 101.,  658.], [ 586.,  435.],[-281.,  526.],[ 582., -730.],[-547.,   29.],[ 445., -507.],
        #                  [-578.,  501.],[ 273.88409957,    0.],[ 273.88409957,  273.88409957]])
        nodes = np.array([[-547., -809.],[-217., -771.],[472., -464.], [-206.,  575.],[169.,  482.],[-150., -323.],
                          [375., -449.],[ 61.,  614.],[848., -928.], [-5.,  500.], [-989., -785.], [-978., -343.],
                          [952.,  270.], [793., -639.], [679., -271.], [-917., -178.], [-929.,  861.],[744., -756.],
                            [-602.,   60.], [1.,  120.], [-192., -355.], [226., -200.], [443.,  888.],
                         [-41., -861.], [327., -652.], [267.,  534.], [-797.,  694.], [612., -461.], [-206., -238.],
                         [-358.,   28.], [637., -620.], [761., -691.], [-638.,  320.], [-770., -582.], [-622.,  585.],
                         [744.,  481.], [476.,  279.], [215.,  761.], [819., -874.], [-998.,  866.], [-767., -444.],
                         [-532.,  101.], [-638.,  997.], [473., -717.], [-940., -954.], [972.,  711.], [473.,  195.],
                         [-77.,  329.], [-418.,  363.], [529.,  769.]])
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
    #det, est = net_estimate_global()
    fixed_est = est
    if not has_anchors:
        fixed_est = fe.fix_est4(est, nodes, ref1, ref2)
    location_errors = fe.get_location_errors(nodes, fixed_est)
    mean_location_error = np.average(location_errors)*nr_of_nodes/(nr_of_nodes + nr_of_anchors)

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

    plot_pos(nodes, fixed_est, mean_location_error, half_netwidth, detect=None, anchors=anchors, mode='save')


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

def netEstimateRound(est, ready, dist, detect, sinkdist=None, sinkdet=None, initial=False, anchors=[]):
    newest = copy.deepcopy(est)
    newready = copy.deepcopy(ready)
    for i in range(dist.shape[0]):
        if i not in ready:
            dist_i = []
            est_i = []
            miss_i = []
            for j in range(dist.shape[0]):
                if (j in ready or not initial) and (j != i):
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
                if i not in anchors:
                    newest[i] = pe.position_estimate_like(est_i, dist_i, pos_miss=miss_i)
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
    for i in range(iters - 1):
        if i%5 == 0:
            print("iteration number:"+str(i))
        prev_est = copy.deepcopy(est)
        iter_ready = []
        while not allReady(iter_ready, dist_err.shape[0], required=ready):
            # est, iter_ready, changed = netEstimateRound(est, iter_ready, dist_err, det, sinkdist_err, sinkdet)
            est, iter_ready, changed = netEstimateRound(est, iter_ready, dist_err, det_full, anchors=anchors)
        if nodes is not None:
            #fixed_est = fe.fix_est(est, ref1, ref2, nodes[ref1], nodes[ref2])
            location_errors = fe.get_location_errors(nodes, est)
            mean_location_error = np.average(location_errors) * len(nodes)/(len(nodes) + len(anchors))
            #plot_pos(nodes, est, mean_location_error, half_nw, anchors=anchors, mode='save', index=i)
        est_diff = est - prev_est
        mean_diff = np.average(fe.get_location_errors(est_diff, np.zeros(est_diff.shape)))
        print("Estimation diff "+str(i)+" "+str(mean_diff))
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
    if index == 0:
        filename = 'node_positions'
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
    if mode == 'show':
        plt.show()
        pass
    elif mode=='save':
        filename = 'posestplot'+str(index)
        plt.savefig(filename)
        plt.close()

if __name__ == '__main__':
    main()

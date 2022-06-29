import numpy as np
import random as rd
import expect as exp
import maximize as mx

def distance_bias_multiplier(sigma, gamma):
    return 10 ** (np.log(10) * (sigma ** 2)/(200 * (gamma ** 2)))

def calculate_path_loss(dist):
    return 2 * 10 * (np.log10(dist) + np.log10((4 * np.pi) / 0.0999308193333333))

def distance_error_generate(r, sigma, tx_pow, bias=True):
    if r == 0:
        return 0
    powdiff = calculate_path_loss(r)
    # powdiff = gamma*10*np.log10(dis/dis_0)
    powerror = np.random.normal(0, sigma)
    if bias:
        r_err = calculate_distance(tx_pow - powdiff + powerror, tx_pow)
    else:
        r_err = nonbiased_calculate_distance(tx_pow - powdiff + powerror, tx_pow, mu, beta, sigma)
    return r_err

def errored_signal_generate(r, sigma, tx_pow):
    if r == 0:
        return 0
    powdiff = 2 * 10 * (np.log10(r) + np.log10(0.0999308193333333 / (4 * np.pi)))
    # powdiff = gamma*10*np.log10(dis/dis_0)
    powerror = np.random.normal(0, sigma)
    return powdiff + powerror

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

def calculate_distance_2(rx_pow, tx_pow, mu, beta, gain_tx, gain_rx):
    '''
    gives the distance by transmission and receiving power of the signal,
    using the Friis model for path loss in free space. The constant
    173.491005787 is the wavelength of signal with frequency 1.728 MHz.
    Antenna gains are accounted for.

    :param rx_pow: receiver power in dB
    :param tx_pow: transmitter power in dB
    :return: distance in meters
    '''
    return 10 ** ((tx_pow - rx_pow + gain_tx + gain_rx)/20 + np.log10(0.0999308193333333 / 4*np.pi*mu * beta))

def calculate_distance_with_ref(rx_pow, tx_pow, mu, beta):
    '''
    gives the distance by transmission and receiving power of the signal,
    using general log-distance path loss model with reference distance of
    1 meter. Reference distance path loss has to be determined in its
    own function, ref_dist_pl(). Free space is assumed, thus the path loss exponent is 2.

    :param rx_pow: receiver power in dB
    :param tx_pow: transmitter power in dB
    :return: distance in meters
    '''
    return 1 * 10 ** ((tx_pow - rx_pow - ref_dist_pl(mu, beta)) / 20)

def ref_dist_pl(mu, beta):
    return -22.8013420153 + 20 * np.log10(mu*beta)

def nonbiased_calculate_distance(rx_pow, tx_pow, mu, beta, sigma):
    '''
    Gives the nonbiased distance estimate provided that the multipath propagation
    variable standard deviation sigma is known.
    :param rx_pow:
    :param tx_pow:
    :param mu:
    :param beta:
    :return:
    '''
    return calculate_distance(rx_pow, tx_pow, mu, beta) / distance_bias_multiplier(sigma, 2)

def update_distance_simple(dist, n, newpow):
    return

def bilaterate(pos1, pos2, d1, d2):
    '''
    Gives two alternative
    :param pos1:
    :param pos2:
    :param d1:
    :param d2:
    :return:
    '''
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
    return np.array([(est1[0] + est2[0] + est3[0]) / 3, (est1[1] + est2[1] + est3[1]) / 3])

def position_estimate(pos, dist):
    length = len(pos)
    if length < 3:
        if length == 2:
            return position_estimate_two(pos[0], pos[1], dist[0], dist[1])
        if length == 1:
            return position_estimate_one(pos[0], dist[0])
        else:
            #length == 0
            return (0, 0, 0)
    else:
        if length > 6:
            min_ix = find_min(dist, 6)
            min_ix = np.array(min_ix, dtype=int)
            trimpos = pos[min_ix]
            trimdist = dist[min_ix]
        else:
            trimpos = pos
            trimdist = dist
        trila = []
        for i in range(len(trimpos)):
            for j in range(i + 1, len(trimpos)):
                for k in range(j + 1, len(trimpos)):
                    trila.append(trilaterate_simple(trimpos[i], trimpos[j], trimpos[k], trimdist[i], trimdist[j], trimdist[k]))
        trila_np = np.array(trila)
        return np.average(trila_np, axis=0)

def position_estimate_two(p0, p1, d0, d1):
    est0, est1 = bilaterate(p0, p1, d0, d1)
    r0, _ = exp.cart_to_polar(p0[0], p0[1])
    r1, _ = exp.cart_to_polar(p1[0], p1[1])
    if r1 > r0:
        return est1
    else:
        return est0

def position_estimate_one(pos, dist):
    r, theta = exp.cart_to_polar(pos)
    return ((r + dist) / r) * pos

def position_estimate_like(pos_det, dist, pos_miss=None, maxrange = np.inf):
    losses = calculate_path_loss(dist)
    xi = pos_det[:, 0]
    yi = pos_det[:, 1]
    xi_miss = None
    yi_miss = None
    if pos_miss is not None and pos_miss.shape[0] != 0:
        xi_miss = pos_miss[:, 0]
        yi_miss = pos_miss[:, 1]
    like_ = lambda x: like(x[0], x[1], losses, xi, yi, xi_miss, yi_miss, maxrange)
    ddx_like = lambda x: (like_(np.array([x[0] + 0.0001, x[1]])) - like_(x)) * 10000
    ddy_like = lambda x: (like_(np.array([x[0], x[1] + 0.0001])) - like_(x)) * 10000
    init = pe_like_initialize(pos_det, dist)
    xy_max, _ = mx.maximize_conjugate_gradient(like_, 2, [ddx_like, ddy_like], init, iters=15,
                                         onedimiters=5, onedimigap=500)
    return xy_max

def loss(x1, y1, x2, y2):
    d = np.sqrt((x2 - x1) ** 2+ (y2 - y1)**2)
    return calculate_path_loss(d)

def like(x, y, losses, xi, yi, x_miss=None, y_miss=None, maxrange = np.inf):
    det = -np.sum(((losses - loss(x, y, xi, yi))**2)/(20/ np.log(10)))
    nondet = 0 if (x_miss is None) else np.sum(logPhi_approx(x, y, x_miss, y_miss, maxrange))
    return det + nondet

def logPhi_approx(x, y, x_m, y_m, maxrange, sigma=1):
    pathloss = loss(x, y, x_m, y_m)
    thresh_loss = calculate_path_loss(maxrange)
    return 2 * np.min(np.array([(thresh_loss - pathloss)/np.sqrt(np.pi), np.zeros(x_m.shape[0]) + np.log(2)]), axis=0)

def find_min(array, number_of_mins):
    min_ix = np.zeros(number_of_mins) - 1
    items = np.zeros(number_of_mins)
    for i in range(len(array)):
        item = array[i]
        if i == 0:
            items[0] = item
            min_ix[0] = 0
            continue
        maxj = min(i - 1, number_of_mins - 1)
        for j in range(maxj, -1, -1):
            if item < items[j] or min_ix[j] == -1:
                if j < number_of_mins - 1:
                    min_ix[j + 1] = min_ix[j]
                    items[j + 1] = items[j]
                    if j == 0:
                        min_ix[0] = i
                        items[0] = item
            else:
                if j < number_of_mins - 1:
                    min_ix[j + 1] = i
                    items[j + 1] = item
                break
    return min_ix

def pe_like_initialize(pos, dist):
    # Finds the three nodes with smallest distances and trilaterates through them
    min_ix = find_min(dist, 3)
    min_ix1 = int(min_ix[0])
    min_ix2 = int(min_ix[1])
    min_ix3 = int(min_ix[2])
    return trilaterate_simple(pos[min_ix1], pos[min_ix2], pos[min_ix3], dist[min_ix1], dist[min_ix2], dist[min_ix3])



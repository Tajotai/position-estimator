import numpy as np
import random as rd
import expect as exp
def distance_bias_multiplier(sigma, gamma):
    return 10 ** (np.log(10) * (sigma ** 2)/(200 * (gamma ** 2)))

def distance_error_generate(r, sigma, tx_pow, mu, beta, bias=True):
    powdiff = 2*10*np.log10(r) + np.log10(4*np.pi*mu*beta/173.491005787)
    # powdiff = gamma*10*np.log10(dis/dis_0)
    powerror = np.random.normal(0, sigma)
    if bias:
        r_err = calculate_distance(tx_pow - powdiff + powerror, tx_pow, mu, beta)
    else:
        r_err = nonbiased_calculate_distance(tx_pow - powdiff + powerror, tx_pow, mu, beta, sigma)
    return r_err

def calculate_distance(rx_pow, tx_pow, mu, beta):
    '''
    gives the distance by transmission and receiving power of the signal,
    using the Friis model for path loss in free space. The constant
    173.491005787 is the wavelength of signal with frequency 1.728 MHz

    :param rx_pow: receiver power in dB
    :param tx_pow: transmitter power in dB
    :return: distance in meters
    '''
    return 10 ** ((tx_pow - rx_pow)/20 - np.log10(173.491005787 /(4 * np.pi * mu * beta)))

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
    return 10 ** ((tx_pow - rx_pow + gain_tx + gain_rx)/20 + np.log10(173.491005787 / 4*np.pi*mu * beta))

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

def te_differentiate(pos1, pos2, d1, d2):
    # something's wrong
    x1 = pos1[0]
    x2 = pos2[0]
    y1 = pos1[1]
    y2 = pos2[1]
    d3 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    c = (d1**2 - d2**2 + (x1-x2)**2 + (y1-y2)**2)/(2*d1*d3)
    p = d1 * c
    h = np.sqrt(1 - c**2)*d1

    dcdx1 = (x1 - x2) / (2 * d1 * d3) - c * d1 * (x1 - x2) / (2 * d3 ** 2)
    dcdx2 = (x2 - x1) / (2 * d1 * d3) - c * d1 * (x2 - x1) / (2 * d3 ** 2)
    dcdy1 = (y1 - y2) / (2 * d1 * d3) - c * d1 * (y1 - y2) / (2 * d3 ** 2)
    dcdy2 = (y2 - y1) / (2 * d1 * d3) - c * d1 * (x2 - x1) / (2 * d3 ** 2)
    dcdd1 = (-d1 - d3 * c) / (d1 * d3)
    dcdd2 = - d2 / (d1 * d3)

    first_dxdx1 = 1 - p/d3 + (x2 - x1)*(d1*dcdx1*d3-p*(x1 - x2)/d3)/(d3**2) \
    + (y2 - y1)*(- c*dcdx1*d1*d3/(np.sqrt(1-c**2)) - h*(x1 - x2)/d3)/d3**2

    first_dxdx2 = p/d3 + (x2 - x1)*(d1*dcdx2*d3-p*(x2 - x1)/d3)/(d3**2) \
    + (y2 - y1)*(- c*dcdx2*d1*d3/(np.sqrt(1-c**2)) - h*(x2 - x1)/d3)/d3**2

    first_dxdy1 =  (x2 - x1)*(d1*dcdy1*d3-p*(y1 - y2)/d3)/(d3**2) \
    - h/d3  +(y2 - y1)*(- c*dcdy1*d1*d3/(np.sqrt(1-c**2)) - h*(y1 - y2)/d3)/d3**2

    first_dxdy2 =  (x2 - x1)*(d1*dcdy2*d3-p*(y2 - y1)/d3)/(d3**2) \
    + h/d3  +(y2 - y1)*(- c*dcdy2*d1*d3/(np.sqrt(1-c**2)) - h*(y2 - y1)/d3)/d3**2

    first_dxdd1 = (x2 - x1)*(c + dcdd1*d1)/d3 + (y2 - y1)*(np.sqrt(1 - c**2) - c*dcdd1*d1/(np.sqrt(1-c**2)))

    first_dxdd2 = (x2 - x1)*d1*(dcdd2)/d3 + (y2 - y1)*(- c*dcdd2*d1/(np.sqrt(1-c**2)))/d3


    first_dydx1 = (y2 - y1)*(d1*dcdx1*d3-p*(x1 - x2)/d3)/(d3**2) \
    + h/d3 + (x1 - x2)*(- c*dcdx1*d1*d3/(np.sqrt(1-c**2)) - h*(x1 - x2)/d3)/d3**2

    first_dydx2 = (y2 - y1)*(d1*dcdx2*d3-p*(x2 - x1)/d3)/(d3**2) \
    - h/d3 + (x1 - x2)*(- c*dcdx2*d1*d3/(np.sqrt(1-c**2)) - h*(x2 - x1)/d3)/d3**2

    first_dydy1 = 1 - p/d3 + (y2 - y1)*(d1*dcdy1*d3-p*(y1 - y2)/d3)/(d3**2) \
     +(x1 - x2)*(- c*dcdy1*d1*d3/(np.sqrt(1-c**2)) - h*(y1 - y2)/d3)/d3**2

    first_dydy2 = p/d3 +(y2 - y1)*(d1*dcdy2*d3-p*(y2 - y1)/d3)/(d3**2) \
     +(x1 - x2)*(- c*dcdy2*d1*d3/(np.sqrt(1-c**2)) - h*(y2 - y1)/d3)/d3**2

    first_dydd1 = (y2 - y1)*(c + dcdd1*d1)/d3 + (x1 - x2)*(np.sqrt(1 - c**2) - c*dcdd1*d1/(np.sqrt(1-c**2)))

    first_dydd2 = (y2 - y1)*d1*(dcdd2)/d3 + (x1 - x2)*(- c*dcdd2*d1/(np.sqrt(1-c**2)))/d3


    second_dxdx1 = 1 - p/d3 + (x2 - x1)*(d1*dcdx1*d3-p*(x1 - x2)/d3)/(d3**2) \
    - (y2 - y1)*(- c*dcdx1*d1*d3/(np.sqrt(1-c**2)) - h*(x1 - x2)/d3)/d3**2

    second_dxdx2 = p/d3 + (x2 - x1)*(d1*dcdx2*d3-p*(x2 - x1)/d3)/(d3**2) \
    - (y2 - y1)*(- c*dcdx2*d1*d3/(np.sqrt(1-c**2)) - h*(x2 - x1)/d3)/d3**2

    second_dxdy1 =  (x2 - x1)*(d1*dcdy1*d3-p*(y1 - y2)/d3)/(d3**2) \
    + h/d3  -(y2 - y1)*(- c*dcdy1*d1*d3/(np.sqrt(1-c**2)) - h*(y1 - y2)/d3)/d3**2

    second_dxdy2 =  (x2 - x1)*(d1*dcdy2*d3-p*(y2 - y1)/d3)/(d3**2) \
    - h/d3  -(y2 - y1)*(- c*dcdy2*d1*d3/(np.sqrt(1-c**2)) - h*(y2 - y1)/d3)/d3**2

    second_dxdd1 = (x2 - x1)*(c + dcdd1*d1)/d3 - (y2 - y1)*(np.sqrt(1 - c**2) - c*dcdd1*d1/(np.sqrt(1-c**2)))

    second_dxdd2 = (x2 - x1)*d1*(dcdd2)/d3 - (y2 - y1)*(- c*dcdd2*d1/(np.sqrt(1-c**2)))/d3


    second_dydx1 = (y2 - y1)*(d1*dcdx1*d3-p*(x1 - x2)/d3)/(d3**2) \
    - h/d3 - (x1 - x2)*(- c*dcdx1*d1*d3/(np.sqrt(1-c**2)) - h*(x1 - x2)/d3)/d3**2

    second_dydx2 = (y2 - y1)*(d1*dcdx2*d3-p*(x2 - x1)/d3)/(d3**2) \
    + h/d3 - (x1 - x2)*(- c*dcdx2*d1*d3/(np.sqrt(1-c**2)) - h*(x2 - x1)/d3)/d3**2

    second_dydy1 = 1 - p/d3 + (y2 - y1)*(d1*dcdy1*d3-p*(y1 - y2)/d3)/(d3**2) \
     -(x1 - x2)*(- c*dcdy1*d1*d3/(np.sqrt(1-c**2)) - h*(y1 - y2)/d3)/d3**2

    second_dydy2 = p/d3 +(y2 - y1)*(d1*dcdy2*d3-p*(y2 - y1)/d3)/(d3**2) \
     -(x1 - x2)*(- c*dcdy2*d1*d3/(np.sqrt(1-c**2)) - h*(y2 - y1)/d3)/d3**2

    second_dydd1 = (y2 - y1)*(c + dcdd1*d1)/d3 - (x1 - x2)*(np.sqrt(1 - c**2) - c*dcdd1*d1/(np.sqrt(1-c**2)))

    second_dydd2 = (y2 - y1)*d1*(dcdd2)/d3 - (x1 - x2)*(- c*dcdd2*d1/(np.sqrt(1-c**2)))/d3

    return (((first_dxdx1, first_dxdx2, first_dxdy1, first_dxdy2, first_dxdd1, first_dxdd2),
            (first_dydx1, first_dydx2, first_dydy1, first_dydy2, first_dydd1, first_dydd2)),
            ((second_dxdx1, second_dxdx2, second_dxdy1, second_dxdy2, second_dxdd1, second_dxdd2),
            (second_dydx1, second_dydx2, second_dydy1, second_dydy2, second_dydd1, second_dydd2)))

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

def position_estimate(pos, dist):
    length = len(pos)
    if length < 3:
        if length == 2:
            return position_estimate_two(pos[0], pos[1], dist[0], dis[1])
        if length == 1:
            return position_estimate_one(pos[0], dist[0])
        else:
            #length == 0
            return (0, 0, 0)
    else:
        trila = []
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                for k in range(j + 1, len(pos)):
                    trila.append(trilaterate_simple(pos[i], pos[j], pos[k], dist[i], dist[j], dist[k]))
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
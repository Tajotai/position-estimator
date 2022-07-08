import numpy as np
import coordinates as co

def fix_est(est, ref1, ref2, node1, node2):
    _, rot = co.cart_to_polar(node1[0], node1[1])
    righthand = np.zeros(est.shape)
    # fix the handedness
    theta = co.cart_to_polar(node2[0], node2[1])[1] - co.cart_to_polar(node1[0], node1[1])[1]
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
        r, theta = co.cart_to_polar(x[0], x[1])
        newx, newy = co.polar_to_cart(r, theta + rot)
        derot[i] = (newx, newy)
    return derot

def fix_est2(est, nodes, ref1, ref2):
    rhest = get_righthand(est, nodes, ref1, ref2)

    _, est_theta = co.cart_to_polar(rhest[:,0], rhest[:,1])
    _, nodes_theta = co.cart_to_polar(nodes[:, 0], nodes[:, 1])
    rots = est_theta - nodes_theta
    avr_rot = avr_angle(rots)
    derot = np.zeros(est.shape)
    for i, x in enumerate(rhest):
        r, theta = co.cart_to_polar(x[0], x[1])
        newx, newy = co.polar_to_cart(r, theta - avr_rot)
        derot[i] = np.array([newx, newy])
    return derot

def fix_est3(est, nodes, tol = 0.0001):
    min_rot = second_rot = 0
    min_err = second_err = np.average(get_location_errors(nodes, est))
    for i in range(1, 4):
        rot = i * np.pi / 2
        newest = rotate_graph(est, rot)
        err = np.average(get_location_errors(nodes, newest))
        if err < min_err:
            second_rot = min_rot
            second_err = min_err
            min_rot = rot
            min_err = err
        else:
            if second_rot < min_rot:
                second_rot = rot
                second_err = err
    while np.abs(min_rot - second_rot) < tol:
        rot_lo = min(min_rot, second_rot)
        rot_hi = max(min_rot, second_rot)
        rot = (rot_hi - rot_lo) / 2
        newest = rotate_graph(est, rot)
        err = np.average(get_location_errors(nodes, newest))
        if err < min_err:
            second_rot = min_rot
            second_err = min_err
            min_rot = rot
            min_err = err
        else:
            second_rot = rot
            second_err = err
    return rotate_graph(est, min_rot)

def fix_est4(est, nodes, ref1, ref2, sink_ix=0):
    est_orig = sink_correct(sink_ix, est)
    nodes_orig = sink_correct(sink_ix, nodes)
    rhest = get_righthand(est_orig, nodes_orig, ref1, ref2)
    min_rot = 0
    min_err = np.average(get_location_errors(nodes_orig, rhest))
    for rot in np.arange(0.0, 2 * np.pi, 0.01):
        newest = rotate_graph(rhest, rot)
        err = np.average(get_location_errors(nodes_orig, newest))
        if err < min_err:
            min_rot = rot
            min_err = err
    rotd = rotate_graph(rhest, min_rot)
    return rotd + nodes[sink_ix]

def sink_correct(sink_ix, graph):
    sink = graph[sink_ix]
    newgraph = graph - sink
    return newgraph

def rotate_graph(graph, rot):
    derot = np.zeros(graph.shape)
    for i, x in enumerate(graph):
        r, theta = co.cart_to_polar(x[0], x[1])
        if r == 0 or theta == np.nan:
            newx, newy = (0, 0)
        else:
            newx, newy = co.polar_to_cart(r, theta + rot)
        derot[i] = np.array([newx, newy])
    return derot

def get_righthand(est, nodes, ref1, ref2, sink_ix=0):
    node1 = nodes[ref1]
    node2 = nodes[ref2]
    righthand = np.zeros(est.shape)
    # fix the handedness
    theta = co.cart_to_polar(node2[0], node2[1])[1] - co.cart_to_polar(node1[0], node1[1])[1]
    # Handedness vote
    nodehand = np.zeros((est.shape[0], est.shape[0]))
    esthand = np.zeros((est.shape[0], est.shape[0]))
    for i in range(est.shape[0]):
        for j in range(i + 1, est.shape[0]):
            if i != sink_ix and j != sink_ix:
                theta1 = (co.cart_to_polar(nodes[j,0], nodes[j,1])[1] -
                          co.cart_to_polar(nodes[i,0], nodes[i,1])[1]) % (2 * np.pi)
                nodehand[i, j] = 1 if theta1 < np.pi else -1
                theta2 = (co.cart_to_polar(est[j, 0], est[j, 1])[1] -
                          co.cart_to_polar(est[i, 0], est[i, 1])[1]) % (2 * np.pi)
                esthand[i, j] = 1 if theta2 < np.pi else -1
    handparities = nodehand * esthand
    handsum = np.sum(handparities)
    if handsum < 0:
        for i, x in enumerate(est):
            righthand[i][0] = x[0]
            righthand[i][1] = -x[1]
    else:
        for i, x in enumerate(est):
            righthand[i][0] = x[0]
            righthand[i][1] = x[1]
    return righthand

def avr_angle(angles):
    avr1 = np.average(angles)
    var1 = np.average((angles - avr1) ** 2)
    angles2 = copy.deepcopy(angles)
    for i, a in np.ndenumerate(angles):
        if a > np.pi:
            angles2[i] = a - 2 * np.pi
    avr2 = np.average(angles2)
    var2 = np.average((angles2 - avr2) ** 2)
    if var2 < var1:
        if avr2 < 0:
            return avr2 + 2 * np.pi
        else:
            return avr2
    else:
        return avr1

def get_location_errors(nodes, est):
    return np.sqrt((nodes[:, 0] - est[:, 0]) ** 2 + (nodes[:, 1] - est[:, 1]) ** 2)

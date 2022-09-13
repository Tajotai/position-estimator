import numpy as np
import coordinates as co
import copy


def maximize_conjugate_gradient(function, dim, partial_diffs, init,
                                iters=10, onedimiters=5, onedimigap=100, tol=0.00000001):
    '''
    Maximizes the function numerically using conjugate gradient method

    :param function: Function to maximize; the function has to intake args as a numpy array unless dim==1!
    :param dim: Dimension, in other words, the number of arguments in the function
    :param partial_diffs: List of partial derivative functions of the function parameter. Should have the number of
    elements indicated by the dim parameter
    :param init: Initial point for arguments
    :param iters: Number of iterations
    :param onedimiters: Number of iterations in the one dimension maximize algorithm
    :param onedimigap: Gap between starting points in one dimension Brent algorithm
    :param tol: Threshold to get under in relative estimate error to stop the algorithm.
    :return: Tuple with second value being the maximum value and first value being the point it's reached at.
    '''
    if dim == 1:
        return maximize_one_dim_brent(function, init, onedimiters)
    P = init
    fP = function(P)
    fret = 0
    g = - grad(init, partial_diffs)
    h = g
    for i in range(iters):
        fun_lin = lambda x : function(P + x * h)
        # min_x, fret = maximize_one_dim_brent(fun_lin, 0, onedimiters, initgap=onedimigap*fdiff/np.sqrt(np.vdot(h, h)))
        min_x, fret = maximize_one_dim_brent(fun_lin, 0, onedimiters,
                                             initgap=onedimigap / np.sqrt(np.vdot(h, h)))
        fdiff = np.abs(fP - fret)
        if 2 * fdiff <= tol * (np.abs(fP) + np.abs(fret) + 0.000000001) and i != 0 and i >= dim:
            return P, fP
        if fret >= fP:
            fP = fret
            P = P + min_x * h
        g_next = - grad(P, partial_diffs)
        gamma = np.vdot(g_next - g, g_next) / np.vdot(g, g)
        h = g_next + gamma * h
        g = g_next
    # print("Warning! Max iters reached in conjugate gradient.")
    return P, function(P)

def grad(x, partial_diffs):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = partial_diffs[i](x)
    return y

def maximize_one_dim_brent(function, init, iters, initgap = 20, tol = 0.0001):
    '''
        Maximizes a one-dimensional function numerically using Brent's method

        :param function: Function to maximize; the function has to intake args as a numpy array unless dim=1!
        :param init: Initial value to start iterations from
        :param iters: Number of iterations
        :param initgap: gap between init and the other two points to form the first parabola
        :return:
    '''
    #initializations
    a = init - initgap / 2
    b = init + initgap / 2
    c = init + initgap
    v = w = x = b
    u = 0
    fx = function(x)
    fv = fw = fx

    e = 0.0
    d = 0.0

    gold = 0.381966
    for i in range(iters):
        tol1 = tol * np.abs(x) + 0.00000001
        tol2 = 2 * tol1
        if np.abs(x - (a + b)/2) <= tol2 - 0.5*(b - a):
            return x, fx
        if (np.abs(e) > tol1):
            #parabolic fit trial
            u, e, d = brent_parab(x, v, w, u, fx, fw, fv, a, b, e, d, tol1, tol2)
        else:
            e = a - x if x >= 0.5*(a + b) else b - x
            d = gold * e
        u = x + d if np.abs(d) >= tol1 else x + tol1*np.sign(d)
        fu = function(u) # The only function evaluation in an iter
        a, b, x, v, w, fx, fv, fw = update_brent(a, b, x, u, v, w, fx, fu, fv, fw)
    # print("Warning! Maximum iters reached in Brent.")
    return x, fx


def brent_parab(x, v, w, u, fx, fw, fv, a, b, e, d, tol1, tol2):
    gold = 0.381966
    x_mid = 0.5 * (a + b)
    r = (x - w) * (fx - fv)
    q = (x - v) * (fx - fw)
    p = (x - v) * q - (x - w) * r
    q = 2.0 * (q - r)
    if (q > 0.0):
        p = -p
    else:
        q = - q
    etemp = e
    e = d
    # Parabolic step acceptability
    if (np.abs(p) >= np.abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x)):
        e = a - x if x >= x_mid else b - x
        d = gold * e
    else:
        d = p / q
        u = x + d
        if (u - a < tol2 or b - u < tol2):
            d = tol1 * np.sign(x_mid -x)
    return u, e, d

def brent_parab_raw(x, v, w, u, fx, fw, fv, a, b, e, d, tol1, tol2):
    gold = 0.381966
    x_mid = 0.5 * (a + b)
    r = (x - w) * (fx - fv)
    q = (x - v) * (fx - fw)
    p = (x - v) * q - (x - w) * r
    q = 2.0 * (q - r)
    if (q > 0.0):
        p = -p
    else:
        q = - q
    etemp = e
    e = d
    # Parabolic step acceptability
    if (np.abs(q) < tol2):
        e = a - x if x >= x_mid else b - x
        d = gold * e
    else:
        d = p / q
        u = x + d
        if (u - a < tol2 or b - u < tol2):
            d = tol1 * np.sign(x_mid - x)
    return u, e, d


def update_brent(a, b, x, u, v, w, fx, fu, fv, fw):
    if (fu >= fx):
        if (u >= x):
            a = x
        else:
            b = x
        v, w, x = w, x, u
        fv, fw, fx = fw, fx, fu
    else:
        if u < x:
            a = u
        else:
            b = u
        if (fu >= fw or w == x):
            v = w
            w = u
            fv = fw
            fw = fu
        elif (fu >= fv or v == x or v == w):
            v = u
            fv = fu
    return a, b, x, v, w, fx, fv, fw

def maximize_sim_ann(function, dim, pos_anchors, n_atoms=2**10, mincoord=-10000, maxcoord=10000,
                     T_0=100, dTperT=-0.002, iters=1000, stepsigma=2000):
    atoms = max_sim_ann_initialize(dim, n_atoms, pos_anchors)
    fp = np.zeros(n_atoms)
    for a in range(n_atoms):
        fp[a] = function(atoms[a,:])
    sigmas = stepsigma * np.ones(n_atoms)
    staycounters = np.zeros(n_atoms)
    T = T_0
    for i in range(iters):
        if i%50 == 0:
            print("iteration",i,": minimum",np.min(fp), " max ",np.max(fp))
        for a in range(n_atoms):
            q = np.array([(atoms[a,j]+2*sigmas[a]*np.random.random()-sigmas[a]) for j in range(dim)])
            fq = function(q)
            if fq > fp[a]:
                atoms[a,:] = q
                fp[a] = fq
                staycounters[a] = 0
            elif np.random.random() < np.exp((fq - fp[a])/T):
                atoms[a, :] = q
                fp[a] = fq
                staycounters[a] = 0
            else:
                staycounters[a] += 1
                if staycounters[a] == 5:
                    sigmas[a] /= 2
                    staycounters[a] = 0
        if i % 10 == 20-1:
            evolve(atoms, dim, fp, 5*T, stepsigma, function)
        T *= 1 + dTperT
    max_ix = np.argmax(fp)
    coords = atoms[max_ix,:]
    print(coords)
    print(fp)
    return coords, fp[max_ix]

def max_sim_ann_initialize(dim, n_atoms, pos_anchors):
    anc_avr = np.average(pos_anchors, axis=0)
    anc_sd = np.sqrt(np.average((pos_anchors[:,0]-anc_avr[0])**2 + (pos_anchors[:,1]-anc_avr[1])**2))
    maxradius = 3 * anc_sd
    coords = np.zeros((n_atoms, dim))
    for i in range(n_atoms):
        for j in range(dim):
            coords[i, j] = anc_avr[j % 2] + 2 * maxradius * (np.random.random() - 0.5)
    return coords

def evolve(atoms, dim, fa, tempr, scale, function):
    max_ix = np.argmax(fa)
    max_fa = fa[max_ix]
    for a in range(len(atoms)):
        if np.random.random() > np.exp((fa[a] - max_fa)/tempr):
            atoms[a] = np.array([(atoms[max_ix, j] + 2 * scale * np.random.random() - scale) for j in range(dim)])
            fa[a] = function(atoms[a])

def hammdist(a, b):
    #ints a and b turned into bit sequences, how many differing bits?
    xor_ab = a ^ b

def loss(x1, y1, x2, y2):
    d = np.sqrt((x2 - x1) ** 2+ (y2 - y1)**2)
    return 20 * (np.log10(d) + np.log10(4 * np.pi / 0.09930819333))

def like(x, y, losses, xi, yi):
    return -np.sum(((losses - loss(x, y, xi, yi))**2)/(20/ np.log(10)))

if __name__ == '__main__':
    losses = np.array([88.1836,   78.2319,   76.1525,   89.2140])
    xi = np.array([-120, 39, -12, -133])
    yi = np.array([139, 58, -62, 189])
    like_ = lambda x : like(x[0], x[1], losses, xi, yi)
    ddx_like = lambda x: (like_(np.array([x[0]+0.0001,x[1]])) - like_(x))*10000
    ddy_like = lambda x: (like_(np.array([x[0], x[1] + 0.0001])) - like_(x)) * 10000
    xy_max = maximize_conjugate_gradient(like_, 2, [ddx_like, ddy_like], np.array([0, 10]), iters=10,
                                         onedimiters=5, onedimigap=10)
    print(xy_max)


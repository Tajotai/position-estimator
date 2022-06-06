import numpy as np
import copy.deepcopy as dc

def maximize_conjugate_gradient(function, dim, partial_diffs, init, iters=10, onedimiters = 10):
    '''
    Maximizes the function numerically using conjugate gradient method

    :param function: Function to maximize; the function has to intake args as a numpy array unless dim=1!
    :param dim: Dimension, in other words, the number of arguments in the function
    :param partial_diffs: List of partial derivative functions of the function parameter. Should have the number of
    elements indicated by the dim parameter
    :param init: Initial point for arguments
    :param iters: Number of iterations
    :return:
    '''
    if dim = 1:
        return maximize_one_dim(function, partial_diffs, init, onedimiters)
    P = init
    g = grad(init, partial_diffs)
    h = init
    for i in range(iters):
        fun_lin = lambda x : function(x * h)
        P, _ = maximize_one_dim(fun_lin, P, onedimiters)
        g_next = - grad(P, partial_diffs)
        gamma = np.vdot(g_next - g, g_next) / np.vdot(g, g)
        h = g_next + gamma * h
    return P, g

def grad(x, partial_diffs):
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = partial_diffs[i](x)
    return y

def maximize_one_dim(function, init, iters):
    return init, function(init)

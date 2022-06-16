import numpy as np
from copy import deepcopy

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def polar_to_cart(r, theta):
    return (r * np.cos(theta), r * np.sin(theta))

def cart_to_polar(x, y):
    if np.size(x) == 1:
        r = np.sqrt(x**2+y**2)
        if x == 0 and y == 0:
            theta = np.nan
        else:
            theta = np.arctan2(y, x)
        return r, theta
    else:
        if x.shape != y.shape:
            raise IndexError("Can not convert shape " + x.shape + " with " + y.shape)
        r = np.zeros(x.shape)
        theta = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_i = x[i]
            y_i = y[i]
            r_i, theta_i = cart_to_polar(x[i], y[i])
            r[i] = r_i
            theta[i] = theta_i
        return r, theta

def covering_probability(r1, r2, R):
    if r1 == 0 and r2 == 0:
        return 1
    cosine = (r1 ** 2 + r2 ** 2 - R ** 2) / (2 * r1 * r2)
    if cosine > 1:
        return 0
    elif cosine < -1:
        return 1
    else:
        return np.arccos(cosine)/np.pi

def location_residue(x, y, a, b, interval):
    return (x//interval)%a + a*((y//interval)%b)

def is_covered(r1, r2, R, theta):
    d = np.sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * np.cos(theta))
    if d <= R:
        return True
    else:
        return False

def select(D, R, r, theta, parent_probs, probs, times, mode, iters):
    if mode == 'deter':
        return select_deter(D, R, r, theta, times, iters)
    else:
        return select_prob(parent_probs, probs, times, iters)

def select_deter(D, R, r, theta, times, iters):
    parent_probs = np.zeros(r.shape)
    probs = np.zeros((r.shape[0], r.shape[0]))
    for i in range(r.shape[0]):
        parent_probs[i] = int(is_covered(D, r[i], R, theta[i]))
        for j in range(r.shape[0]):
            probs[i][j] = int(is_covered(r[i], r[j], theta[j] - theta[i]))
    return select_prob(parent_probs, probs, times, iters)

def select_prob(parent_probs, probs, times, iters):
    ET = np.zeros(parent_probs.shape)
    for i in range(parent_probs.shape[0]):
        ET[i] = calculate_ET(i, parent_probs, probs, times, iters)
    return np.argmax(ET)

def calculate_ET(i, parent_probs, probs, times, iters):
    parent_probs2 = np.delete(parent_probs, i)
    probs2 = np.delete(np.delete(probs, i, 0), i, 1)
    times2 = np.delete(times, i)
    tail_expectation = 0
    for j in range(parent_probs2.shape[0]):
        jj = j
        if jj >= i:
            jj += 1
        tail_expectation += probs[i, jj] * times[i]
        if iters > 1:
            tail_expectation += calculate_ET(j, parent_probs2, probs2, times2, iters - 1)
    ET = parent_probs[i]*(times[i] + tail_expectation)
    return ET


def simulate(parent, R, coords, times, mode, iters):
    # Use a breakpoint in the code line below to debug your script.
    recoords = np.moveaxis(coords, 0, -1)
    x = recoords[0]
    y = recoords[1]
    r, theta = cart_to_polar(x, y)
    parent_probs = np.zeros(r.shape)
    probs = np.zeros((r.shape[0], r.shape[0]))
    D = cart_to_polar(parent[0], parent[1])[0]
    for i in range(r.shape[0]):
        parent_probs[i] = covering_probability(r[i], D, R)
        for j in range(r.shape[0]):
            probs[i, j] = covering_probability(r[i], r[j], R)
    index = select(D, R, r, theta, parent_probs, probs, times, mode, iters)
    print(index, probs)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parent = (1000, 0)
    R = 1500
    node_coords = np.array([[50, 50],[100,0],[600,0],[700, 300], [-100, 1100],[-750, -750]])
    node_times = np.array([50, 150, 200, 100, 100, 200])
    simulate(parent, R, node_coords, node_times, 'prob', 2)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

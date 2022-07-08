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

def coord_transform(coords, increment, rot, rot_ix = 0):
    coords2 = coords - coords[rot_ix,:]
    rs, thetas = cart_to_polar(coords2[:,0], coords2[:,1])
    rot_coords = polar_to_cart(rs, thetas + rot)
    finalcoords = rot_coords + coords[rot_ix,:] + increment
    return finalcoords
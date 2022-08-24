import position_estimator as pe
import maximize as mx
import numpy as np

def four_node_like(x2, x3, y3, x4, y4, p12, p13, p14, p23, p24, p34):
    x = np.array([0, x2, x3, x4])
    y = np.array([0, 0, y3, y4])
    p = np.array([[0, p12, p13, p14], [p12, 0, p23, p24], [p13, p23, 0, p34]])
    sum = 0
    for i in range(3):
        for j in range(i+1, 4):
            sum -= (p[i, j] - 41.99020831627663 - 10 * np.log((x[i] - x[j])**2 + (y[i] - y[j])**2)) ** 2
    return sum

def inc(x, i):
    x[i] += 10000
    return x

def four_node_est(p12, p13, p14, p23, p24, p34):
    like = lambda x: four_node_like(x[0], x[1], x[2], x[3], x[4], p12, p13, p14, p23, p24, p34)
    ddx2_like = lambda x: (like(inc(x, 0)) - like(x)) * 10000
    ddx3_like = lambda x: (like(inc(x, 1)) - like(x)) * 10000
    ddy3_like = lambda x: (like(inc(x, 2)) - like(x)) * 10000
    ddx4_like = lambda x: (like(inc(x, 3)) - like(x)) * 10000
    ddy4_like = lambda x: (like(inc(x, 4)) - like(x)) * 10000
    partial_diffs = np.array([ddx2_like, ddx3_like, ddy3_like, ddx4_like, ddy4_like])
    init = np.array([100, 100, 100, -100, 100])
    x, like_x = mx.maximize_conjugate_gradient(like, 5, partial_diffs, init)
    x2, x3, y3, x4, y4 = x[0], x[1], x[2], x[3], x[4]
    sigma = np.sqrt(-like(np.array([x2, x3, y3, x4, y4])) / 3)
    return x2, x3, y3, x4, y4, sigma

def sim_four_node_sigma():
    x1, y1 = 0, 0
    x2, y2 = 250, 0
    x3, y3 = -100, 250
    x4, y4 = 0, 300
    sigma = 1
    p12 = pe.errored_signal_generate(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), sigma)
    p13 = pe.errored_signal_generate(np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2), sigma)
    p14 = pe.errored_signal_generate(np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2), sigma)
    p23 = pe.errored_signal_generate(np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2), sigma)
    p24 = pe.errored_signal_generate(np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2), sigma)
    p34 = pe.errored_signal_generate(np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2), sigma)
    x2_est, x3_est, y3_est, x4_est, y4_est, sigma_est = four_node_est(p12, p13, p14, p23, p24, p34)
    print(x2_est)
    print(x3_est)
    print(y3_est)
    print(x4_est)
    print(y4_est)
    print(sigma_est)

if __name__ =='__main__':
    sim_four_node_sigma()




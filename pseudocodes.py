def calculate_distance(rx_pow, tx_pow, mu, beta):
    '''
    gives the distance by transmission and receiving power of the signal,
    using the simplest model for path loss in free space. The constant
    173.491005787 is the wavelength of signal with frequency 1.728 MHz

    :param rx_pow: receiver power in dB
    :param tx_pow: transmitter power in dB
    :return: distance in meters
    '''
    return 10 ** ((tx_pow - rx_pow - log10(4 * pi * mu * beta / 173.491005787)) / 20)


def MINIMUM(node, linkends(node)):
    '''
    Pseudocode for minimum distance in the network:
    node is the current node, and the minimum distance is counted for its branch. The method is called for sink,
    and then called recursively for each of the node connected to it, given by linkends(node).
    When MINIMUM method has finished in a node, the node **transmits** the result to the node that called it recursively.
    '''
    min = inf
    for n in linkends:
        min_n = MINIMUM(n, linkends(n))
        if min_n < min:
            min = min_n
        dist = distance(node, n)
        if dist < min:
            min = dist
    return min


def gridsize(rot_interval, rot_window):
    max_squares = rot_interval / rot_window
    m = int(sqrt(max_squares))
    n = m
    if (m+1)*m <= max_squares:
        n = m+1
        if (m+2)*m <= max_squares:
            n = m+2
    return (m, n)

def gridclass(pos, gridsize, mindist):
    width = (2/3)*mindist
    x_raw = int((pos[0]//width)%gridsize[0])
    y_raw = int((pos[1] // width) % gridsize[1])
    return gridsize[1]*x_raw + y_raw

def gridsize_gcd(gridsize):
    a = gridsize[0]
    b = gridsize[1]
    if a == b:
        return a
    elif a + 1 == b:
        return 1
    elif a % 2 == 0:
        return 2
    else:
        return 1

def bezout_multiplier(mod, inc, prod):
    euclid_mults = []
    a = mod
    b = inc
    c = a - b*(a // b)
    euclid_mults.append(a // b)
    while c != 1:
        a = b
        b = c
        c = a - b * (a // b)
        euclid_mults.append(a // b)
    x = (-1) * euclid_mults[-1]
    y = 1
    del euclid_mults[-1]
    while len(euclid_mults) > 0:
        x_new = y - x * euclid_mults[-1]
        y_new = x
        x = x_new
        y = y_new
        del euclid_mults[-1]
    return (prod * x) % mod


def rotation_time(pos, gridsize, mindist, rot_interval):
    modulo = gridsize[0] * gridsize[1]
    class_ = gridclass(pos, gridsize, mindist)
    window = rot_interval / modulo
    increment = gridsize[0] + int(np.sqrt(gridsize[1]))
    gcd = np.gcd(modulo, increment)
    order = modulo // gcd
    q = class_ // gcd
    r = class_ % gcd
    return (order*r + bezout_multiplier(modulo, increment, q)) * window

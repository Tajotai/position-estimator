import clock

#Speed of light, meters per second
c = 299792458

def main():
    sink_ix = 0
    coordfixer = False
    static = False
    glob = False
    half_netwidth = 1000
    nr_of_nodes = 200
    nr_of_anchors = 20
    maxrange = 750
    # Clock sqrt mean squared error in seconds
    sigma = 0.000001
    nodes = generate_nodes(nr_of_nodes, half_netwidth)
    if coordfixer:
        min = np.min([x[0] ** 2 + x[1] ** 2 for x in nodes])
        nodes = np.concatenate((nodes, np.array([((1 / 2) * np.sqrt(min), 0),
                                                 ((1 / 2) * np.sqrt(min), (1 / 2) * np.sqrt(min))], )), axis=0)
    has_anchors = bool(nr_of_anchors)
    if static:
        # nodes = np.array([[-712., -777.],[ 533 ,  972.],[-699., -774.],[ 702.,  661.],[-335.,  994.],[ 366., -829.],
        #                  [ 930., -533.],[ 382.,  518.],[-435., -566.],[ 423., -930.],[ 827.,  332.],[ 804.,  722.],
        #                  [-328.,  987.],[-388.,  601.],[-524., -551.],[ 978.,  538.],[ 794., -251.],[-498., -907.],
        #                  [ 101.,  658.], [ 586.,  435.],[-281.,  526.],[ 582., -730.],[-547.,   29.],[ 445., -507.],
        #                  [-578.,  501.],[ 273.88409957,    0.],[ 273.88409957,  273.88409957]])
        nodes = np.array([[473, -947.],
                          [142, -835.], [875, 189.], [820, 748.], [-762, 331.], [125, -713.], [676, 262.], [229, -92.],
                          [651, -977.], [-698, -551.], [868, -809.], [70, 654.], [-952, 562.], [29, -149.],
                          [365, 528.], [915, 92.], [126, -41.], [-288, 987.], [-488, 577.], [172, -894.],
                          [-24, -613.], [454, -101.], [107, -776.], [-576, -550.], [-811, 605.], [-496, -700.],
                          [-225, -431.], [946, -190.], [132, 885.], [-347, 596.], [305, -573.], [148, 702.],
                          [861, 475.], [-123, -368.], [309, 711.], [-603, -850.], [49, -489.], [523, -908.],
                          [-414, -679.], [-661, 332.], [544, 996.], [641, -906.], [-450, 308.], [-800, -958.],
                          [-661, -444.], [-430, 999.], [-895, -884.], [-162, -203.], [-209, 307.], [-192, -524.]])
    anchors = []
    anchor_locs = []
    if has_anchors:
        anchors = np.arange(nr_of_anchors)
        anchor_locs = nodes[anchors]
    dist = distances(nodes)
    clocks = []

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
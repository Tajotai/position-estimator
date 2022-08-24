import net_estimator as ne

def get_params():
    drops = 100
    sink_ix = 0
    coordfixer = False
    static = False
    half_netwidth = 2000
    nr_of_nodes = 500
    nr_of_anchors = 20
    maxrange = 750
    sigma = 10
    tx_pow = 100
    iters = 20
    plot_interval = 1
    args = [sink_ix, coordfixer, static, half_netwidth, nr_of_nodes, nr_of_anchors, maxrange, sigma, tx_pow,
            iters, plot_interval]
    return drops, args

def main():
    drops, args = get_params()
    for i in range(drops):
        ne.main(args)
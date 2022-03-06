import numpy as np
import random as rd
import expect as exp
import net_estimator as ne

def power_attenuate(pow, dis, mu, beta):
    # constant 173.491005787 is the wavelength in meters with f = 1.728 MHz
    return pow - 2*10*np.log10(dis) - np.log10(4*np.pi*mu*beta/173.491005787)

def ref_select(nodenrs, sinkdist):
    nr = len(nodenrs)

def getRouters(net):
    routers = []
    for e in net:
        if e[1] not in routers and e[1]!=-1:
            routers.append(e[1])
    routers.sort()
    return routers

def bestDistanceNode(bc, nodenr, dist, R, dead):
    best = np.infty
    bestnode = -1
    for m in range(dist.shape[0]):
        if (m in bc) and (dist[nodenr, m] < min(best, R)) and (m not in dead):
            best = dist[nodenr, m]
            bestnode = m
    return bestnode

def netRound(net, bc, dist, R, sinkdist, dead = []):
    newbc = []
    for nodenr in range(dist.shape[0]):
        if (not nodeIsInNet(net, nodenr)) and (not nodenr in dead) :
            bestnode = bestDistanceNode(bc, nodenr, dist, R, dead)
            if bestnode != -1:
                net.append((nodenr, bestnode))
                newbc.append(nodenr)
            else:
                if sinkdist[nodenr] < R:
                    net.append((nodenr, -1))
                    newbc.append(nodenr)

    return net, newbc


def nodeIsInNet(net, n):
    found = False
    for e in net:
        if e[0] == n or e[1] == n:
            found = True
    return found


def netFull(net, nodes):
    found = np.zeros(nodes)
    for e in net:
        found[e[0]] = 1
        found[e[1]] = 1
    return (found == 1).all()

def netOrganize(net, beaconing, nrofnodes, distances, rge, sinkdistances, dead=[], rotatenro = None):
    while not netFull(net, nrofnodes):
        net, beaconing = netRound(net, beaconing, distances, rge, sinkdistances, dead)
        if rotatenro in beaconing:
            beaconing.remove(rotatenro)
        if beaconing == []:
            return
    return

def recursiveDelete(net, n):
    for e in net:
        if e[1] == n:
            recursiveDelete(net, e[0])
            net.remove(e)

# removes a node from the net and triggers rotation
def killNode(net, nodenr, dist, nrofnodes, rge, sinkdist, dead):
    dead.append(nodenr)
    rotate(net, nodenr, dist, nrofnodes, rge, sinkdist, dead)
    for e in net:
        if e[0] == nodenr:
            net.remove(e)
            break
    return

def netRefactor(net, bat_levels, dist, rot_times, dead, nrofnodes, rge, sinkdist):
    # True when rotation caused by node death, False when caused by regular rotation
    kill = False
    min = float(inf)
    node_ix = 0
    for r in range(len(rot_times)):
        if rot_times[r] < min and r not in dead:
            min = rot_times[r]
            node_ix = r
    for l in range(len(bat_levels)):
        if bat_levels[l] < min and l not in dead:
            min = bat_levels[l]
            node_ix = l
            kill = True
    if kill:
        killNode(net, node_ix, dist, nrofnodes, rge, sinkdist, dead)
        print ("killed"+str(node_ix))
        print(net)
    else:
        rotate(net, node_ix, dist, nrofnodes, rge, sinkdist, dead)
        rot_times[node_ix] += 1250
    for r in range(len(rot_times)):
        rot_times[r] -= min
    for l in range(len(bat_levels)):
        bat_levels[l] -= min
    return net, bat_levels, rot_times, dead

def rotate(net, rotatenro, dist, nrofnodes, rge, sinkdist, dead=[]):
    recursiveDelete(net, rotatenro)
    bc = getRotateBeacons(net, rotatenro, dist)
    netOrganize(net, bc, nrofnodes, dist, rge, sinkdist, dead)

def getRotateBeacons(net, rnro, dist):
    bc = []
    parent = None
    for e in net:
        if e[0] == rnro:
            parent = e[1]
            break
    for ee in net:
        if ee[1] == parent and dist[ee[0], rnro]<500 and ee[0]!= rnro:
            bc.append(ee[0])
    return bc


if __name__ == '__main__':
    sink = (0, 0)
    nodes = np.zeros((100, 2))
    rge = 750
    for a in range(nodes.shape[0]):
        for b in range(nodes.shape[1]):
            nodes[a, b] = rd.randrange(-2000, 2000)
    distances = ne.distances(nodes)
    sinkdistances = np.zeros(nodes.shape[0])
    for a in range(nodes.shape[0]):
        r, theta = exp.cart_to_polar(nodes[a, 0] - sink[0], nodes[a, 1] - sink[1])
        sinkdistances[a] = r
    net = []
    beaconing = []
    netOrganize(net, beaconing, nodes.shape[0], distances, rge, sinkdistances)
    lifetimes = np.zeros(nodes.shape[0])
    for i in range(nodes.shape[0]):
        # exponential(100) distributed random lifetimes
        lifetimes[i] = 1000-np.log((1000-rd.randrange(1000))/1000)*1000
    battery_levels = np.zeros(nodes.shape[0])
    location_residues = []
    for n in nodes:
        location_residues.append(expect.location_residue(n[0], n[1], 5, 5, 200))
    rotation_times = []
    for i in range(nodes.shape[0]):
        rotation_times.append(50 * location_residues[i] + 100)
    for i in range(nodes.shape[0]):
        battery_levels[i] = lifetimes[i]
    dead = []
    while not net == []:
        net, battery_levels, rotation_times, dead = \
            netRefactor(net, battery_levels, distances, rotation_times, dead, nodes.shape[0], rge, sinkdistances)
    print(nodes)
    print(lifetimes)
    print(np.average(lifetimes))




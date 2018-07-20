# Ce module rassemble les fonctions permettant d'utiliser l'opérateur EC

import cvrp.route as route
import cvrp.utile as utile
import cvrp.const as const

def reject(route, routes, voisins, inst, demand,capacity):
    point = route[1]
    for i in voisins[point]:
        r = route.find_route(i, routes)
        if r != route and len(r) > 2 and route.route_demand(r, demand)+demand[point] <= capacity:
            routes.remove(route)
            r.insert(r.index(i)+1, point)
            return routes
    return routes

# Compute the saving of the new edge


def saving(i, ri, j, rj, inst):
    ri.append(0)
    rj.append(0)
    s = utile.distance(inst[ri[i]], inst[ri[i+1]])
    s += utile.distance(inst[ri[i]], inst[ri[i-1]])
    s -= utile.distance(inst[ri[i+1]], inst[ri[i-1]])
    s += utile.distance(inst[rj[j]], inst[rj[j+1]])
    s -= utile.distance(inst[ri[i]], inst[rj[j]])
    s -= utile.distance(inst[ri[i]], inst[rj[j+1]])
    ri.pop()
    rj.pop()
    return s

# Code for ejection-chain operator. Apply the operator for a certain edge.




# evalue a possible next edge.



def eval_cand(point, voisins, routes, inst, demand,capacity, fe,mode):
    (r1, r2), v = route.another_route(point, voisins, routes, demand, inst,capacity, fe, "EC", mode)
    if v < 0:
        return const.Error
    i_v, i = r2.index(v), r1.index(point)
    return (saving(i, r1, i_v, r2, inst), (i, i_v), (r1, r2))

# Return the point to relocate and his neighbour considered.


def rd_cand(route, np, voisins, routes, inst, demand,capacity, fe, mode):
    parcours = utile.permut([i for i in range(len(route))])
    best = 0
    for i in parcours:
        p = route[i]
        if p != np:
            cp = utile.rd_point((route[i-1], p), routes, inst)
            cand = eval_cand(cp, voisins, routes, inst, demand,capacity, fe,mode)
            if cand[0] > best:
                if mode == "RD":
                    return cand
                elif mode == "DE":
                    best = cand[0]
                    best_cand = cand
    if best > 0:
        return best_cand
    return const.Error


def ejection_chain(l, point, voisins, routes, inst, demand,capacity, fe, mode):
    S = 0  # global cost modification of the current solution

    s, I, R = eval_cand(point, voisins, routes, inst, demand,capacity, fe,mode)

    if (s, I, R) == const.Error:
        return routes

    S += s
    relocated_cust = R[0][I[0]]

    # update the routes

    R[1].insert(I[1]+1, relocated_cust)
    R[0].remove(relocated_cust)

    for k in range(l-1):
        curr_route = R[1]
        s, I, R = rd_cand(curr_route, relocated_cust,
                          voisins, routes, inst, demand,capacity, fe, mode)

        if (s, I, R) == const.Error:
            return routes
        S += s

        relocated_cust = R[0][I[0]]
        R[1].insert(I[1]+1, relocated_cust)
        R[0].remove(relocated_cust)

    return routes
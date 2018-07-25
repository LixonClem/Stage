# Ce module rassemble les fonctions permettant d'utiliser l'opérateur LK
# Code for LK, take only one route in argument

#import cvrp.route as route
#import cvrp.utile as utile
import cvrp.const as const

# Try to improve the cost of the route by making a 2-opt around the depot


def decross_route(route, inst):
    route.append(0)
    d = (utile.distance(inst[route[2]], inst[route[1]]) +
         utile.distance(inst[route[0]], inst[route[-2]]) -
         utile.distance(inst[route[0]], inst[route[2]]) -
         utile.distance(inst[route[-2]], inst[route[1]]))
    if d > 0:   # There is an improvement
        cand = route.copy()
        cand.remove(route[1])
        cand.insert(-1, route[1])
        cand.pop()
        return cand
    else:
        route.pop()
        return route

# Code for 2-opt operator


def DeuxOpt(route, inst):
    l = len(route)-1
    best_tuple = (0, 0)
    best = 2e-5  # to avoid floating errors
    for i in range(l-1):
        pi = inst[route[i]]
        spi = inst[route[i+1]]

        for j in range(i+2, l-1):
            pj = inst[route[j]]
            spj = inst[route[j+1]]
            d = (utile.distance(pi, spi) + utile.distance(pj, spj)) - \
                utile.distance(pi, pj) - utile.distance(spi, spj)

            if d > best:
                best_tuple = (i, j)
                best = d
    if best_tuple[0] != best_tuple[1]:
        cand = route.copy()
        cand[best_tuple[0]+1], cand[best_tuple[1]
                                    ] = cand[best_tuple[1]], cand[best_tuple[0]+1]
        return cand
    else:
        return route


# The code for LK corresponds of an iteration of 2-opt
# because we don't need to apply an other k-opt

def LK(route, inst):
    route.append(0)
    next_cand = DeuxOpt(route, inst)
    while route != next_cand:
        if len(route) >= 3:
            route = decross_route(route, inst)
        route = next_cand.copy()
        next_cand = DeuxOpt(route, inst)
    route.pop()
    return next_cand

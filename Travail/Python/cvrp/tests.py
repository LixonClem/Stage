# Ce module rassemble des fonctions de tests.

import cvrp.route as route

def verification(sol, demand,capacity):
    for r in sol:
        if route.route_demand(r, demand) > capacity:
            return False
    return True

def is_in_route(i, routes):
    booleen = False
    for r in routes:
        if i in r:
            booleen = True
    return booleen
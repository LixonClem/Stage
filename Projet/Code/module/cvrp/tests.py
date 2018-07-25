# This module contains tests functions

import cvrp.route as route

# Verify if the solution is correct
# (ie if each route doesn't exceed his capacity)


def verification(sol, demand, capacity):
    for r in sol:
        if route.route_demand(r, demand) > capacity:
            return False
    return True

# Verify if the customer i is in the current solution


def is_in_route(i, routes):
    booleen = False
    for r in routes:
        if i in r:
            booleen = True
    return booleen

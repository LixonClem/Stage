# This module contains the possible operations we can make on solutions and routes

import cvrp.utile as utile
import cvrp.learning as learn
import numpy as np

# Compute the demand of the route given


def route_demand(route, demand):
    d = 0
    for i in route:
        d += demand[i]
    return d

# Compute the cost of the solution given


def cost_sol(solution, inst, mode):
    c = 0
    for r in solution:

        # Distances are floating numbers
        if mode == "Float":
            for i in range(len(r)-1):
                a = inst[r[i]]
                b = inst[r[i+1]]

                c += utile.distance(a, b)
            c += utile.distance(inst[r[len(r)-1]], inst[r[0]])

        # Distances are int
        elif mode == "Int":
            for i in range(len(r)-1):
                a = inst[r[i]]
                b = inst[r[i+1]]
                c += round(utile.distance(a, b))
            c += round(utile.distance(inst[r[len(r)-1]], inst[r[0]]))
    return c

# Find the route, which contains the customer i


def find_route(i, routes):
    for k in range(len(routes)):
        if i in routes[k]:
            return routes[k]

# Return a true copy of the sol given


def copy_sol(sol):
    new_sol = []
    for i in sol:
        r = list(np.copy(i))
        new_sol += [r.copy()]
    return new_sol

 # Return the nearest route of the point given


def another_route(point, voisins, routes, inst, demand, capacity, fixedEdges, operator, mode):

    r1 = find_route(point, routes)
    adja = utile.fixed_adjacents(point, fixedEdges)

    if mode == "RD":
        permut_voisins = utile.permut(voisins[point])
    elif mode == "DE":
        permut_voisins = voisins[point]

    if operator == "CE":
        for i in permut_voisins:
            r2 = find_route(i, routes)
            # we verify that the future demand on the route won't exceed his capacity

            if r2 != r1 and i != 0 and route_demand(r1, demand)-demand[point]+demand[i] <= capacity and route_demand(r2, demand)-demand[i]+demand[point] <= capacity:
                return ((r1, r2), i, point)

        # error case, we haven't found a second route, so no modifications
        return ((r1, r1), -1, -1)

    elif operator == "EC":
        for i in permut_voisins:
            r2 = find_route(i, routes)
            if r2 != r1 and i != 0 and len(adja) == 0 and route_demand(r2, demand)+demand[point] <= capacity:
                return ((r1, r2), i)
        return (r1, r1), -1

# Normalize the solution given


def normalize_solution(sol):
    for i in range(len(sol)):
        if sol[i][1] > sol[i][len(sol[i])-1]:
            sol[i].reverse()
            sol[i].pop()
            sol[i].insert(0, 0)
    sol.sort()
    return sol

# Return all edges of the solution given


def all_edges(sol):
    E = []
    for r in sol:
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            E.append((pi, pj))
        E.append((r[len(r)-1], r[0]))
    return E

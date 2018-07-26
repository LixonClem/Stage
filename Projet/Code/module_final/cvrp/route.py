# This module contains the possible operations we can make on solutions and routes

import cvrp.utile as utile
import numpy as np
import cvrp.const as const
# Compute the demand of the route given


def route_demand(route):
    d = 0
    for i in route:

        d += const.demand[i]
    return d

# Compute the cost of the solution given


def cost_sol(solution, mode):
    c = 0
    for r in solution:

        # Distances are floating numbers
        if mode == "Float":
            for i in range(len(r)-1):
                a = const.instance[r[i]]
                b = const.instance[r[i+1]]

                c += utile.distance(a, b)
            c += utile.distance(const.instance[r[len(r)-1]],
                                const.instance[r[0]])

        # Distances are int
        elif mode == "Int":
            for i in range(len(r)-1):
                a = const.instance[r[i]]
                b = const.instance[r[i+1]]
                c += round(utile.distance(a, b))
            c += round(utile.distance(
                const.instance[r[len(r)-1]], const.instance[r[0]]))
    return c

# Verify if the solution is correct
# (ie if each route doesn't exceed his capacity)


def verification(sol):
    for r in sol:
        if route_demand(r) > const.capacity:
            return False
    return True


# Find the route, which contains the customer i


def find_route(i, routes):
    for k in range(len(routes)):
        if i in routes[k]:
            return routes[k]


# Verify if the customer i is in the current solution


def is_in_route(i, routes):
    booleen = False
    for r in routes:
        if i in r:
            booleen = True
    return booleen

# Return a true copy of the sol given


def copy_sol(sol):
    new_sol = []
    for i in sol:
        r = list(np.copy(i))
        new_sol += [r.copy()]
    return new_sol

 # Return the nearest route of the point given


def another_route(point, voisins, routes, fixedEdges, operator, mode):

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

            if (r2 != r1 and i != 0) and (route_demand(r1)-const.demand[point]+const.demand[i] <= const.capacity) and (route_demand(r2)-const.demand[i]+const.demand[point] <= const.capacity):
                return ((r1, r2), i, point)

        # error case, we haven't found a second route, so no modifications
        return ((r1, r1), -1, -1)

    elif operator == "EC":
        for i in permut_voisins:
            r2 = find_route(i, routes)
            if r2 != r1 and i != 0 and len(adja) == 0 and route_demand(r2)+const.demand[point] <= const.capacity:
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

# Complete a partial solution by adding customers which haven't route.
# And verify if the solution given is correct


def complete(routes):
    for i in range(len(routes)):
        routes[i].insert(0, 0)
    while not verification(routes):

        for ri in routes:

            if route_demand(ri) > const.capacity:

                routes.remove(ri)
                d = 0
                i = 0
                nr1 = []
                while i < len(ri) and d <= const.capacity:

                    nr1.append(ri[i])
                    i += 1
                    d += const.demand[ri[i]]

                nr2 = [0] + ri[ri.index(ri[i-1]):]

                routes.append(nr1)
                routes.append(nr2)
    for p in range(len(const.instance)):
        if not is_in_route(p, routes):
            routes.append([0, p])
    for i in range(len(routes)):
        routes[i].append(0)
    return routes

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

# Return the list of edges in common between two solutions


def common_edges(sol1, sol2):
    E1 = all_edges(sol1)
    E2 = all_edges(sol2)
    E = []
    E_init = []
    E_final = []
    for i in E1:
        for j in E2:
            if utile.are_equal(i, j) and (i[0], i[1]) not in E and (i[1], i[0]) not in E:
                E.append(i)

    for i in E1:
        if i not in E and (i[1], i[0]) not in E:
            E_init.append(i)
    for j in E2:
        if j not in E and (j[1], j[0]) not in E:
            E_final.append(j)
    return E, E_init, E_final

# Ce module décrit les opérations possibles sur les tournées

import cvrp.utile as utile
import cvrp.learning as learn
import numpy as np

def route_demand(route, demand):

    d = 0
    for i in route:
        d += demand[i]
    return d

def cost_sol(routes, inst, mode):
    c = 0
    for r in routes:

        if mode == "Float":
            for i in range(len(r)-1):
                a = inst[r[i]]
                b = inst[r[i+1]]
            
                c += utile.distance(a, b)
            c += utile.distance(inst[r[len(r)-1]], inst[r[0]])

        elif mode == "Int":
            for i in range(len(r)-1):
                a = inst[r[i]]
                b = inst[r[i+1]]
                c += round(utile.distance(a, b))
            c += round(utile.distance(inst[r[len(r)-1]], inst[r[0]]))
    return c

def find_route(i, routes):  # Trouve la route à laquelle appartient l'usager i
    for k in range(len(routes)):
        if i in routes[k]:
            return routes[k]

def copy_sol(sol):
    new_sol = []
    for i in sol:
        r = list(np.copy(i))
        new_sol += [r.copy()]
    return new_sol

 # Return the nearest route of the point given

def another_route(point, voisins, routes, inst, demand, capacity,fixedEdges, operator, mode):
    
    r1 = find_route(point, routes)
    adja = learn.adjacents(point,fixedEdges)
    if mode == "RD":
        permut_voisins = utile.permut(voisins[point])
    elif mode == "DE":
        permut_voisins = voisins[point]

    if operator == "CE":
        for i in permut_voisins:
            r2 = find_route(i, routes)
            # we verify that the future demand on the route won't exceed his capacity

            if r2 != r1 and i != 0  and route_demand(r1, demand)-demand[point]+demand[i] <= capacity and route_demand(r2, demand)-demand[i]+demand[point] <= capacity:
                return ((r1, r2), i ,point)

    # error case, we haven't found a second route, so no modifications
        return ((r1, r1), -1 , -1)

    elif operator == "EC":
        for i in permut_voisins:
            r2 = find_route(i, routes)
            if r2 != r1 and i != 0 and len(adja) == 0 and route_demand(r2, demand)+demand[point] <= capacity:
                return ((r1, r2), i)
        return (r1, r1), -1

def normalize_solution(sol):
    for i in range(len(sol)):
        if sol[i][1] > sol[i][len(sol[i])-1]:
            sol[i].reverse()
            sol[i].pop()
            sol[i].insert(0, 0)
    sol.sort()
    return sol
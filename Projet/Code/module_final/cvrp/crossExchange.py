# Ce module rassemble les fonctions permettant d'utiliser l'opérateur CE
# Code for the cross-exchange operator. Apply the operator for a certain edge.

import cvrp.route as route
import cvrp.utile as utile
import cvrp.const as const

# Apply the cross-exchange operator


def cross_exchange(point, voisins, routes, fixedEdges, mode):
    # compute the two routes considered, and the nearest neighbor of the point we remove.
    (r1, r2), neigh, c = route.another_route(point, voisins,
                                             routes,  fixedEdges, "CE", mode)
    if neigh < 0:
        return routes

    # copy of the current solution
    current_cand = [r1.copy(), r2.copy()]
    cand = []
    c_init = route.cost_sol(current_cand, const.quality_cost)

    i_neigh = current_cand[1].index(neigh)
    i_c = current_cand[0].index(c)

    # we verify that we won't exchange the depot
    if i_neigh != 1:
        current_cand[0][i_c], current_cand[1][i_neigh -
                                              1] = current_cand[1][i_neigh-1], c
    else:
        current_cand[0][i_c], current_cand[1][i_neigh] = current_cand[1][i_neigh], c

    # random exploration
    if mode == "RD":
        parcours_i = utile.permut([i for i in range(len(r2)-1)])
        parcours_j = utile.permut([j for j in range(len(r1)-1)])

    # order exploration
    if mode == "DE":
        parcours_i = [i for i in range(len(r2)-1)]
        parcours_j = [j for j in range(len(r1)-1)]

    # we try to exchange two customers between the routes of c and neigh
    for i in parcours_i:
        if i != i_neigh-1:
            for j in parcours_j:
                if j != i_c-1:
                    p1 = current_cand[0][j+1]
                    p2 = current_cand[1][i+1]

                    current_cand[0][j+1], current_cand[1][i + 1] = p2, p1

                    if route.cost_sol(current_cand,  const.quality_cost) < c_init and route.route_demand(current_cand[0]) <= const.capacity and route.route_demand(current_cand[1]) <= const.capacity:

                        # first improve
                        if mode == "RD":
                            routes.remove(r1)
                            routes.remove(r2)
                            routes = routes + current_cand
                            return routes

                        # return the best
                        elif mode == "DE":
                            c_init = route.cost_sol(
                                current_cand, const.quality_cost)
                            cand = route.copy_sol(current_cand)

                # reset the modification
                current_cand = [r1.copy(), r2.copy()]

    if mode == "DE" and cand != []:
        routes.remove(r1)
        routes.remove(r2)
        routes = routes + cand
    return routes

# Ce module rassemble les fonctions permettant d'utiliser l'opérateur CE
# Code for the cross-exchange operator. Apply the operator for a certain edge.

import cvrp.route as route
import cvrp.utile as utile
import cvrp.const as const

# Apply the cross-exchange operator


def cross_exchange(point, voisins, routes, inst, demand,capacity, fixedEdges, mode):
    # compute the two routes considered, and the NN of the point we remove (a). v is a point
    (r1, r2), v,c = route.another_route(point, voisins, routes, inst,demand,capacity, fixedEdges, "CE", mode)
    if v < 0:
        return routes

    # copy of the current solution
    current_cand = [r1.copy(), r2.copy()]
    cand = []
    c_init = route.cost_sol(current_cand, inst, const.quality_cost)     # for a future comparison

    i_v = current_cand[1].index(v)
    i_c = current_cand[0].index(c)

    if i_v != 1:
        current_cand[0][i_c], current_cand[1][i_v -
                                              1] = current_cand[1][i_v-1], c
    else:
        current_cand[0][i_c], current_cand[1][i_v] = current_cand[1][i_v], c

    if mode == "RD":
        parcours_i = utile.permut([i for i in range(len(r2)-1)])
        parcours_j = utile.permut([j for j in range(len(r1)-1)])

    if mode == "DE":
        parcours_i = [i for i in range(len(r2)-1)]
        parcours_j = [j for j in range(len(r1)-1)]

    for i in parcours_i:
        if i != i_v-1:
            for j in parcours_j:
                if j != i_c-1:
                    p1 = current_cand[0][j+1]
                    p2 = current_cand[1][i+1]

                    current_cand[0][j+1], current_cand[1][i + 1] = p2, p1

                    if route.cost_sol(current_cand, inst, const.quality_cost) < c_init and route.route_demand(current_cand[0], demand) <= capacity and route.route_demand(current_cand[1], demand) <= capacity:
                        if mode == "RD":
                            routes.remove(r1)
                            routes.remove(r2)
                            routes = routes + current_cand
                            return routes
                        elif mode == "DE" :
                           
                            c_init = route.cost_sol(current_cand, inst,const.quality_cost)
                            cand = route.copy_sol(current_cand)

                current_cand = [r1.copy(), r2.copy()]
    if mode == "DE" and cand != []:
        routes.remove(r1)
        routes.remove(r2)
        routes = routes + cand
    return routes
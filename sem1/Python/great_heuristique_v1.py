import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m

global ylim
global xlim
global clim
global Capacity
global instance_test
global nb_cust
global lam
global Error
global KNN
global relocation

ylim = 200
xlim = 200
clim = 20
nb_cust = 100
Capacity = 100
KNN = 10
relocation = 10

lam = 1.4

Error = (0, (0, 0), ([[0], 0], [[0], 0]))

# Instances tests #

instance_test = [(0, 0), (119, -99), (-21, 100), (-30, -24), (6, -155), (6, 189), (148, 174), (68, -97), (161, 14), (-200, -32), (79, -133),
                 (-94, 71), (-39, 111), (180, -8), (-29, -141), (164, 181), (129, -190), (-148, 122), (-47, -16), (-105, -177), (109, -134)]
demand_test = [0, 16, 3, 14, 8, 18, 8, 18, 16,
               15, 9, 0, 18, 1, 20, 20, 13, 13, 2, 12, 14]

#######################
# fonctions d'affichage #
#######################


def print_instance(inst):
    dep = inst[0]
    cust = inst[1:]
    py.plot(dep[0], dep[1], color='blue', marker='o')
    for i in cust:
        py.plot(i[0], i[1], color='red', marker='o')


def print_route(route, inst, c):
    x = []
    y = []
    for i in range(len(route)):
        x.append(inst[route[i]][0])
        y.append(inst[route[i]][1])
    x.append(inst[route[0]][0])
    y.append(inst[route[0]][1])
    py.plot(x, y, label="route " + str(c))


def print_routes(routes, inst):
    c = 1
    for i in routes:
        print_route(i[0], inst, c)
        c += 1


def print_current_sol(routes, inst):
    print_instance(inst)
    print_routes(routes, inst)
    py.show()

 ####################
# fonctions communes #
 ####################

 # Compute the cost of a solution


def distance(p1, p2):
    return m.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def cost_sol(routes, inst):
    c = 0
    for r in routes:
        for i in range(len(r[0])-1):
            a = inst[r[0][i]]
            b = inst[r[0][i+1]]
            c += distance(a, b)
        c += distance(inst[r[0][len(r[0])-1]], inst[r[0][0]])
    return c

# Compute the kNN for each node


def voisins(k, inst):
    v = []
    for i in range(len(inst)):
        vi = []
        couples = []
        for j in range(len(inst)):
            if i != j:
                vi.append([distance(inst[i], inst[j]), j])
        vi.sort()
        for l in vi:
            couples.append(l[1])
        v.append(couples[:k])
    return v


def find_route(i, routes):  # Trouve la route à laquelle appartient l'usager i
    for k in range(len(routes)):
        if i in routes[k][0]:
            return routes[k]

 #####################
# Implemenation of CW #
 #####################

# Code for a basic CW heuristic, give an initial solution for the pb.


def init_routes(inst, demand):
    routes = []
    for j in range(1, len(inst)):
        routej = [[0, j, 0], demand[j]]
        routes.append(routej)
    return routes


def compute_savings(inst, lam):
    savings = [[0 for j in range(len(inst)-1)] for i in range(len(inst)-1)]
    for i in range(len(inst)-1):
        for j in range(len(inst)-1):
            if (i == j):
                savings[i][j] = 0
            else:
                savings[i][j] = distance(
                    inst[i+1], inst[0]) + distance(inst[j+1], inst[0]) - lam*distance(inst[i+1], inst[j+1])
    return savings


def max_savings(n, savings):
    cand = (-1, 0, 0)
    for i in range(n):
        for j in range(i+1, n):
            if cand[0] < 0 or savings[i][j] > cand[2]:
                cand = (i+1, j+1, savings[i][j])
    return cand


def can_merge(i, r1, j, r2):
    if r1 == r2:
        return -1
    elif (r1[0][1] == i and r2[0][len(r2[0])-2] == j and r2[1]+r1[1] <= Capacity):
        return 1
    elif (r1[0][len(r1[0])-2] == i and r2[0][1] == j and r1[1]+r2[1] <= Capacity):
        return 2
    else:
        return -1


def merge_routes(cand, routes, savings, inst):
    i, j = cand[0], cand[1]
    r1, r2 = find_route(i, routes), find_route(j, routes)
    mrge = can_merge(i, r1, j, r2)
    if mrge > 0:
        routes.remove(r1)
        routes.remove(r2)
        if mrge == 1:
            r1[0].pop()
            r2[0].remove(0)
            new_road = [r1[0] + r2[0], r1[1] + r2[1]]
        else:
            r2[0].pop()
            r1[0].remove(0)
            new_road = [r2[0] + r1[0], r1[1] + r2[1]]
        routes.append(new_road)
    savings[i-1][j-1] = 0


def ClarkeWright(inst, demand, lam):
    routes = init_routes(inst, demand)
    savings = compute_savings(inst, lam)
    (i, j, s) = max_savings(len(inst)-1, savings)
    while s > 0:
        merge_routes((i, j, s), routes, savings, inst)
        (i, j, s) = max_savings(len(inst)-1, savings)
    for i in range(len(routes)):
        routes[i][0].pop()
    return routes

 ##################
# Cross - Exchange #
 ##################

# Code for the cross-exchange operator. Apply the operator for a certain edge.

 # Return the nearest route of the edge given


def another_routeCE(edge, voisins, routes, demand):
    (a, b) = edge
    r1 = find_route(a, routes)
    for i in voisins[a]:
        r2 = find_route(i, routes)
        # we verify that the future deman on the route won't exceed his capacity
        if r2 != r1 and i != 0 and r1[1]-demand[b]+demand[i] <= Capacity and r2[1]-demand[i]+demand[b] <= Capacity:
            return ((r1, r2), i)
    # error case, we haven't found a second route, so no modifications
    return ((r1, r1), -1)

# Apply the cross-exchange operator


def cross_exchange(edge, voisins, routes, inst, demand):
    (a, b) = edge

    # compute the two routes considered, and the NN of the point we remove (a). v is a point
    (r1, r2), v = another_routeCE(edge, voisins, routes, demand)
    if v < 0:
        return routes

    c_init = cost_sol(routes, inst)     # for a future comparison
    i_v = r2[0].index(v)
    i_a = r1[0].index(a)

    r1[0][i_a], r2[0][i_v-1] = r2[0][i_v-1], a

    r1[1] = r1[1] - demand[a] + demand[v]  # update the demands on each road
    r2[1] = r2[1] - demand[v] + demand[a]

    # copy of the current solution
    current_cand = [[r1[0].copy(), r1[1]].copy(), [r2[0].copy(), r2[1]].copy()]

    for i in range(len(r2[0])-1):
        if i != i_v-1:
            for j in range(len(r1[0])-1):
                if j != i_a-1:
                    p1 = current_cand[0][0][j+1]
                    p2 = current_cand[1][0][i+1]
                    current_cand[0][1] = current_cand[0][1] - \
                        demand[p1] + demand[p2]
                    current_cand[1][1] = current_cand[1][1] - \
                        demand[p2] + demand[p1]

                    current_cand[0][0][j+1], current_cand[1][0][i + 1] = p2, p1

                    if cost_sol(current_cand, inst) < c_init and current_cand[0][1] <= Capacity and current_cand[1][1] <= Capacity:

                        routes.remove(r1)
                        routes.remove(r2)
                        routes = routes + current_cand
                        return routes

                current_cand = [[r1[0].copy(), r1[1]].copy(), [
                    r2[0].copy(), r2[1]].copy()]
    return routes

 ##################
# Ejection - Chain #
 ##################

# Code for ejection-chain operator. Apply the operator for a certain edge.


def another_routeEC(a, voisins, routes, demand):
    r1 = find_route(a, routes)
    for i in voisins[a]:
        r2 = find_route(i, routes)
        if r2 != r1 and r2[1]+demand[a] <= Capacity:
            return ((r1, r2), i)
    return (r1, r1), -1

# Compute the saving of the new edge


def saving(i, ri, j, rj, inst):
    ri.append(0)
    rj.append(0)
    s = distance(inst[ri[i]], inst[ri[i+1]])
    s += distance(inst[ri[i]], inst[ri[i-1]])
    s -= distance(inst[ri[i+1]], inst[ri[i-1]])
    s += distance(inst[rj[j]], inst[rj[j+1]])
    s -= distance(inst[ri[i]], inst[rj[j]])
    s -= distance(inst[ri[i]], inst[rj[j+1]])
    ri.pop()
    rj.pop()
    return s

# evalue a possible next edge.


def eval_cand(edge, voisins, routes, inst, demand):
    (a, b) = edge
    if b != 0:
        (r1, r2), v = another_routeEC(b, voisins, routes, demand)
        if v < 0:
            return Error
        i_v, i = r2[0].index(v), r1[0].index(b)

    else:
        (r1, r2), v = another_routeEC(a, voisins, routes, demand)
        if v < 0:
            return Error
        i_v, i = r2[0].index(v), r1[0].index(a)
    return (saving(i, r1[0], i_v, r2[0], inst), (i, i_v), (r1, r2))

# return the best relocation for each point p in the route.
# Return the point to relocate and his neighbour considered.


def best_cand(route, np, voisins, routes, inst, demand):
    S = []
    for p in route:
        i = route.index(p)
        if p != np:
            S.append(eval_cand((route[i-1], p), voisins, routes, inst, demand))
    S.sort()
    return S[-1]


def ejection_chain(l, edge, voisins, routes, inst, demand):
    S = 0  # global cost modification of the current solution
    initial_routes = routes.copy()
    print(initial_routes)
    s, I, R = eval_cand(edge, voisins, routes, inst, demand)

    if (s, I, R) == Error:
        return routes

    S += s
    relocated_cust = R[0][0][I[0]]

    # update the routes

    R[1][0].insert(I[1]+1, relocated_cust)
    R[1][1] += demand[relocated_cust]
    R[0][1] -= demand[relocated_cust]
    R[0][0].remove(relocated_cust)

    for k in range(l-1):
        curr_route = R[1][0]
        s, I, R = best_cand(curr_route, relocated_cust,
                            voisins, routes, inst, demand)

        if (s, I, R) == Error:
            return routes
        S += s

        relocated_cust = R[0][0][I[0]]
        R[1][0].insert(I[1]+1, relocated_cust)
        R[1][1] += demand[relocated_cust]
        R[0][1] -= demand[relocated_cust]
        R[0][0].remove(relocated_cust)

    if S < 0:  # If the final result is worse than the initial then we don't apply changes
        return initial_routes

    return routes

 #########################
# Lin-Kernighan Heuristic #
 #########################

# Code for LK, take only one route in argument


def DeuxOpt(route, inst):
    l = len(route)-1
    best_tuple = (0, 0)
    best = 0
    for i in range(l-1):
        pi = inst[route[i]]
        spi = inst[route[i+1]]
        for j in range(i+2, l-1):
            pj = inst[route[j]]
            spj = inst[route[j+1]]
            d = (distance(pi, spi) + distance(pj, spj)) - \
                distance(pi, pj)-distance(spi, spj)
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


# Itérations successives de 2-opt. Pas suffisant si grandes tournées,
# mais suffisant sur des petits morceaux de tournées (en considérant les plus
# proches voisins de la zone autour de l'arête à éliminer).
# i et j délimitent la partie de la tournée à optimiser

def LK(route, inst):
    route.append(0)
    next_cand = DeuxOpt(route, inst)
    while next_cand != route:
        print("yes")
        route = next_cand.copy()
        next_cand = DeuxOpt(route, inst)
    route.pop()
    return next_cand

 #############
# Heuristique #
 #############


def gravity_center(route, inst):
    xg = 0
    yg = 0
    c = 0
    for i in route:
        pi = inst[i]
        xg += pi[0]
        yg += pi[1]
        c += 1
    return (xg/c, yg/c)


def width(i, j, G):
    theta = m.acos(G[1]/distance(G, (0, 0)))
    proj_i = (i[0]*m.sin(theta), i[1]*m.cos(theta))
    proj_j = (j[0]*m.sin(theta), j[1]*m.cos(theta))
    return abs(distance(i, proj_i)-distance(j, proj_j))


def cost(i, j):
    return distance(i, j)


def depth(i, j):
    return max(distance(i, (0, 0)), distance(j, (0, 0)))


def max_depth(inst):
    d = 0
    for i in inst:
        di = distance(i, (0, 0))
        if distance(i, (0, 0)) > d:
            d = distance(i, (0, 0))
    return d


def penalization_function(lw, lc, ld, max_d):
    return lambda i, j, G, p: ((lw * width(i, j, G) + lc * cost(i, j))*(depth(i, j)/max_d)**(ld/2))/(1 + p)


def bad_edge(b, p, routes, inst):
    cand = [0, (0, 0)]
    for r in routes:
        G = gravity_center(r[0], inst)
        for i in range(len(r[0])-1):
            pi = r[0][i]
            pj = r[0][i+1]
            b_ij = b(inst[pi], inst[pj], G, p[pi][pj])
            if b_ij > cand[0]:
                cand[0] = b_ij
                cand[1] = (pi, pj)
    return cand


# to do : retenir le nb de fois ou on ne fait rien: a certains moments faire un reset, global opt.
def apply_heuristic(inst, demand, lam, k, l):
    # Initial solution
    initial_solution = ClarkeWright(inst, demand, lam)
    print(cost_sol(initial_solution, inst))
    print_current_sol(initial_solution, inst)

    # compute global variables
    max_d = max_depth(inst)
    v = voisins(k, inst)
    b = penalization_function(1, 0, 0, max_d)
    p = [[0 for j in range(len(inst))] for i in range(len(inst))]

    # find the worst edge
    for time in range(10):
        worst = bad_edge(b, p, initial_solution, inst)[1]
        p[worst[0]][worst[1]] += 1

        # apply ejection-chain
        routes = ejection_chain(l, worst, v, initial_solution, inst, demand)
        print("Ejection succes")
        print(cost_sol(routes, inst))
        print_current_sol(routes, inst)

        # apply LK
        for i in range(len(routes)):
            routes[i][0] = LK(routes[i][0], inst)
        print("LK success")
        print(cost_sol(routes, inst))
        print_current_sol(routes, inst)

        # apply cross-exchange

        routes = cross_exchange(worst, v, routes, inst, demand)

        print("Cross success")
        print(cost_sol(routes, inst))
        print_current_sol(routes, inst)

        # apply LK
        for i in range(len(routes)):
            routes[i][0] = LK(routes[i][0], inst)
        print("LK success")
        print(cost_sol(routes, inst))
        print_current_sol(routes, inst)

    return routes


# Tests #

reso = apply_heuristic(instance_test, demand_test, lam, KNN, relocation)
print(reso)

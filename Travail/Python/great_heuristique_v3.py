
#-*- coding: utf-8 -*-


import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m
from lxml import etree

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
KNN = 30
relocation = 3

lam = 1

Error = (0, (0, 0), ([[0], 0], [[0], 0]))

#######################
# Lecture fichier xml #
#######################


def read(file):  # give the path of the file
    x = []
    y = []
    demand = [0]
    tree = etree.parse("" + file)
    for abs in tree.xpath("/instance/network/nodes/node/cx"):
        x.append(int(float(abs.text)))
    for ord in tree.xpath("/instance/network/nodes/node/cy"):
        y.append(int(float(ord.text)))
    inst = [(x[i], y[i]) for i in range(len(x))]
    for dem in tree.xpath("/instance/requests/request/quantity"):
        demand.append(int(float(dem.text)))
    return inst, demand

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


def print_edges(edges, inst):
    for e in edges:
        x = [inst[e[0]][0], inst[e[1]][0]]
        y = [inst[e[0]][1], inst[e[1]][1]]
        py.plot(x, y, color='red')


def print_current_sol(routes, inst):
    print_instance(inst)
    print_routes(routes, inst)

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


def copy_sol(sol):
    new_sol = []
    for i in sol:
        r = [list(np.copy(i[0])), i[1]]
        new_sol += [r.copy()]
    return new_sol

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

    # copy of the current solution
    current_cand = [[r1[0].copy(), r1[1]].copy(), [r2[0].copy(), r2[1]].copy()]

    c_init = cost_sol(current_cand, inst)     # for a future comparison

    i_v = current_cand[1][0].index(v)
    i_a = current_cand[0][0].index(a)

    if i_v != 1:
        current_cand[0][0][i_a], current_cand[1][0][i_v -
                                                    1] = current_cand[1][0][i_v-1], a
    else:
        current_cand[0][0][i_a], current_cand[1][0][i_v] = current_cand[1][0][i_v], a

    current_cand[0][1] = current_cand[0][1] - demand[a] + \
        demand[v]  # update the demands on each road
    current_cand[1][1] = current_cand[1][1] - demand[v] + demand[a]

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

def reject(route,routes,voisins,inst,demand):
    point = route[0][1]
    for i in voisins[point]:
        r = find_route(i,routes)
        if r != route and r[1]+demand[point]<=Capacity:
            print(routes)
            print(route)
            routes.remove(route)
            r[0].insert(r[0].index(i),point)
            r[1] += demand[point] 
            return routes

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

# Code for ejection-chain operator. Apply the operator for a certain edge.


def another_routeEC(a, voisins, routes, demand, inst):
    r1 = find_route(a, routes)
    for i in voisins[a]:
        r2 = find_route(i, routes)

        if r2 != r1 and i != 0 and r2[1]+demand[a] <= Capacity:
            return ((r1, r2), i)
    return (r1, r1), -1


# evalue a possible next edge.

def best_point(edge, routes, inst):
    (a, b) = edge
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        r = find_route(a, routes)
        a0 = r[0][r[0].index(a)-1]
        b1 = r[0][r[0].index(b)-1]
        if distance(inst[a0], inst[a])+distance(inst[a], inst[b1]) > distance(inst[a0], inst[b])+distance(inst[b], inst[b1]):
            return a
        else:
            return b


def eval_cand(point, voisins, routes, inst, demand):
    (r1, r2), v = another_routeEC(point, voisins, routes, demand, inst)
    if v < 0:
        return Error
    i_v, i = r2[0].index(v), r1[0].index(point)
    return (saving(i, r1[0], i_v, r2[0], inst), (i, i_v), (r1, r2))

# return the best relocation for each point p in the route.
# Return the point to relocate and his neighbour considered.


def best_cand(route, np, voisins, routes, inst, demand):
    S = []
    for p in route:
        i = route.index(p)
        if p != np:
            cp = best_point((route[i-1], p), routes, inst)
            S.append(eval_cand(cp, voisins, routes, inst, demand))

    S.sort()
    return S[-1]


def ejection_chain(l, point, voisins, routes, inst, demand):
    S = 0  # global cost modification of the current solution
    initial_routes = routes.copy()

    s, I, R = eval_cand(point, voisins, routes, inst, demand)

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
            if S > 0:
                return routes
            else:
                return initial_routes
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


def decross_route(route,inst):
    route.append(0)
    d = (distance(inst[route[2]], inst[route[1]])+distance(inst[route[0]], inst[route[-2]]) -
        distance(inst[route[0]], inst[route[2]]) - distance(inst[route[-2]], inst[route[1]]))
    if d>0:
        cand = route.copy()
        cand.remove(route[1])
        cand.insert(-1,route[1])
        cand.pop()
        return cand
    else:
        route.pop()
        return route

def DeuxOpt(route, inst):
    l = len(route)-1
    best_tuple = (0, 0)
    best = 0
    for i in range(l):
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
    while route!=next_cand:
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


def cost(i, j, p):
    return distance(i, j)*(1 + 0.2*p)


def depth(i, j):
    return max(distance(i, (0, 0)), distance(j, (0, 0)))


def max_depth(inst):
    d = 0
    for i in inst:
        di = distance(i, (0, 0))
        if di > d:
            d = di
    return d


def penalization_function(lw, lc, ld, max_d):
    return lambda i, j, G, p: ((lw * width(i, j, G) + lc * cost(i, j, p))*(depth(i, j)/max_d)**(ld/2))/(1 + p)


def bad_edge(b, p, routes, inst):
    cand = [0, (0, 0)]
    for r in routes:
        G = gravity_center(r[0], inst)
        for i in range(len(r[0])-1):
            pi = r[0][i]
            pj = r[0][i+1]
            b_ij = b(inst[pi], inst[pj], G, p[pi][pj])
            if b_ij > cand[0] and pi != 0 and pj != 0:
                cand[0] = b_ij
                cand[1] = (pi, pj)
    return cand


def apply_heuristic(inst, demand, lam, k, l):
    # Initial solution

    initial_solution = ClarkeWright(inst, demand, lam)
    print(initial_solution)
    for i in range(len(initial_solution)):
        #initial_solution[i][0] = decross_route(initial_solution[i][0].copy(), inst)
        initial_solution[i][0] = LK(initial_solution[i][0].copy(), inst)

    routes2 = copy_sol(initial_solution)
    routes = initial_solution

    # compute global variables
    max_d = max_depth(inst)
    v = voisins(k, inst)

    B = [penalization_function(1, 0, 0, max_d), penalization_function(1, 1, 0, max_d), penalization_function(
        1, 0, 1, max_d), penalization_function(1, 1, 1, max_d), penalization_function(0, 1, 0, max_d), penalization_function(0, 1, 1, max_d)]
    b_i = 0
    b = B[b_i]

    p = [[0 for j in range(len(inst))] for i in range(len(inst))]

    print_current_sol(routes, inst)
    py.show()
    N = 0  # laps without improvement
    gs = 0  # laps for last improvement
    c_init = cost_sol(routes, inst)
    # find the worst edge
    for time in range(3000):

        worst = bad_edge(b, p, routes, inst)[1]

        p[worst[0]][worst[1]] += 1

        # apply ejection-chain
        cp = best_point(worst, routes, inst)
        routes = ejection_chain(l, cp, v, routes, inst, demand)

        # apply cross-exchange

        routes = cross_exchange(worst, v, routes, inst, demand)

        # apply LK
        for i in range(len(routes)):
            routes[i][0] = LK(routes[i][0], inst)

        c_final = cost_sol(routes, inst)

        if gs > 25:
            # return to the last global solution, for gs iterations
            routes = copy_sol(routes2)
            print("pshuitt")
            gs = 0

        if c_final < c_init:

            routes2 = copy_sol(routes)  # new optimum
            N = 0
            c_init = cost_sol(routes2, inst)
            print("youpi")

        if N > 100:
            print("boom")
            b_i += 1
            if b_i < len(B):
                b = B[b_i]
                p = [[0 for j in range(len(inst))]
                     for i in range(len(inst))]
                N = 0
            else:
                b_i = 0
                b = B[b_i]
                p = [[0 for j in range(len(inst))]
                     for i in range(len(inst))]
                N = 0

        gs += 1
        N += 1


    print_current_sol(routes2, inst)
    py.show()

    for i in (routes2):
        if len(i[0])==2:
            routes2 = reject(i,routes2,v,inst,demand)


    for i in range(len(routes2)):
        routes2[i][0] = decross_route(routes2[i][0].copy(), inst)
        routes2[i][0] = LK(routes2[i][0], inst)


    print_current_sol(routes2, inst)
    py.show()
    return initial_solution, routes2

 ###########
# Solutions #
 ###########


def are_equal(edge1, edge2):
    return (edge1 == edge2) or (edge1[1] == edge2[0] and edge1[0] == edge2[1])


def all_edges(sol):
    E = []
    for r in sol:
        for i in range(len(r[0])-1):
            pi = r[0][i]
            pj = r[0][i+1]
            E.append((pi, pj))
        E.append((r[0][len(r[0])-1], r[0][0]))
    return E


def common_edges(sol1, sol2):
    E1 = all_edges(sol1)
    E2 = all_edges(sol2)
    E = []
    for i in E1:
        for j in E2:
            if are_equal(i, j) and (i[0], i[1]) not in E and (i[1], i[0]) not in E:
                E.append(i)
    return E

# Tests #


A_n32_k05 = read("Instances/A-n32-k05.xml")

sol_A3205 = [[[0, 30, 16, 1, 12], 100], [[0, 14, 24], 82], [[0, 20, 5, 25, 10, 29, 15, 22, 9, 18, 8, 28, 4, 11], 82], [[0, 7, 13, 26], 47], [[0, 27, 6, 23, 3, 2, 17, 19, 31, 21], 99]]

A_n33_k05 = read("Instances/A-n33-k05.xml")

sol_A3305 = [[[0, 22, 15, 16, 3, 9, 17], 94], [[0, 23, 11, 6, 24, 2], 82], [[0, 28, 18, 19, 14, 21, 1, 31, 29], 98], [[0, 20, 32, 13, 8, 7, 26, 4], 78], [[0, 10, 30, 25, 27, 5, 12], 94]]

A_n33_k06 = read("Instances/A-n33-k06.xml")

sol_A3306 = [[[0, 21, 12], 91], [[0, 1, 7, 6, 18, 14], 96], [[0, 4, 8, 3, 2, 15, 9, 20, 19], 100], [[0, 32, 10, 11, 29, 17], 80], [[0, 13, 5, 22, 26, 24, 23, 31], 80], [[0, 28, 27, 30, 16, 25],94]]

"""
init, reso = apply_heuristic(A_n33_k06[0], A_n33_k06[1], lam, KNN, relocation)
print(reso)
"""

initial_solution = ClarkeWright(A_n33_k05[0],A_n33_k05[1], lam)
print_current_sol(initial_solution,A_n33_k05[0])
py.show()

for i in range(len(initial_solution)):
    initial_solution[i][0] = LK(initial_solution[i][0].copy(), A_n33_k05[0])
print_current_sol(initial_solution,A_n33_k05[0])
py.show()

print_current_sol(sol_A3305,A_n33_k05[0])
py.show()
"""
E = common_edges(initial_solution,sol_A3305)
print(E)

print_instance(A_n33_k05[0])
print_edges(E,A_n33_k05[0])
py.show()
"""

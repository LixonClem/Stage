
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m
from lxml import etree
import os.path


global Capacity
global instance_test
global lam
global Error
global KNN
global relocation
global mu
global nu
global execute


Capacity = 100
KNN = 30
relocation = 3

Error = (0, (0, 0), ([[0], 0], [[0], 0]))

#######################
# Gestion de fichiers #
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

def writef(namefile, text):
    if not os.path.isfile(namefile):
        f = open(namefile,'w')
        f.write(text + '\n')
        f.close()
    else:
        f = open(namefile,'a')
        f.write(text + '\n')
        f.close()

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
    py.plot(x, y)  # , label="route " + str(c))


def print_routes(routes, inst):
    c = 1
    for i in routes:
        print_route(i, inst, c)
        c += 1


def print_edges(edges, inst,col):
    for e in edges:
        x = [inst[e[0]][0], inst[e[1]][0]]
        y = [inst[e[0]][1], inst[e[1]][1]]
        py.plot(x, y, color=col)


def print_current_sol(routes, inst):
    print_instance(inst)
    print_routes(routes, inst)

 ####################
# fonctions communes #
 ####################

# compute the demand of the route


def route_demand(route, demand):
    d = 0
    for i in route:
        d += demand[i]
    return d

def verification(sol,demand):
    for r in sol:
        if route_demand(r,demand)>Capacity:
            return False
    return True

 # Compute the cost of a solution


def distance(p1, p2):
    return m.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def cost_sol(routes, inst):
    c = 0
    for r in routes:
        for i in range(len(r)-1):
            a = inst[r[i]]
            b = inst[r[i+1]]
            c += distance(a, b)
        c += distance(inst[r[len(r)-1]], inst[r[0]])
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
        if i in routes[k]:
            return routes[k]


def copy_sol(sol):
    new_sol = []
    for i in sol:
        r = list(np.copy(i))
        new_sol += [r.copy()]
    return new_sol

def fixed_alea(edges):
    fe = []
    n = len(edges)
    for i in range(n//10):
        alea = rd.randint(0,n-i-1)
        choice = edges[alea]
        edges.remove(choice)
        fe.append(choice)
    return fe

def fixed_0(edges):
    fe = []
    n = len(edges)
    for i in range(n//2):
        if 0 in edges[i]:
            fe.append(edges[i])
            edges.remove(edges[i])
    return fe


def adjacents(pi,fe):
    a = []
    for e in fe:
        if e[0]==pi and e[1] not in a:
            a.append(e[1])
        elif e[1]==pi and e[0] not in a:
            a.append(e[0])
    return a
 #####################
# Implemenation of CW #
 #####################

# Code for a basic CW heuristic, give an initial solution for the pb.


def init_routes(inst, demand):
    routes = []
    for j in range(1, len(inst)):
        routej = [0, j, 0]
        routes.append(routej)
    return routes


def mean_demand(demand):
    n = len(demand)
    d = 0
    for i in demand:
        d += i
    return d/(n-1)


def compute_savings(inst, demand, lam, mu, nu):
    savings = [[0 for j in range(len(inst)-1)] for i in range(len(inst)-1)]
    d_bar = mean_demand(demand)
    for i in range(len(inst)-1):
        for j in range(i+1,len(inst)-1):
            if (i == j):
                savings[i][j] = 0
            else:
                savings[i][j] = distance(inst[i+1], inst[0]) + distance(inst[j+1], inst[0])- lam*distance(inst[i+1], inst[j+1])+ mu*abs(distance(inst[i+1], inst[0]) -distance(inst[j+1], inst[0]))+ (nu*(demand[i+1] + demand[j+1])/d_bar)
    return savings


def max_savings(n, savings):
    cand = (-1, 0, 0)
    for i in range(n):
        for j in range(i+1, n):
            if cand[0] < 0 or savings[i][j] > cand[2]:
                cand = (i+1, j+1, savings[i][j])
    return cand


def can_merge(i, r1, j, r2, demand):
    if r1 == r2:
        return -1
    elif (r1[1] == i and r2[len(r2)-2] == j and route_demand(r1, demand)+route_demand(r2, demand) <= Capacity):
        return 1
    elif (r1[len(r1)-2] == i and r2[1] == j and route_demand(r1, demand)+route_demand(r2, demand) <= Capacity):
        return 2
    else:
        return -1


def merge_routes(cand, routes, savings, inst, demand):
    i, j = cand[0], cand[1]
    r1, r2 = find_route(i, routes), find_route(j, routes)
    mrge = can_merge(i, r1, j, r2, demand)
    new_road = []
    if mrge > 0:
        routes.remove(r1)
        routes.remove(r2)
        if mrge == 1:
            r1.pop()
            r2.remove(0)
            new_road = r1 + r2
        else:
            r2.pop()
            r1.remove(0)
            new_road = r2 + r1
        routes.append(new_road)
    savings[i-1][j-1] = 0
    savings[j-1][i-1] = 0






def ClarkeWright(inst, demand, lam, mu, nu):
    routes = init_routes(inst, demand)
    savings = compute_savings(inst, demand, lam, mu, nu)
    (i, j, s) = max_savings(len(inst)-1, savings)
    while s > 0:
        merge_routes((i, j, s), routes, savings, inst, demand)
        (i, j, s) = max_savings(len(inst)-1, savings)
    for i in range(len(routes)):
        routes[i].pop()
    return routes

 ##################
# Cross - Exchange #
 ##################

# Code for the cross-exchange operator. Apply the operator for a certain edge.

 # Return the nearest route of the edge given


def another_routeCE(edge, voisins, routes, demand,fe):
    (a, b) = edge
    r1 = find_route(a, routes)
    adja = adjacents(a,fe)
    for i in voisins[a]:
        r2 = find_route(i, routes)
        adjpi = adjacents(r2[r2.index(i)-1],fe)
        # we verify that the future demand on the route won't exceed his capacity
        if r2 != r1 and i != 0 and len(adjpi)==0 and len(adja) == 0 and route_demand(r1, demand)-demand[b]+demand[i] <= Capacity and route_demand(r2, demand)-demand[i]+demand[b] <= Capacity:
            return ((r1, r2), i)
    # error case, we haven't found a second route, so no modifications
    return ((r1, r1), -1)

# Apply the cross-exchange operator


def cross_exchange(edge, voisins, routes, inst, demand,fe):

    copy_routes = copy_sol(routes)
    (a, b) = edge
    feasible = []
    

    # compute the two routes considered, and the NN of the point we remove (a). v is a point
    (r1, r2), v = another_routeCE(edge, voisins, copy_routes, demand,fe)
    if v < 0:
        return routes

    copy_routes.remove(r1)
    copy_routes.remove(r2)
    routesBis = copy_sol(copy_routes)

    # copy of the current solution
    current_cand = copy_sol([r1.copy(), r2.copy()])

    i_v = current_cand[1].index(v)
    i_a = current_cand[0].index(a)

    if i_v != 1:
        current_cand[0][i_a], current_cand[1][i_v -
                                              1] = current_cand[1][i_v-1], a
    else:
        current_cand[0][i_a], current_cand[1][i_v] = current_cand[1][i_v], a

    current_current_cand = copy_sol(current_cand)
    for j in range(len(r2)-1):
        if (i_v != 1 and j != i_v-2) or (j!=0):
            for i in range(len(r1)-1):
                if i != i_a-1:

                    p1 = current_current_cand[0][i+1]
                    p2 = current_current_cand[1][j+1]

                    current_current_cand[0][i+1], current_current_cand[1][j + 1] = p2, p1
                    routesBis = routesBis + current_current_cand
                    adj1 = adjacents(p1,fe)
                    adj2 = adjacents(p2,fe)
                    if  verification(routesBis,demand) and len(adj1)==0 and len(adj2)==0:

                        feasible.append((i+1,j+1))
                current_current_cand = copy_sol(current_cand)
                routesBis = copy_sol(copy_routes)

    if len(feasible)==0:
        return routes

    pivot = feasible[rd.randint(0,len(feasible)-1)]

    p1 = current_cand[0][pivot[0]]
    p2 = current_cand[1][pivot[1]]
    current_cand[0][pivot[0]], current_cand[1][pivot[1]] = p2, p1
    routes = copy_routes + current_cand

    return routes

 ##################
# Ejection - Chain #
 ##################


def reject(route, routes, voisins, inst, demand):
    point = route[1]
    for i in voisins[point]:
        r = find_route(i, routes)
        if r != route and route_demand(r, demand)+demand[point] <= Capacity:
            routes.remove(route)
            r.insert(r.index(i)+1, point)
            return routes
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


def another_routeEC(a, voisins, routes, demand, inst,fe):
    r1 = find_route(a, routes)
    adja = adjacents(a,fe)
    for i in voisins[a]:
        r2 = find_route(i, routes)

        if r2 != r1 and i != 0 and len(adja)==0 and route_demand(r2, demand)+demand[a] <= Capacity:
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
        a0 = r[r.index(a)-1]
        b1 = r[r.index(b)-1]
        if distance(inst[a0], inst[a])+distance(inst[a], inst[b1]) > distance(inst[a0], inst[b])+distance(inst[b], inst[b1]):
            return a
        else:
            return b


def eval_cand(point, voisins, routes, inst, demand,fe):
    (r1, r2), v = another_routeEC(point, voisins, routes, demand, inst,fe)
    if v < 0:
        return Error
    i_v, i = r2.index(v), r1.index(point)
    return (saving(i, r1, i_v, r2, inst), (i, i_v), (r1, r2))

# return the best relocation for each point p in the route.
# Return the point to relocate and his neighbour considered.


def best_cand(route, np, voisins, routes, inst, demand,fe):
    S = []
    for p in route:
        i = route.index(p)
        if p != np:
            cp = best_point((route[i-1], p), routes, inst)
            S.append(eval_cand(cp, voisins, routes, inst, demand, fe))

    S.sort()
    return S[-1]


def ejection_chain(l, point, voisins, routes, inst, demand,fe):
    S = 0  # global cost modification of the current solution

    s, I, R = eval_cand(point, voisins, routes, inst, demand,fe)

    if (s, I, R) == Error:
        return routes

    S += s
    relocated_cust = R[0][I[0]]

    # update the routes

    R[1].insert(I[1]+1, relocated_cust)
    R[0].remove(relocated_cust)

    for k in range(l-1):
        curr_route = R[1]
        s, I, R = best_cand(curr_route, relocated_cust,
                            voisins, routes, inst, demand,fe)

        if (s, I, R) == Error:
            return routes
        S += s

        relocated_cust = R[0][I[0]]
        R[1].insert(I[1]+1, relocated_cust)
        R[0].remove(relocated_cust)
    """
    if S < 0:  # If the final result is worse than the initial then we don't apply changes
        return initial_routes
"""
    return routes

 #########################
# Lin-Kernighan Heuristic #
 #########################

# Code for LK, take only one route in argument


def decross_route(route, inst):
    route.append(0)
    d = (distance(inst[route[2]], inst[route[1]])+distance(inst[route[0]], inst[route[-2]]) -
         distance(inst[route[0]], inst[route[2]]) - distance(inst[route[-2]], inst[route[1]]))
    if d > 0:
        cand = route.copy()
        cand.remove(route[1])
        cand.insert(-1, route[1])
        cand.pop()
        return cand
    else:
        route.pop()
        return route


def DeuxOpt(route, inst):
    l = len(route)-1
    best_tuple = (0, 0)
    best = 2e-10
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
    while route != next_cand:
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


def bad_edge(b, p, routes, inst,fixed):
    cand = [0, (0, 0)]
    for r in routes:
        G = gravity_center(r, inst)
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            b_ij = b(inst[pi], inst[pj], G, p[pi][pj])
            if b_ij > cand[0] and pi != 0 and pj != 0 and (pi,pj) not in fixed and (pj,pi) not in fixed:
                cand[0] = b_ij
                cand[1] = (pi, pj)
    return cand


def apply_heuristic(inst, demand, lam, mu, nu, l,max_d,v):
    # Initial solution
    record = [[0, 7, 25, 35, 16], [0, 27, 32, 15, 30, 13], [0, 24, 29, 36, 6, 14], [0, 4, 10, 11, 12, 22, 23, 28, 2, 33], [0, 20, 8, 5, 3, 1, 34, 17], [0, 18, 31, 19, 9, 21, 26]]
    initial_solution = ClarkeWright(inst, demand, lam, mu, nu)

    for i in range(len(initial_solution)):
        initial_solution[i] = decross_route(initial_solution[i].copy(), inst)
        initial_solution[i] = LK(initial_solution[i].copy(), inst)

    routes2 = copy_sol(initial_solution)
    routes = copy_sol(initial_solution)

    # compute global variables


    B = [penalization_function(1, 0, 0, max_d), penalization_function(1, 1, 0, max_d), penalization_function(
        1, 0, 1, max_d), penalization_function(1, 1, 1, max_d), penalization_function(0, 1, 0, max_d), penalization_function(0, 1, 1, max_d)]
    b_i = 0
    b = B[b_i]

    p = [[0 for j in range(len(inst))] for i in range(len(inst))]


    N = 0  # laps without improvement
    gs = 0  # laps for last improvement
    c_init = cost_sol(routes, inst)
    time = 0
    e,ei,ef = common_edges(initial_solution,record)
    fixed_edges = e
    # find the worst edge
    while time < 1500:

        worst = bad_edge(b, p, routes, inst,fixed_edges)[1]

        p[worst[0]][worst[1]] += 1
        p[worst[1]][worst[0]] += 1

        # apply ejection-chain
        cp = best_point(worst, routes, inst)

        routes = ejection_chain(l, cp, v, routes, inst, demand,fixed_edges)

        for i in range(len(routes)):
            routes[i] = LK(routes[i], inst)
        # apply cross-exchange

        routes = cross_exchange(worst, v, routes, inst, demand,fixed_edges)

        # apply LK
        for i in range(len(routes)):
            routes[i] = LK(routes[i], inst)

        c_final = cost_sol(routes, inst)

        if c_final < c_init:
            routes2 = copy_sol(routes)  # new optimum
            fixed_edges = fixed_alea(all_edges(routes))

            gs = 0
            N = 0
            c_init = cost_sol(routes2, inst)
            time = 0

        if gs > 10:
            # return to the last global solution, for gs iterations
            routes = copy_sol(routes2)
            fixed_edges = fixed_alea(all_edges(routes2))
            gs = 0

        if N > 100:

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
                for i in (routes2):
                    if len(i) == 2:
                        routes2 = reject(i, routes2, v, inst, demand)
                for i in range(len(routes2)):
                    routes2[i] = decross_route(routes2[i].copy(), inst)
                    routes2[i] = LK(routes2[i], inst)
                routes = copy_sol(routes2)
        gs += 1
        N += 1
        time += 1

    for i in (routes2):
        if len(i) == 2:
            routes2 = reject(i, routes2, v, inst, demand)

    for i in range(len(routes2)):
        routes2[i] = decross_route(routes2[i].copy(), inst)
        routes2[i] = LK(routes2[i], inst)

    if not verification(routes2,demand):
        routes2 = initial_solution
    

    return initial_solution, routes2

 ###########
# Solutions #
 ###########


def are_equal(edge1, edge2):
    return (edge1 == edge2) or (edge1[1] == edge2[0] and edge1[0] == edge2[1])


def all_edges(sol):
    E = []
    for r in sol:
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            E.append((pi, pj))
        E.append((r[len(r)-1], r[0]))
    return E


def common_edges(sol1, sol2):
    E1 = all_edges(sol1)
    E2 = all_edges(sol2)
    E = []
    E_init = []
    E_final = []
    for i in E1:
        for j in E2:
            if are_equal(i, j) and (i[0], i[1]) not in E and (i[1], i[0]) not in E:
                E.append(i)
    
    for i in E1:
        if i not in E and (i[1],i[0]) not in E:
            E_init.append(i)
    for j in E2:
        if j not in E and (j[1],j[0]) not in E:
            E_final.append(j)
    return E,E_init,E_final


def rank_costs(E, inst):
    r = []
    rc = []
    for e in E:
        c = distance(inst[e[0]], inst[e[1]])
        r.append((c, e))
    r.sort()
    for i in r:
        rc.append(i[1])
    return rc


def rank_depth(E, inst):
    r = []
    rd = []
    dmax = max_depth(inst)
    for e in E:
        d = depth(inst[e[0]], inst[e[1]])/dmax
        r.append((d, e))
    r.sort()
    for i in r:
        rd.append(i[1])
    return rd


def rank_width(E, sol, inst):
    r = []
    rw = []
    for e in E:
        route = find_route(e[0], sol)
        G = gravity_center(route, inst)
        w = width(inst[e[0]], inst[e[1]], G)
        r.append((w, e))
    r.sort()
    for i in r:
        rw.append(i[1])
    return rw


def rank_edges(sol, inst):
    E = all_edges(sol)
    n = len(E)
    rc = rank_costs(E, inst)
    rd = rank_depth(E, inst)
    rw = rank_width(E, sol, inst)
    return n, rc, rd, rw


def give_rank(e, rank):
    for i in range(len(rank)):
        if are_equal(e, rank[i]):
            return (i+1)


def all_ranks(cE, sol, inst):
    n, rc, rd, rw = rank_edges(sol, inst)
    r = []
    r_mean = []
    for e in cE:
        g1 = give_rank(e, rc)
        g2 = give_rank(e, rd)
        g3 = give_rank(e, rw)

        r_mean.append((g1+g2+g3)/3)
        g = [g1, g2, g3]
        g.sort()
        r.append(g)
        r.sort()
    return n, r, r_mean


def analyse(n, ranks):
    a = [0 for i in range(16)]
    for r in ranks:
        if r[0] < n/3 or r[1] < n/3 or r[2] < n/3:
            a[0] += 1
        if r[0] < n/3 and r[1] < n/3 and r[2] < n/3:
            a[1] += 1
        if r[0] < 15 or r[1] < 15 or r[2] < 15:
            a[2] += 1
        if r[0] < 15 and r[1] < 15 and r[2] < 15:
            a[3] += 1
        if r[0] < 10 or r[1] < 10 or r[2] < 10:
            a[4] += 1
        if r[0] < 10 and r[1] < 10 and r[2] < 10:
            a[5] += 1
        if r[0] < 5 or r[1] < 5 or r[2] < 5:
            a[6] += 1
        if r[0] < 5 and r[1] < 5 and r[2] < 5:
            a[7] += 1

        if r[0] > n-n/3 or r[1] > n-n/3 or r[2] > n-n/3:
            a[8] += 1
        if r[0] > n-n/3 and r[1] > n-n/3 and r[2] > n-n/3:
            a[9] += 1
        if r[0] > n-15 or r[1] > n-15 or r[2] > n-15:
            a[10] += 1
        if r[0] > n-15 and r[1] > n-15 and r[2] > n-15:
            a[11] += 1
        if r[0] > n-10 or r[1] > n-10 or r[2] > n-10:
            a[12] += 1
        if r[0] > n-10 and r[1] > n-10 and r[2] > n-10:
            a[13] += 1
        if r[0] > n-5 or r[1] > n-5 or r[2] > n-5:
            a[14] += 1
        if r[0] > n-5 and r[1] > n-5 and r[2] > n-5:
            a[15] += 1
    return a

# Tests #


##########
A_n32_k05 = read("Instances/A-n32-k05.xml")
# sol_A3205 = [[[0, 30, 16, 1, 12], 100], [[0, 14, 24], 82], [[0, 20, 5, 25, 10, 29, 15, 22, 9, 18, 8, 28, 4, 11], 82], [[0, 7, 13, 26], 47], [[0, 27, 6, 23, 3, 2, 17, 19, 31, 21], 99]]
init_A3205 = [[0, 18, 22, 9, 11, 4, 28, 8], [0, 29, 15, 10, 25, 5, 20], [
    0, 21, 31, 19, 17, 13, 7, 26], [0, 27, 23, 2, 3, 6, 14, 24], [0, 12, 1, 16, 30]]
sol_A3205 = [[0, 21, 31, 19, 17, 13, 7, 26], [0, 28, 11, 4, 23, 2, 3, 6], [
    0, 20, 5, 25, 10, 29, 15, 22, 9, 8, 18], [0, 27, 24, 14], [0, 12, 1, 16, 30]]
##########
A_n33_k05 = read("Instances/A-n33-k05.xml")

# sol_A3305 = [[[0, 22, 15, 16, 3, 9, 17], 94], [[0, 23, 11, 6, 24, 2], 82], [[0, 28, 18, 19, 14, 21, 1, 31, 29], 98], [[0, 20, 32, 13, 8, 7, 26, 4], 78], [[0, 10, 30, 25, 27, 5, 12], 94]]
init_A3305 = [[0, 10, 30, 25, 27, 5, 12], [0, 4, 26, 7, 8, 13, 32, 20], [
    0, 29, 3, 9, 17, 16, 15], [0, 28, 18, 31, 1, 21, 14, 19, 11], [0, 2, 24, 6, 23, 22]]
sol_A3305 = [[0, 20, 32, 13, 8, 7, 26, 4, 22], [0, 10, 30, 25, 27, 5, 12], [
    0, 11, 19, 14, 21, 1, 31, 18, 28], [0, 2, 24, 6, 23], [0, 15, 17, 9, 3, 16, 29]]
##########
A_n33_k06 = read("Instances/A-n33-k06.xml")
init_A3306 = [[0, 4, 8, 3, 9, 15, 20, 2, 5], [0, 11, 29, 6, 7, 19], [0, 13, 1, 18, 17], [
    0, 21, 12, 10], [0, 31, 23, 24, 26, 22, 14], [0, 32, 25, 16, 30, 27, 28]]
sol_A3306 = [[0, 4, 8, 3, 9, 15, 20, 2, 5], [0, 17, 11, 29, 19, 7], [0, 21, 12, 10], [
    0, 32, 25, 16, 30, 27, 28], [0, 31, 23, 24, 26, 22], [0, 13, 6, 18, 1, 14]]
##########
A_n34_k05 = read("Instances/A-n34-k05.xml")
init_A3405 = [[0, 8, 11, 23, 27, 1, 29], [0, 7, 15, 19, 17, 25, 28, 32, 31], [
    0, 21, 3, 12, 9, 22, 16, 2, 33], [0, 4, 26, 30, 24, 5], [0, 14, 6, 13, 10, 20, 18]]
sol_A3405 = [[0, 5, 30, 24, 29, 6, 7], [0, 27, 1, 23, 11, 8, 15, 14], [
    0, 19, 17, 25, 31, 28, 13, 10], [0, 26, 4, 33, 16, 2, 18], [0, 21, 32, 3, 12, 9, 22, 20]]
##########
A_n36_k05 = read("Instances/A-n36-k05.xml")
init_A3605 = [[0, 9, 23, 2, 35, 8, 34, 14], [0, 21, 18, 33, 29, 30, 17, 13, 32, 22, 1], [
    0, 12, 31, 19, 4, 3, 6, 28, 15], [0, 26, 7, 10], [0, 20, 5, 25, 27, 24, 11, 16]]
sol_A3605 = [[0, 28, 14, 34, 23, 2, 35, 8, 15], [0, 1, 22, 32, 13, 17, 30, 29, 33, 18, 21], [
    0, 12, 31, 19, 4, 3, 6, 9], [0, 10, 7, 26], [0, 20, 5, 25, 27, 24, 11, 16]]
##########
A_n37_k05 = read("Instances/A-n37-k05.xml")
init_A3705 = [[0, 30, 25, 35, 18, 26, 31, 28, 32, 29], [0, 17, 14, 23, 20, 19, 2, 12, 1], [
    0, 22, 13, 10, 6, 5, 33, 4, 7], [0, 21, 16], [0, 3, 24, 9, 11, 27, 8, 34, 36, 15]]
sol_A3705 = [[0, 22, 13, 10, 6, 5, 33, 4, 7], [0, 21, 16], [0, 1, 12, 2, 19, 20, 23, 14, 17], [
    0, 3, 24, 9, 11, 27, 8, 25, 35, 18, 26, 15], [0, 34, 36, 29, 32, 28, 31, 30]]
##########
A_n37_k06 = read("Instances/A-n37-k06.xml")
init_A3706 = [[0, 4], [0, 5, 3], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [0, 16], [0, 18], [0, 20], [0, 21], [
    0, 22], [0, 23], [0, 24], [0, 26], [0, 27], [0, 29], [0, 32], [0, 33], [0, 36], [0, 25, 35], [0, 19, 31], [0, 15, 30], [0, 2, 28], [0, 17, 34], [0, 1]]
# sol_A3706 = [[[0, 29, 36, 14], 65], [[0, 24, 16, 7], 47], [[0, 27, 32, 15, 30, 13], 89], [[0, 25, 35], 81], [[0, 26, 21, 9, 1, 3, 5, 8], 96], [[0, 10, 11, 12, 22, 23, 28, 2, 33, 20], 97], [[0, 18, 4, 17, 34, 19, 31, 6], 95]]
# sol_A3706 = [[0, 7, 25, 35, 16], [0, 13, 30, 15, 32, 27], [0, 10, 11, 12, 22, 23, 28, 2, 33], [0, 24, 29, 36, 6, 14], [0, 4, 26, 19, 31, 34, 17, 18], [0, 20, 8, 5, 3, 1, 9, 21]]
sol_A3706 = [[0, 7, 25, 35, 16], [0, 27, 32, 15, 30, 13], [0, 20, 33, 2, 28, 23, 22, 12, 11, 10], [
    0, 14, 6, 36, 29, 24], [0, 31, 19, 9, 21, 26, 4], [0, 18, 17, 34, 1, 3, 5, 8]]
##########
A_n38_k05 = read("Instances/A-n38-k05.xml")
init_A3805 = [[0, 2], [0, 9], [0, 14], [0, 15], [0, 24], [0, 4, 16, 25], [0, 12, 1, 3, 26], [0, 7, 22, 27, 11, 5], [
    0, 31, 37, 28], [0, 8, 23, 35, 33], [0, 18, 6, 34, 29, 19], [0, 10, 30, 21], [0, 17, 36, 13], [0, 20, 32]]
sol_A3805 = [[0, 18, 19, 34, 29, 30, 10], [0, 28, 31, 37, 11, 27, 22, 5], [0, 7, 20, 32,
                                                                           15, 13, 36, 17, 2, 24], [0, 9, 8, 23, 35, 33, 14], [0, 6, 25, 16, 4, 1, 3, 12, 26, 21]]
##########
A_n39_k05 = read("Instances/A-n39-k05.xml")
init_A3905 = [[0, 2, 22, 3, 7, 16, 32, 10, 20], [0, 38, 15, 5, 29, 23, 1, 31, 12], [
    0, 13, 28, 6, 26, 17, 11, 8, 9], [0, 24, 35, 37, 34, 27, 36, 30, 21], [0, 4, 18, 33, 25, 19, 14]]
sol_A3905 = [[0, 2, 22, 3, 7, 16, 32, 10], [0, 38, 15, 5, 29, 20, 23, 1, 31, 12], [
    0, 14, 19, 25, 33, 18, 9, 4], [0, 6, 36, 27, 28, 13, 30, 21], [0, 17, 24, 35, 37, 34, 26, 11, 8]]
##########
A_n39_k06 = read("Instances/A-n39-k06.xml")
init_A3906 = [[0, 3], [0, 5], [0, 11], [0, 13], [0, 15], [0, 20], [0, 24], [0, 26], [0, 30], [0, 27, 16, 10], [
    0, 2, 33, 19, 4, 7, 8], [0, 12, 38], [0, 9, 28, 29], [0, 32, 34, 22, 18], [0, 21, 23, 17, 36, 1, 6], [0, 37, 31, 35, 25, 14]]
sol_A3906 = [[0, 15, 30, 13], [0, 24, 3, 38, 12, 9, 28, 29], [0, 7, 8, 4, 16, 10, 27, 18], [
    0, 5, 26, 11], [0, 37, 31, 14, 35, 25, 33, 19, 2], [0, 6, 1, 36, 17, 23, 21, 22, 34, 32, 20]]
##########
A_n65_k09 = read("Instances/A-n65-k09.xml")


lam = 1.0
mu = 0.2
nu = 0.6
execute = 20
t = "A-n37-k06"
instance, demand = A_n37_k06
initiale = init_A3706
solution = sol_A3706

max_d = max_depth(instance)
v = voisins(KNN, instance)
# print(route_demand([0, 22, 13, 10, 6, 5, 33, 4, 7],demand)) # 3705
# print(route_demand([0, 21, 31, 19, 17, 13, 7, 26],demand)) # 3205
# print(route_demand([0, 10, 30, 25, 27, 5, 12],demand))  # 3305
"""
record = [[0, 16, 35, 25, 7], [0, 4, 18, 14, 36, 29, 24], [0, 31, 33, 5, 3, 1, 8, 6], [0, 27, 32, 15, 21, 34, 17], [0, 13, 30, 10, 26, 20], [0, 11, 12, 22, 23, 28, 2, 9, 19]]

for r in record:
    print(route_demand(r, demand))


print_current_sol(record,instance)
py.show()
"""
"""
init, reso = apply_heuristic(instance, demand, lam, mu,nu, relocation,max_d,v)
print(cost_sol(init,instance),cost_sol(reso,instance))
"""

costs = []
best = []
for i in range(execute):
    c_best = 100000
    init, reso = apply_heuristic(instance, demand, lam, mu,nu, relocation,max_d,v)
    c_sol = cost_sol(reso,instance)
    print(i,c_sol)
    costs.append(round(c_sol,3))
    if c_sol < c_best:
        best = reso
"""
namefile = "resultats/Heuristic_results/Values/"+t+"/results.txt"
"""
print(costs)
mean = 0
for c in costs:
    mean += c
print(mean/len(costs))
print(min(costs))
print(best)
"""
writef(namefile,'\n')
writef(namefile,'#################')
writef(namefile,'lambda = '+ str(lam))
writef(namefile,'mu = ' + str(mu))
writef(namefile,'nu = ' + str(nu))
writef(namefile,'execute = ' + str(execute))
writef(namefile,'')
writef(namefile,str(costs))
writef(namefile,'')
writef(namefile,'mean = ' + str(round(mean/len(costs),3)))
writef(namefile,'min = ' + str(min(costs)))
writef(namefile,'')
writef(namefile,str(best))
"""

"""
def total_execution(min_lam,max_lam,min_mu,max_mu,min_nu,max_nu,execute):
    for li in range(int(10*min_lam),int(10*max_lam)):
        for mi in range(int(10*min_mu),int(10*max_mu)):
            for ni in range(int(10*min_nu),int(10*max_nu)):
                c_best = 100000
                lam = 0.1*li
                mu = 0.1 * mi
                nu = 0.1*ni
                print(lam,mu,nu)
                costs = []
                best = []
                for i in range(execute):
                    init, reso = apply_heuristic(instance, demand, lam, mu,nu, relocation,max_d,v)
                    c_sol = cost_sol(reso,instance)
                    print(i,c_sol)
                    costs.append(round(c_sol,3))
                    if c_sol < c_best:
                        best = reso
                        c_best = c_sol

                namefile = "resultats/Heuristic_results/Values/"+t+"/stochastic_results.txt"

                mean = 0
                for c in costs:
                    mean += c
                print(mean/len(costs))
                print(min(costs))

                writef(namefile,'\n')
                writef(namefile,'#################')
                writef(namefile,'lambda = '+ str(lam))
                writef(namefile,'mu = ' + str(mu))
                writef(namefile,'nu = ' + str(nu))
                writef(namefile,'execute = ' + str(execute))
                writef(namefile,'')
                writef(namefile,str(costs))
                writef(namefile,'')
                writef(namefile,'mean = ' + str(round(mean/len(costs),3)))
                writef(namefile,'min = ' + str(min(costs)))
                writef(namefile,'gap = ' + str(round((1-(949/min(costs)))*100,3)))
                writef(namefile,'')
                writef(namefile,str(best))

total_execution(0.0,0.1,1.3,1.4,1.7,1.8,30)
"""
"""
sol_para = []

for li in range(1,21):
    for mj in range(21):
        for nk in range(21):
            lam = 0.1*li
            mu = 0.1*mj
            nu = 0.1*nk
            print("")
            print(lam,mu,nu)
            init,reso = apply_heuristic(instance,demand,lam,mu,nu,relocation,max_d,v)
            sol_para.append(((lam,mu,nu),(cost_sol(init,instance),cost_sol(reso,instance))))
            print(cost_sol(init,instance),cost_sol(reso,instance))
print(sol_para)

"""

"""
print_current_sol(initiale,instance)
py.title("Solution initiale " + t)
py.savefig("resultats/Heuristic_results/litterature_instances/"+t+"/initiale_"+t+".png")
py.close()


print_current_sol(solution,instance)
py.title("Solution obtenue pour " + t)
py.savefig("resultats/Heuristic_results/litterature_instances/"+t+"/solution_"+t+".png")
py.close()

E = common_edges(initiale,solution)

print_instance(instance)
print_edges(E,instance)
py.title("Arêtes communes pour " + t)
py.savefig("resultats/Heuristic_results/litterature_instances/"+t+"/commonEdges_"+t+".png")
py.close()
"""
"""
Eref = all_edges(initiale)
E = common_edges(initiale,solution)

n,rei,r_mean = all_ranks(E,initiale,instance)
nref,reiref,r_meanref = all_ranks(Eref,initiale,instance)
print(n)

r_mean.sort()
r_meanref.sort()

# print(reiref)

# print(rei)
print(r_meanref)
print(reiref)
print(r_mean)
print(rei)
"""

"""
instanceA = np.array(instance)
tri = Delaunay(instanceA)
print_instance(instance)
py.triplot(instanceA[:,0], instanceA[:,1], tri.simplices)
py.show()
"""

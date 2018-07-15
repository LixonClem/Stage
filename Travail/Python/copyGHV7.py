
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m
from lxml import etree
import os.path

import itertools as it
import time
import pickle

global ylim
global xlim
global clim
global Capacity
global instance_test
global lam
global Error
global KNN
global relocation
global mu
global nu

ylim = 200
xlim = 200
clim = 20

KNN = 30
relocation = 3


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
        x.append((float(abs.text)))
    for ord in tree.xpath("/instance/network/nodes/node/cy"):
        y.append((float(ord.text)))
    inst = [(x[i], y[i]) for i in range(len(x))]
    for dem in tree.xpath("/instance/requests/request/quantity"):
        demand.append((float(dem.text)))
    for c in tree.xpath("/instance/fleet/vehicle_profile/capacity"):
        C = float(c.text)
    return inst, demand,C


def writef(namefile, text):
    if not os.path.isfile(namefile):
        f = open(namefile, 'w')
        f.write(text + '\n')
        f.close()
    else:
        f = open(namefile, 'a')
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


def print_edges(edges, inst, col):
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


def verification(sol, demand):
    for r in sol:
        if route_demand(r, demand) > Capacity:
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


def all_edges(sol):
    E = []
    for r in sol:
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            E.append((pi, pj))
        E.append((r[len(r)-1], r[0]))
    return E


def fixed_alea(edges):
    tirage = edges.copy()
    n = len(edges)
    b = [False for i in range(n)]
    fe = []
    for i in range(n//2):
        alea = rd.randint(0, n-i-1)
        choice = tirage[alea]
        tirage.remove(choice)
        b[edges.index(choice)] = True
    for i in range(n):
        if b[i]:
            fe.append(edges[i])
    return fe


def fixed_0(edges):
    fe = []
    n = len(edges)
    for i in range(n//2):
        if 0 in edges[i]:
            fe.append(edges[i])
            edges.remove(edges[i])
    return fe


def adjacents(pi, fe):
    a = []
    for e in fe:
        if e[0] == pi and e[1] not in a:
            a.append(e[1])
        elif e[1] == pi and e[0] not in a:
            a.append(e[0])
    return a


def is_in_route(i, routes):
    booleen = False
    for r in routes:
        if i in r:
            booleen = True
    return booleen


def ignore_0(edges):
    n_edges = []
    for e in edges:
        if e[0] != 0:
            n_edges.append(e)
    return n_edges


def merge_edges(edges, l):
    a = l[0]
    b = l[-1]
    for e in edges:
        if e[0] == a:
            l = [e[1]] + l
            edges.remove(e)
            return l
        elif e[1] == a:
            l = [e[0]] + l
            edges.remove(e)
            return l
        elif e[0] == b:
            l = l + [e[1]]
            edges.remove(e)
            return l
        elif e[1] == b:
            l = l + [e[0]]
            edges.remove(e)
            return l
    return l


def destruction2(edges):
    r = []
    edges = ignore_0(edges)
    while edges != []:
        (a, b) = edges[0]
        edges.remove((a, b))
        l = [a, b]
        nl = merge_edges(edges, l)
        while nl != l:
            l = nl
            nl = merge_edges(edges, l)
        r.append(l)
    return r


def destruction(edges):
    edges.append((0, 0))
    r = []
    curr = [0]
    for i in range(len(edges)-1):
        e = edges[i]
        next_e = edges[i+1]
        if e[1] == 0:
            curr.append(e[0])
            r.append(curr)
            curr = [0]
        elif e[0] == 0 and e[1] != next_e[0]:
            curr.append(e[1])
            r.append(curr)
            curr = [0]
        elif e[0] != 0 and e[1] != next_e[0]:
            curr.append(e[0])
            curr.append(e[1])
            r.append(curr)
            curr = [0]
        elif e[0] != 0 and e[1] == next_e[0]:
            curr.append(e[0])
    return r


def complete(routes, inst,demand):
    for i in range(len(routes)):
        routes[i].insert(0, 0)
    while not verification(routes,demand):
        for r in routes:
            if route_demand(r,demand) > Capacity:

                routes.remove(r)
                d = 0
                i = 0
                nr1 = []
                while i<len(r) and d <= Capacity:
                    nr1.append(r[i])
                    i +=1 
                    d += demand[r[i]]
                    
                nr2 = [0] + r[r.index(r[i-1]):]
                
                routes.append(nr1)
                routes.append(nr2)
    for p in range(len(inst)):
        if not is_in_route(p, routes):
            routes.append([0, p])
    for i in range(len(routes)):
        routes[i].append(0)
    return routes


def permut(l):
    r = rd.randint(0, len(l)-1)
    i = 0
    for p in it.permutations(l):
        if i == r:
            return list(p)
        i += 1

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
        for j in range(i+1, len(inst)-1):
            if (i == j):
                savings[i][j] = 0
            else:
                savings[i][j] = distance(inst[i+1], inst[0]) + distance(inst[j+1], inst[0]) - lam*distance(inst[i+1], inst[j+1]) + mu*abs(
                    distance(inst[i+1], inst[0]) - distance(inst[j+1], inst[0])) + (nu*(demand[i+1] + demand[j+1])/d_bar)
    return savings


def compute_savings2(inst, demand, lam, mu, nu):
    savings = []
    d_bar = mean_demand(demand)
    for i in range(len(inst)-1):
        s = 0
        for j in range(i+1, len(inst)-1):
            if (i == j):
                savings.append([0, (i+1, j+1)])
            else:
                s = distance(inst[i+1], inst[0]) + distance(inst[j+1], inst[0]) - lam*distance(inst[i+1], inst[j+1]) + mu*abs(
                    distance(inst[i+1], inst[0]) - distance(inst[j+1], inst[0])) + (nu*(demand[i+1] + demand[j+1])/d_bar)
                if s >= 0:
                    savings.append([s, (i+1, j+1)])
    savings.sort()
    return savings


def max_savings(n, savings):
    cand = (-1, 0, 0)
    for i in range(n):
        for j in range(i+1, n):
            if cand[0] < 0 or savings[i][j] > cand[2]:
                cand = (i+1, j+1, savings[i][j])
    return cand


def max_savings2(savings, cpt):
    maximum = savings[-cpt]
    return maximum


def can_merge(i, r1, j, r2, demand):
    if r1 == r2:
        return -1
    elif (r1[1] == i and r2[len(r2)-2] == j and route_demand(r1, demand)+route_demand(r2, demand) <= Capacity):
        return 1
    elif (r1[len(r1)-2] == i and r2[1] == j and route_demand(r1, demand)+route_demand(r2, demand) <= Capacity):
        return 2
    else:
        return -1


def merge_routes(i, j, routes, inst, demand, detailed_cust):
    ir1, ir2 = detailed_cust[i-1], detailed_cust[j-1]
    r1, r2 = routes[ir1].copy(), routes[ir2].copy()
    mrge = can_merge(i, r1, j, r2, demand)
    new_road = []
    if mrge > 0:
        if mrge == 1:
            r1.pop()
            r2.remove(0)
            new_road = r1 + r2
        else:
            r2.pop()
            r1.remove(0)
            new_road = r2 + r1
        routes.append(new_road)
        routes[ir1] = []
        routes[ir2] = []
        detailed_cust[i-1] = len(routes)-1
        detailed_cust[j-1] = len(routes)-1
        for k in new_road:
            detailed_cust[k-1] = len(routes)-1
    

def ClarkeWright(routes, inst, demand, lam, mu, nu, detailed_cust):
    new_routes = []
    cpt = 1
    
    savings = compute_savings2(inst, demand, lam, mu, nu)
    [s, (i, j)] = max_savings2(savings, cpt)
    while s > 0 and cpt < len(savings):

        savings[-cpt][0] = 0
        cpt += 1
        
        merge_routes(i, j, routes, inst, demand, detailed_cust)
        [s, (i, j)] = max_savings2(savings, cpt)
    for i in range(len(routes)):
        if routes[i] != []:
            routes[i].pop()
            new_routes.append(routes[i])
    
    return new_routes

 ##################
# Cross - Exchange #
 ##################

# Code for the cross-exchange operator. Apply the operator for a certain edge.

 # Return the nearest route of the edge given


def another_routeCE(edge, voisins, routes, demand, fe):
    (a, b) = edge
    r1 = find_route(a, routes)
    adja = adjacents(a, fe)
    permut_voisins = permut(voisins[a])
    for i in permut_voisins:
        r2 = find_route(i, routes)
        adjpi = adjacents(r2[r2.index(i)-1], fe)
        # we verify that the future demand on the route won't exceed his capacity
        if r2 != r1 and i != 0 and len(adjpi) == 0 and len(adja) == 0 and route_demand(r1, demand)-demand[b]+demand[i] <= Capacity and route_demand(r2, demand)-demand[i]+demand[b] <= Capacity:
            return ((r1, r2), i)
    # error case, we haven't found a second route, so no modifications
    return ((r1, r1), -1)

# Apply the cross-exchange operator


def cross_exchange(edge, voisins, routes, inst, demand, fe):
    (a, b) = edge

    # compute the two routes considered, and the NN of the point we remove (a). v is a point
    (r1, r2), v = another_routeCE(edge, voisins, routes, demand, fe)
    if v < 0:
        return routes

    # copy of the current solution
    current_cand = [r1.copy(), r2.copy()]

    c_init = cost_sol(current_cand, inst)     # for a future comparison

    i_v = current_cand[1].index(v)
    i_a = current_cand[0].index(a)

    if i_v != 1:
        current_cand[0][i_a], current_cand[1][i_v -
                                              1] = current_cand[1][i_v-1], a
    else:
        current_cand[0][i_a], current_cand[1][i_v] = current_cand[1][i_v], a

    parcours_i = permut([i for i in range(len(r2)-1)])
    parcours_j = permut([j for j in range(len(r1)-1)])

    for i in parcours_i:
        if i != i_v-1:
            for j in parcours_j:
                if j != i_a-1:
                    p1 = current_cand[0][j+1]
                    p2 = current_cand[1][i+1]

                    current_cand[0][j+1], current_cand[1][i + 1] = p2, p1
                    adj1 = adjacents(p1, fe)
                    adj2 = adjacents(p2, fe)
                    if cost_sol(current_cand, inst) < c_init and len(adj1) == 0 and len(adj2) == 0 and route_demand(current_cand[0], demand) <= Capacity and route_demand(current_cand[1], demand) <= Capacity:
                        routes.remove(r1)
                        routes.remove(r2)
                        routes = routes + current_cand
                        return routes

                current_cand = [r1.copy(), r2.copy()]
    return routes

 ##################
# Ejection - Chain #
 ##################


def reject(route, routes, voisins, inst, demand):
    point = route[1]
    for i in voisins[point]:
        r = find_route(i, routes)
        if r != route and len(r) > 3 and route_demand(r, demand)+demand[point] <= Capacity:
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


def another_routeEC(a, voisins, routes, demand, inst, fe):
    r1 = find_route(a, routes)
    adja = adjacents(a, fe)
    permut_voisins = permut(voisins[a])
    for i in permut_voisins:
        r2 = find_route(i, routes)
        if r2 != r1 and i != 0 and len(adja) == 0 and route_demand(r2, demand)+demand[a] <= Capacity:
            return ((r1, r2), i)
    return (r1, r1), -1


# evalue a possible next edge.

def rd_point(edge, routes, inst):
    (a, b) = edge
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        i = rd.randint(0, 1)
        return edge[i]


def eval_cand(point, voisins, routes, inst, demand, fe):
    (r1, r2), v = another_routeEC(point, voisins, routes, demand, inst, fe)
    if v < 0:
        return Error
    i_v, i = r2.index(v), r1.index(point)
    return (saving(i, r1, i_v, r2, inst), (i, i_v), (r1, r2))

# return the best relocation for each point p in the route.
# Return the point to relocate and his neighbour considered.


def rd_cand(route, np, voisins, routes, inst, demand, fe):
    parcours = permut([i for i in range(len(route))])
    for i in parcours:
        p = route[i]
        if p != np:
            cp = rd_point((route[i-1], p), routes, inst)
            cand = eval_cand(cp, voisins, routes, inst, demand, fe)
            if cand[0] > 0:
                return cand
    return Error


def ejection_chain(l, point, voisins, routes, inst, demand, fe):
    S = 0  # global cost modification of the current solution

    s, I, R = eval_cand(point, voisins, routes, inst, demand, fe)

    if (s, I, R) == Error:
        return routes

    S += s
    relocated_cust = R[0][I[0]]

    # update the routes

    R[1].insert(I[1]+1, relocated_cust)
    R[0].remove(relocated_cust)

    for k in range(l-1):
        curr_route = R[1]
        s, I, R = rd_cand(curr_route, relocated_cust,
                          voisins, routes, inst, demand, fe)

        if (s, I, R) == Error:
            return routes
        S += s

        relocated_cust = R[0][I[0]]
        R[1].insert(I[1]+1, relocated_cust)
        R[0].remove(relocated_cust)

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
    best = 2e-5
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

#################
# Apprentissage #
#################


def is_edge_in(e, l):
    for i in l:
        if are_equal(e, i):
            return True
    return False

def unfeasable_edge(e,l):
    c1 = 0
    c2 = 0
    for ed in l:
        if e[0] == ed[0] or e[0] == ed[1]:
            c1 += 1
        elif e[1] == ed[0] or e[1] == ed[1]:
            c2 += 1
    return (c2>1 or c1 >1)

def normalize_solution(sol):
    for i in range(len(sol)):
        if sol[i][1] > sol[i][len(sol[i])-1]:
            sol[i].reverse()
            sol[i].pop()
            sol[i].insert(0, 0)
    sol.sort()
    return sol


def init_matrix(nb):
    return [[0 for i in range(nb)] for j in range(nb)]


def update_matrix(mat, sol):
    for r in sol:
        for i in range(len(r)-1):
            e = [r[i], r[i+1]]
            e.sort()
            mat[e[0]][e[1]] += 1
        mat[0][r[len(r)-1]] += 1
    return mat


def rd_generate(nb, instance, demand, initial):

    Base = []
    me = 0
    for j in range(nb):
        l = round(0.1*rd.randint(1, 20),1)
        m = round(0.1*rd.randint(1, 20),1)
        n = round(0.1*rd.randint(1, 20),1)
        detailed_cust = [0 for i in range(len(instance))]
        for r in range(len(initial)):
            for i in initial[r]:
                detailed_cust[i-1] = r
        routes = ClarkeWright(copy_sol(initial), instance,
                              demand, l, m, n, detailed_cust)
        for i in range(len(routes)):
            routes[i] = decross_route(routes[i].copy(), instance)
            routes[i] = LK(routes[i].copy(), instance)
        routes = normalize_solution(routes)
        me += cost_sol(routes, instance)
        Base.append((cost_sol(routes, instance), routes,(l,m,n)))
    Base.sort()

    return Base, [Base[0][0], Base[len(Base)-1][0], me/nb]


def all_generate(instance, demand):
    initial = init_routes(instance, demand)
    Base = []
    me = 0
    for li in range(1, 20):
        for mi in range(1, 20):
            for ni in range(1, 20):
                print(li, mi, ni)
                l = 0.1*li
                m = 0.1*mi
                n = 0.1*ni
                detailed_cust = [i for i in range(len(initial))]
                routes = ClarkeWright(
                    copy_sol(initial), instance, demand, l, m, n, detailed_cust)
                for i in range(len(routes)):
                    routes[i] = decross_route(routes[i].copy(), instance)
                    routes[i] = LK(routes[i].copy(), instance)
                routes = normalize_solution(routes)
                me += cost_sol(routes, instance)
                Base.append((cost_sol(routes, instance), routes))
    Base.sort()
    return Base, [Base[0][0], Base[len(Base)-1][0], me/8000]


def learning_set_quantity(Base, percent):
    ens = Base[:len(Base)//percent]
    ls = []
    for s in ens:
        ls.append(s[1])
    return ls


def learning_set_quality(Base, lim):
    ls = []
    i = 0
    while i<len(Base) and Base[i][0] <= lim:
        ls.append(Base[i][1])
        i += 1
    return ls


def learn(mat, ls):
    for sol in ls:
        update_matrix(mat, sol)
    return mat


def mat_info_req(lim, mat):
    ed_brut = []
    ed = []
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i][j] > lim:
                ed_brut.append((mat[i][j], i, j))
    ed_brut.sort()
    for e in ed_brut:
        ed.append((e[1], e[2]))
    return ed


def mat_info_rg(rg, mat):
    ed_brut = []
    ed = []
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i][j] > 0:
                ed_brut.append((mat[i][j], i, j))
    ed_brut.sort()
    ed_brut.reverse()
    for i in range(rg):
        e = ed_brut[i]
        ed.append((e[1], e[2]))
    return ed


def learning_results(crit,iterations, generate, instance, demand, initial):
    edges = []
    bigBase = []
    for lg in range(iterations):
        tps = time.time()
        Base, stat = rd_generate(generate, instance, demand,initial)
        bigBase += Base
        quality = (stat[1]-stat[0])/10 + stat[0]
        tps1 = time.time()
        print(tps1-tps)
        ls_qual = learning_set_quality(Base, quality)
        mat_qual = init_matrix(len(instance))
        mat_qual = learn(mat_qual, ls_qual)

        e_qual = mat_info_rg(int(len(demand)*crit), mat_qual)
        #e_qual = mat_info_req(int(len(ls_qual)*crit),mat_qual)
        for e in e_qual:
            if not is_edge_in(e, edges) and not unfeasable_edge(e,edges):
                edges.append(e)
    bigBase.sort()
    param = []
    for b in bigBase:
        param.append(b[2])
    return edges,param

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
    return distance(i, j)*(1 + 0.1*p)


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


def bad_edge(b, p, routes, inst, fixed):
    cand = [0, (0, 0)]
    for r in routes:
        G = gravity_center(r, inst)
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            b_ij = b(inst[pi], inst[pj], G, p[pi][pj])
            if b_ij > cand[0] and pi != 0 and pj != 0 and (pi, pj) not in fixed and (pj, pi) not in fixed:
                cand[0] = b_ij
                cand[1] = (pi, pj)
    return cand


def core_heuristic(initial_routes, inst, demand, lam, mu, nu, l, max_d, v):
    tps1 = time.time()
    B = [penalization_function(1, 0, 0, max_d), penalization_function(1, 1, 0, max_d), penalization_function(
        1, 0, 1, max_d), penalization_function(1, 1, 1, max_d), penalization_function(0, 1, 0, max_d), penalization_function(0, 1, 1, max_d)]

    b_i = 0
    b = B[b_i]
    p = [[0 for j in range(len(inst))] for i in range(len(inst))]
    N = 0  # laps without improvement
    gs = 0  # laps for last improvement
    detailed_cust = [0 for i in range(len(inst))]
    for r in range(len(initial_routes)):
        for i in initial_routes[r]:
            detailed_cust[i-1] = r
    initial_routes = ClarkeWright(
        initial_routes, inst, demand, lam, mu, nu, detailed_cust)

    routes = copy_sol(initial_routes)
    routes2 = copy_sol(routes)
    fixed_edges = []
    c_init = cost_sol(routes, inst)
    print(c_init)
    tps2 = time.time()
    tpsGS = time.time()
    tpsCH = time.time()
    while tps2-tps1 < len(demand)/3:

        # find the worst edge
        worst = bad_edge(b, p, routes, inst, fixed_edges)[1]

        p[worst[0]][worst[1]] += 1
        p[worst[1]][worst[0]] += 1

        # apply ejection-chain
        cp = rd_point(worst, routes, inst)

        routes = ejection_chain(l, cp, v, routes, inst, demand, fixed_edges)
        for i in routes:
            if len(i)==2:
                routes = reject(i, routes, v, inst, demand)
        for i in range(len(routes)):
            if len(routes[i])>=3:
                routes[i] = decross_route(routes[i].copy(), inst)
            routes[i] = LK(routes[i], inst)
        # apply cross-exchange

        routes = cross_exchange(worst, v, routes, inst, demand, fixed_edges)

        # apply LK
        for i in range(len(routes)):
            routes[i] = decross_route(routes[i].copy(), inst)
            routes[i] = LK(routes[i], inst)

        c_final = cost_sol(routes, inst)

        if c_final < c_init:
            routes2 = copy_sol(routes)  # new optimum
            #fixed_edges = fixed(all_edges(routes2))

            gs = 0
            N = 0
            c_init = cost_sol(routes2, inst)
            print(tps2-tps1, c_init)
            tps1 = time.time()
            tpsCH = time.time()
            tpsGS = time.time()

        if tps2-tpsGS > len(demand)/50:
            # return to the last best solution, for gs iterations
            
            routes = copy_sol(routes2)
            gs = 0
            tpsGS = time.time()

        if tps2-tpsCH > len(demand)/100:
            tpsCH = time.time()
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


        tps2 = time.time()

    for i in (routes2):
        if len(i) == 2:
            routes2 = reject(i, routes2, v, inst, demand)
        if len(i) == 1:
            routes2.remove(i)

    for i in range(len(routes2)):
        routes2[i] = decross_route(routes2[i].copy(), inst)
        routes2[i] = LK(routes2[i], inst)

    return initial_routes, routes2


def apply_heuristic(instance, demand, l):
    # compute global variables
    namefile ="resultats/Heuristic_results/Values/all/golden3.txt"
    all_sol = []
    tps_deb = time.time()
    max_d = max_depth(instance)
    v = voisins(KNN, instance)
    initial = init_routes(instance,demand)
    edges, param = learning_results(0.5,5,50,instance,demand,initial)
    initial_routes = complete(destruction2(ignore_0(edges)),instance,demand)
    tps_learn = time.time()
    
    
    writef(namefile,'Time = '+ str(tps_learn-tps_deb))
    
    new_base = []
    costs = 0
    edges = []
    best_sol = cost_sol(initial_routes,instance)
    for i in range(10):
        print(i)

        if new_base==[]:
            for j in range(10):
                print(j)
                (lam,mu,nu) = param[j]
                
                init, sol = core_heuristic(
                    copy_sol(initial_routes), instance, demand, lam, mu, nu, l, max_d, v)
                c_sol = cost_sol(sol, instance)
                if c_sol < best_sol:
                    best_sol=c_sol
                    mat_qual = init_matrix(len(instance))
                    base = [sol]
                    mat_qual = learn(mat_qual, base)
                    edges = []
                    e_qual = mat_info_rg(int(len(demand)*0.8), mat_qual)
                    for e in e_qual:
                        if not is_edge_in(e, edges) and not unfeasable_edge(e,edges):
                            edges.append(e)
                    initial_routes = complete(destruction2(ignore_0(edges)),instance,demand)


                new_base.append(sol)
                all_sol.append((c_sol,sol))
        

        else:
            print("learn")
            edges =[]
            mat_qual = init_matrix(len(instance))
            mat_qual = learn(mat_qual, new_base)
            e_qual = mat_info_rg(int(len(demand)/2), mat_qual)
            for e in e_qual:
                if not is_edge_in(e, edges) and not unfeasable_edge(e,edges):
                    edges.append(e)
            initial_routes = complete(destruction2(ignore_0(edges)),instance,demand)
            edges, param = learning_results(0.5-i/20,5,50,instance,demand,initial_routes)
            initial_routes = complete(destruction2(ignore_0(edges)),instance,demand)
            best_sol = cost_sol(initial_routes,instance)
            new_base = []
            for j in range(10):
                print(j)
                (lam,mu,nu) = param[j]
                
                init, sol = core_heuristic(
                    copy_sol(initial_routes), instance, demand, lam, mu, nu, l, max_d, v)
                c_sol = cost_sol(sol, instance)
                costs += c_sol
                c_init = cost_sol(init, instance)

                if c_sol < best_sol:
                    best_sol=c_sol
                    base = [sol]
                    mat_qual = init_matrix(len(instance))
                    mat_qual = learn(mat_qual, base)
                    e_qual = mat_info_rg(int(len(demand)*0.8), mat_qual)
                    edges =[]
                    for e in e_qual:
                        if not is_edge_in(e, edges) and not unfeasable_edge(e,edges):
                            edges.append(e)
                    initial_routes = complete(destruction2(ignore_0(edges)),instance,demand)

                new_base.append(sol)
                all_sol.append((c_sol,sol))
    
    all_sol.sort()
    tps_fin = time.time()
    print(tps_fin-tps_deb)
    costs = 0
    for i in range(10):
        c_sol,sol = all_sol[i]
        costs += c_sol
        
        writef(namefile,'')
        writef(namefile,'res = ' + str(round(c_sol,3)))
        writef(namefile,'solution = ' + str(sol))

    writef(namefile,'')
    writef(namefile,'Mean = ' + str(costs/100))
    writef(namefile,'Execution = ' + str(tps_fin-tps_deb))
    writef(namefile,'')
    
 ###########
# Solutions #
 ###########


def are_equal(edge1, edge2):
    return (edge1 == edge2) or (edge1[1] == edge2[0] and edge1[0] == edge2[1])


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
        if i not in E and (i[1], i[0]) not in E:
            E_init.append(i)
    for j in E2:
        if j not in E and (j[1], j[0]) not in E:
            E_final.append(j)
    return E, E_init, E_final



t = "B-n45-k06"
instance,demand,Capacity = read("Instances/"+t+".xml")

#record = [[0, 7, 25, 35, 16], [0, 27, 32, 15, 30, 13], [0, 24, 29, 36, 6, 14], [0, 4, 10, 11, 12, 22, 23, 28, 2, 33], [0, 20, 8, 5, 3, 1, 34, 17], [0, 18, 31, 19, 9, 21, 26]]
#record1 = [[0, 27, 32, 15, 30, 13], [0, 10, 11, 12, 22, 23, 28, 2, 33], [0, 7, 25, 35, 16], [0, 24, 29, 36, 6, 14], [0, 18, 17, 34, 1, 3, 5, 8, 20], [0, 31, 19, 9, 21, 26, 4]]
#record = [[0, 55, 29, 62, 39, 51, 17], [0, 45, 61, 42, 38, 2, 41, 16, 50, 60], [0, 21, 25, 52, 24, 13, 12, 1, 33], [0, 49, 4, 3, 36, 35, 37, 30], [
#   0, 47, 34, 31, 26, 6, 64, 46], [0, 28, 23, 57, 48, 54, 63, 11, 7], [0, 44, 59, 40, 58, 20, 32], [0, 5, 53, 56, 10, 8, 19, 18], [0, 43, 27, 14, 9, 22, 15]]
#record3205 = [[0,21, 31, 19, 17, 13, 7, 26],[0,12, 1, 16, 30],[0,27, 24],[0,29, 18, 8, 9, 22, 15, 10, 25, 5, 20],[0,14, 28, 11, 4, 23, 3, 2, 6]]
#record3305 = [[0, 15, 17, 9, 3, 16, 29],[0, 12, 5, 26, 7, 8, 13, 32, 2],[0, 20, 4, 27, 25, 30, 10],[0, 23, 28, 18, 22],[0, 24, 6, 19, 14, 21, 1, 31, 11]]
#record = normalize_solution(record)
#record1 = normalize_solution(record1)
#best = [[0, 30, 37, 35, 36, 3, 4, 49], [0, 7, 11, 63, 54, 48, 57, 23, 28], [0, 15, 22, 9, 14, 27, 43], [0, 47, 34, 31, 26, 6, 64, 46], [0, 53, 44, 56,
 #                                                                                                                                       25, 21], [0, 8, 10, 24, 13, 12, 1, 33], [0, 17, 51, 39, 62, 29, 55], [0, 5, 32, 20, 58, 40, 59, 52, 19, 18], [0, 60, 50, 16, 41, 2, 38, 42, 61, 45]]
#best = normalize_solution(best)
#record = [[0,17, 24, 35, 37, 34, 26, 11, 8], [0, 2, 22, 3, 7, 16, 32, 10], [0, 21, 30, 13, 28, 27, 36, 6], [0, 14, 19, 25, 33, 12, 18, 4], [0, 9, 38, 15, 5 ,29, 20, 23, 1 ,31]]
#record = [[0, 1, 7, 21, 40],[0, 10, 63, 11, 24, 6, 23],[0, 13, 74, 60, 39, 3, 77, 51] ,[0, 17, 31, 27, 59, 5, 44, 12, 62] ,[0, 29, 20, 75, 57, 19, 26, 35, 65, 69, 56, 47, 15, 33, 64] , [0, 30, 78, 61, 16, 43, 68, 8, 37, 2, 34 ],[0, 38, 72, 54, 9, 55, 41, 25, 46 ],[0, 42, 53, 66, 67, 36, 73, 49 ],[0, 52, 28, 79, 18, 48, 14, 71 ],[0, 58, 32, 4, 22, 45, 50, 76, 70] ]
#solution = [[0,37, 11, 27, 22, 5, 7], [0,10 ,30 ,29 ,34 ,19, 18], [0,20, 32, 15, 13, 36, 17, 2, 14], [0,28, 31, 6, 25, 16, 4, 1, 3, 12, 26, 21], [0,24 ,33, 35, 23, 8 ,9]]
#print(cost_sol(solution,instance))
"""
initial_solution = init_routes(instance, demand)


detailed_cust = [0 for i in range(len(instance))]
for r in range(len(initial_solution)):
    for i in initial_solution[r]:
        detailed_cust[i-1] = r


initial_solution = ClarkeWright(initial_solution,instance, demand, 0, 1, 1.5,detailed_cust)

for i in range(len(initial_solution)):
    initial_solution[i] = decross_route(initial_solution[i].copy(), instance)
    initial_solution[i] = LK(initial_solution[i].copy(), instance)
print(cost_sol(initial_solution,instance))

ae = all_edges(initial_solution)
aer = all_edges(record)
ce = common_edges(initial_solution,record)
print_instance(instance)

print_edges(aer,instance,'green')
py.show()
"""

#apply_heuristic(instance, demand, relocation)

allinstances = os.listdir('toExecute')
allinstances.sort()
print(allinstances)

for fileinstance in allinstances:
    namefile = "resultats/Heuristic_results/Values/all/golden3.txt"
    print(fileinstance)
    writef(namefile,'Instance : ' + fileinstance)
    instance,demand,Capacity = read('toExecute/'+fileinstance)
    print(Capacity)
    print("")
    apply_heuristic(instance, demand, relocation)

s = 0
"""
Gen = 50
initial = init_routes(instance,demand)
all_results = [[[],[],[]] for i in range(9)]
for lg in range(10):
    print(lg)
    edges =[]
    base,stat = rd_generate(Gen,instance,demand,initial)
    param = []
    for b in base:
        param.append([b[2]])
    namefile = "resultats/Heuristic_results/Values/tests/seuilP10104.txt"
    writef(namefile,'Base : '+str(lg))
    writef(namefile,''+str(param))
    quality = (stat[1]-stat[0])/10 + stat[0]
    ls_qual = learning_set_quality(base,quality)
    all_pre = []
    mat_qual = init_matrix(len(instance))
    mat_qual = learn(mat_qual,ls_qual)
    crit_nb = 0
    for crit in [0.2,0.25,0.33,0.4,0.5,0.6,0.67,0.75,0.8]:
        edges =[]
        e_c = mat_info_req(int(len(ls_qual)*crit),mat_qual)
        for e in e_c:
            if not is_edge_in(e, edges) and not unfeasable_edge(e,edges):
                edges.append(e)

        res = []
        for i in range(10):
            (la,mu,nu) = param[i][0]
            initial_routes = complete(destruction2(ignore_0(edges)),instance,demand)
            detailed_cust = [0 for k in range(len(instance))]
            for r in range(len(initial_routes)):
                for j in initial_routes[r]:
                    detailed_cust[j-1] = r
            sol = ClarkeWright(initial_routes,instance,demand,la,mu,nu,detailed_cust)
            res.append(cost_sol(sol,instance))
        all_results[crit_nb][0] += [res[0]]
        all_results[crit_nb][1] += res[:5]
        all_results[crit_nb][2] += res
        crit_nb += 1
crit_nb = 1
for i in all_results:
    namefile = "resultats/Heuristic_results/Values/tests/seuilP10104.txt"
    writef(namefile,'critère '+str(crit_nb))
    for j in i:
        writef(namefile,'')
        writef(namefile,str(j))
    crit_nb += 1




        s += len(ls_qual)
        n1 = len(e_c)
        inf = []
        inf_dist = []
        infopt = []
        infopt_dist = []
        pre = []
        accuracy = 0

        
        for i in e_c:

            if unfeasable_edge(i,pre) and is_edge_in(i,true_edges):
                n4 +=1
                infopt.append(i)
            elif unfeasable_edge(i,pre):
                n3 += 1
                inf.append(i)
            elif is_edge_in(i,true_edges) :
                pre.append(i)
                n2 += 1
                accuracy += 1
            else:
                pre.append(i)

        all_pre.append(pre)

        tps_fin = time.time()
        
        for e in inf:
            inf_dist.append(distance(instance[e[0]],instance[e[1]]))

        for e in infopt:
            infopt_dist.append(distance(instance[e[0]],instance[e[1]]))
        
        print(n1,n2,n3,n4)
        print(inf_dist)
        print(infopt_dist)
    print(s)"""
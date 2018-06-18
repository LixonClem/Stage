
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m
from lxml import etree
import os.path
import itertools as it

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
global mu
global nu

ylim = 200
xlim = 200
clim = 20
nb_cust = 100
Capacity = 100
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
        alea = rd.randint(0,n-i-1)
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


def adjacents(pi,fe):
    a = []
    for e in fe:
        if e[0]==pi and e[1] not in a:
            a.append(e[1])
        elif e[1]==pi and e[0] not in a:
            a.append(e[0])
    return a


def is_in_route(i,routes):
    booleen = False
    for r in routes:
        if i in r:
            booleen=True
    return booleen

def destruction(edges):
    edges.append((0,0))
    r = []
    curr = [0]
    for i in range(len(edges)-1):
        e = edges[i]
        next_e = edges[i+1]
        if e[1]==0:
            curr.append(e[0])
            r.append(curr)
            curr = [0]
        elif e[0]==0 and e[1]!=next_e[0]:
            curr.append(e[1])
            r.append(curr)
            curr= [0]
        elif e[0]!=0 and e[1]!=next_e[0]:
            curr.append(e[0])
            curr.append(e[1])
            r.append(curr)
            curr = [0]
        elif e[0]!=0 and e[1]==next_e[0]:
            curr.append(e[0])
    return r

def complete(routes,inst):
    for p in range(len(inst)):
        if not is_in_route(p,routes):
            routes.append([0,p])
    for i in range(len(routes)):
        routes[i].append(0)
    return routes

def permut(l):
    r = rd.randint(0,len(l)-1)
    i = 0
    for p in it.permutations(l):
        if i ==r :
            return list(p)
        i+=1
            
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
    n = len(new_road)





def ClarkeWright(routes,inst, demand, lam, mu, nu):
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
    (a, b) = edge

    # compute the two routes considered, and the NN of the point we remove (a). v is a point
    (r1, r2), v = another_routeCE(edge, voisins, routes, demand,fe)
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

    for i in range(len(r2)-1):
        if i != i_v-1:
            for j in range(len(r1)-1):
                if j != i_a-1:
                    p1 = current_cand[0][j+1]
                    p2 = current_cand[1][i+1]

                    current_cand[0][j+1], current_cand[1][i + 1] = p2, p1
                    adj1 = adjacents(p1,fe)
                    adj2 = adjacents(p2,fe)
                    if cost_sol(current_cand, inst) < c_init and len(adj1)==0 and len(adj2)==0 and route_demand(current_cand[0], demand) <= Capacity and route_demand(current_cand[1], demand) <= Capacity:
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
        if r != route and len(r)>3 and route_demand(r, demand)+demand[point] <= Capacity:
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
            S.append(eval_cand(cp, voisins, routes, inst, demand,fe))

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


def apply_heuristic(initial_solution,inst, demand, lam, mu, nu, l,max_d,v):

    # compute global variables


    B = [penalization_function(1, 0, 0, max_d), penalization_function(1, 1, 0, max_d), penalization_function(
        1, 0, 1, max_d), penalization_function(1, 1, 1, max_d), penalization_function(0, 1, 0, max_d), penalization_function(0, 1, 1, max_d)]
    b_i = 0
    b = B[b_i]

    p = [[0 for j in range(len(inst))] for i in range(len(inst))]


    N = 0  # laps without improvement
    gs = 0  # laps for last improvement
    time = 0

    c_init = cost_sol(initial_solution, inst)
    

    routes2 = copy_sol(initial_solution)
    routes = copy_sol(initial_solution)
    fixed_edges = []
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
            #fixed_edges = fixed(all_edges(routes2))


            gs = 0
            N = 0
            c_init = cost_sol(routes2, inst)
            time = 0

        if gs > 20:
            # return to the last global solution, for gs iterations
            routes = copy_sol(routes2)
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

    return routes2

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

A_n37_k06 = read("Instances/A-n37-k06.xml")



lam = 0.8
mu = 0.0
nu = 1.0
t = "A-n37-k06"
instance, demand = A_n37_k06


max_d = max_depth(instance)
v = voisins(KNN, instance)
"""
record = [[0, 16, 35, 25, 7], [0, 4, 18, 14, 36, 29, 24], [0, 31, 33, 5, 3, 1, 8, 6], [0, 27, 32, 15, 21, 34, 17], [0, 13, 30, 10, 26, 20], [0, 11, 12, 22, 23, 28, 2, 9, 19]]

initial_solution = init_routes(instance, demand)
initial_solution = ClarkeWright(initial_solution,instance, demand, lam, mu, nu)
for i in range(len(initial_solution)):
    initial_solution[i] = decross_route(initial_solution[i].copy(), instance)
    initial_solution[i] = LK(initial_solution[i].copy(), instance)

init, reso = apply_heuristic(
    instance, demand, lam, mu, nu, relocation, max_d, v)
print(cost_sol(init, instance), cost_sol(reso, instance))

"""
"""
costs = []
best = []
c_best = 100000
for i in range(20):
    
    init, reso = apply_heuristic(instance, demand, lam, mu,nu, relocation,max_d,v)
    c_sol = cost_sol(reso,instance)
    print(i,c_sol)
    costs.append(round(c_sol,3))
    if c_sol < c_best:
        best = reso
        c_best  = c_sol

namefile = "resultats/Heuristic_results/Values/"+t+"/results.txt"

print(costs)
mean = 0
for c in costs:
    mean += c
print(mean/len(costs))
print(min(costs))
print(best)
"""
"""
def total_execution(min_lam,max_lam,min_mu,max_mu,min_nu,max_nu):
    deja_com = []
    for li in range(int(10*min_lam),int(10*max_lam)):
        for mi in range(int(10*min_mu),int(10*max_mu)):
            for ni in range(int(10*min_nu),int(10*max_nu)):

                lam = 0.1*li
                mu = 0.1 * mi
                nu = 0.1*ni
                print(lam,mu,nu)
                routes = init_routes(instance,demand)
                initial_solution = ClarkeWright(routes,instance, demand, lam, mu, nu)
                for i in range(len(initial_solution)):
                    initial_solution[i] = decross_route(initial_solution[i].copy(), instance)
                    initial_solution[i] = LK(initial_solution[i].copy(), instance)
                
                if round(cost_sol(initial_solution,instance),3) not in deja_com:
                    deja_com.append(round(cost_sol(initial_solution,instance),3))

                    init, reso = apply_heuristic(instance, demand, lam, mu,nu, relocation,max_d,v)
                    c_sol = cost_sol(reso,instance)
                    c_init = cost_sol(initial_solution,instance)
                    

                    namefile = "resultats/Heuristic_results/Values/"+t+"/results_det-De.txt"


                    writef(namefile,'\n')
                    writef(namefile,'#################')
                    writef(namefile,'lambda = '+ str(lam))
                    writef(namefile,'mu = ' + str(mu))
                    writef(namefile,'nu = ' + str(nu))
                    writef(namefile,'')
                    writef(namefile,'init = ' + str(round(c_init,3)))
                    writef(namefile,'det = ' + str(round(c_sol,3)))
                    writef(namefile,'gap = ' + str(round((1-949/c_sol)*100,3)))
                    writef(namefile,'')
                    writef(namefile,'solution = ' + str(reso))
             
                else:
                    print("deja calculé !")

total_execution(0.0,2.0,0.0,2.0,0.0,2.0)
"""

def improved_heuristic(instance,demand,lam,mu,nu):
    record = [[0, 7, 25, 35, 16], [0, 27, 32, 15, 30, 13], [0, 24, 29, 36, 6, 14], [0, 4, 10, 11, 12, 22, 23, 28, 2, 33], [0, 20, 8, 5, 3, 1, 34, 17], [0, 18, 31, 19, 9, 21, 26]]
    routes = init_routes(instance,demand)
    initial_solution = ClarkeWright(routes,instance, demand, lam, mu, nu)
    for i in range(len(initial_solution)):
        initial_solution[i] = decross_route(initial_solution[i].copy(), instance)
        initial_solution[i] = LK(initial_solution[i].copy(), instance)

    last_solution = apply_heuristic(initial_solution,instance,demand,lam,mu,nu,relocation,max_d,v)

    print(cost_sol(initial_solution,instance))
    print(cost_sol(last_solution,instance))
    for i in range(10):
        print(i)
        e,ei,ef = common_edges(initial_solution,last_solution)
        fixed_edges = e
        initial_solution = complete(destruction(fixed_edges),instance)
        initial_solution = ClarkeWright(initial_solution,instance,demand,lam,mu/2,nu/2)
        for i in range(len(routes)):
            initial_solution[i] = decross_route(initial_solution[i].copy(), instance)
            initial_solution[i] = LK(initial_solution[i].copy(), instance)
        
        last_solution = apply_heuristic(initial_solution,instance,demand,lam,mu,nu,relocation,max_d,v)
        print(cost_sol(initial_solution,instance))
        print(cost_sol(last_solution,instance))

#improved_heuristic(instance,demand,lam,mu,nu)


def CW_iteration(instance,demand,lam,mu,nu):
    record = [[0, 7, 25, 35, 16], [0, 27, 32, 15, 30, 13], [0, 24, 29, 36, 6, 14], [0, 4, 10, 11, 12, 22, 23, 28, 2, 33], [0, 20, 8, 5, 3, 1, 34, 17], [0, 18, 31, 19, 9, 21, 26]]
    routes = init_routes(instance,demand)
    initial_solution = ClarkeWright(routes,instance, demand, lam, mu, nu)
    for i in range(len(initial_solution)):
        initial_solution[i] = decross_route(initial_solution[i].copy(), instance)
        initial_solution[i] = LK(initial_solution[i].copy(), instance)
    
    print(cost_sol(initial_solution,instance))

    for i in range(100):
        lam /= 1.2
        mu /= 1.1
        nu /= 1.1
        print(i)
        e,ei,ef = common_edges(initial_solution,record)
        fixed_edges = e
        initial_solution = complete(destruction(fixed_edges),instance)
        initial_solution = ClarkeWright(initial_solution,instance,demand,lam,mu,nu)
        for i in range(len(initial_solution)):
            initial_solution[i] = decross_route(initial_solution[i].copy(), instance)
            initial_solution[i] = LK(initial_solution[i].copy(), instance)

        print(cost_sol(initial_solution,instance))

CW_iteration(instance,demand,lam,mu,nu)
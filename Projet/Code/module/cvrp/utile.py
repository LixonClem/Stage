# This module contains functions that are used in other modules

import random as rd
import math as m
import itertools as it
import cvrp.tests as tests
import cvrp.route as route

# Compute the euclidean distance between two points p1 and p2


def distance(p1, p2):
    return m.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# Compute the k nearest neighbors (kNN) for each customer


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

# Compute the maximal depth of the instance given


def max_depth(inst):
    d = 0
    for i in inst:
        di = distance(i, (0, 0))
        if di > d:
            d = di
    return d

# Compute the mean of demands of the instance


def mean_demand(demand):
    n = len(demand)
    d = 0
    for i in demand:
        d += i
    return d/(n-1)

# Return n*nb edges randomly


def fixed_alea(edges, nb):
    tirage = edges.copy()
    n = len(edges)
    b = [False for i in range(n)]
    fe = []
    for i in range(int(n*nb)):
        alea = rd.randint(0, n-i-1)
        choice = tirage[alea]
        tirage.remove(choice)
        b[edges.index(choice)] = True
    for i in range(n):
        if b[i]:
            fe.append(edges[i])
    return fe

# Return a permutation of the list l


def permut(l):
    r = rd.randint(0, len(l)-1)
    i = 0
    for p in it.permutations(l):
        if i == r:
            return list(p)
        i += 1

# Return a point of the edge (a,b) randomly
# (except if a or b is the depot)


def rd_point(edge, routes, inst):
    (a, b) = edge
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        i = rd.randint(0, 1)
        return edge[i]

# Return a list of edges which don't contain the depot


def ignore_0(edges):
    n_edges = []
    for e in edges:
        if e[0] != 0 and e[1] != 0:
            n_edges.append(e)
    return n_edges

# For a route l = [a,...,b], if an edge (x,a) or (b,x) exists in edges
# then return the new route l = [x,a,...,b] or l = [a,...,b,x]


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

# Return a partial initial solution by merging edges


def destruction(edges):
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

# Complete a partial solution by adding customers which haven't route.
# And verify if the solution given is correct


def complete(routes, inst, demand, capacity):
    for i in range(len(routes)):
        routes[i].insert(0, 0)
    while not tests.verification(routes, demand, capacity):

        for ri in routes:

            if route.route_demand(ri, demand) > capacity:

                routes.remove(ri)
                d = 0
                i = 0
                nr1 = []
                while i < len(ri) and d <= capacity:

                    nr1.append(ri[i])
                    i += 1
                    d += demand[ri[i]]

                nr2 = [0] + ri[ri.index(ri[i-1]):]

                routes.append(nr1)
                routes.append(nr2)
    for p in range(len(inst)):
        if not tests.is_in_route(p, routes):
            routes.append([0, p])
    for i in range(len(routes)):
        routes[i].append(0)
    return routes

# Return the list of neighbors of pi which are fixed by fixed_edges


def fixed_adjacents(pi, fixed_edges):
    a = []
    for e in fixed_edges:
        if e[0] == pi and e[1] not in a:
            a.append(e[1])
        elif e[1] == pi and e[0] not in a:
            a.append(e[0])
    return a

# Verify if two edges are equal


def are_equal(edge1, edge2):
    return (edge1 == edge2) or (edge1[1] == edge2[0] and edge1[0] == edge2[1])

# Verify if a certain edge is in a list of edges


def is_edge_in(e, l):
    for i in l:
        if are_equal(e, i):
            return True
    return False

# Verify if by adding e to l, a customer
# won't have more than 2 neighbors


def unfeasable_edge(e, l):
    c1 = 0
    c2 = 0
    for ed in l:
        if e[0] == ed[0] or e[0] == ed[1]:
            c1 += 1
        elif e[1] == ed[0] or e[1] == ed[1]:
            c2 += 1
    return (c2 > 1 or c1 > 1)

# Return the list of edges in common between two solutions


def common_edges(sol1, sol2):
    E1 = route.all_edges(sol1)
    E2 = route.all_edges(sol2)
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

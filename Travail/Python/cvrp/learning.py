# # Ce module rassemble les fonctions utilisées lors de l'apprentissage

import cvrp.route as route
import cvrp.utile as utile
import cvrp.const as const
import cvrp.tests as tests
import random as rd
import cvrp.ClarkeWright as CW
import cvrp.linKernighan as LK
import time

def all_edges(sol):
    E = []
    for r in sol:
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            E.append((pi, pj))
        E.append((r[len(r)-1], r[0]))
    return E

def ignore_0(edges):
    n_edges = []
    for e in edges:
        if e[0] != 0 and e[1] != 0:
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

def complete(routes, inst, demand,capacity):
    for i in range(len(routes)):
        routes[i].insert(0, 0)
    while not tests.verification(routes, demand,capacity):

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

def adjacents(pi, fe):
    a = []
    for e in fe:
        if e[0] == pi and e[1] not in a:
            a.append(e[1])
        elif e[1] == pi and e[0] not in a:
            a.append(e[0])
    return a

def are_equal(edge1, edge2):
    return (edge1 == edge2) or (edge1[1] == edge2[0] and edge1[0] == edge2[1])

def is_edge_in(e, l):
    for i in l:
        if are_equal(e, i):
            return True
    return False


def unfeasable_edge(e, l):
    c1 = 0
    c2 = 0
    for ed in l:
        if e[0] == ed[0] or e[0] == ed[1]:
            c1 += 1
        elif e[1] == ed[0] or e[1] == ed[1]:
            c2 += 1
    return (c2 > 1 or c1 > 1)


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


def rd_generate(nb, instance, demand,capacity, initial):

    Base = []
    me = 0
    for j in range(nb):
        l = round(0.1*rd.randint(1, 20), 1)
        m = round(0.1*rd.randint(1, 20), 1)
        n = round(0.1*rd.randint(1, 20), 1)
        detailed_cust = [0 for i in range(len(instance))]
        for r in range(len(initial)):
            for i in initial[r]:
                detailed_cust[i-1] = r
        routes = CW.ClarkeWright(route.copy_sol(initial), instance,
                              demand,capacity, l, m, n, detailed_cust)
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i].copy(), instance)
        routes = normalize_solution(routes)
        me += route.cost_sol(routes, instance,const.quality_cost)
        Base.append((route.cost_sol(routes, instance,const.quality_cost), routes, (l, m, n)))
    Base.sort()

    return Base, [Base[0][0], Base[len(Base)-1][0], me/nb]

def learning_set_quantity(Base, percent):
    ens = Base[:len(Base)//percent]
    ls = []
    for s in ens:
        ls.append(s[1])
    return ls


def learning_set_quality(Base, lim):
    ls = []
    i = 0
    while i < len(Base) and Base[i][0] <= lim:
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


def learning_results(crit, iterations, generate, instance, demand,capacity, initial):
    edges = []
    bigBase = []
    for lg in range(iterations):
        tps = time.time()
        Base, stat = rd_generate(generate, instance, demand,capacity, initial)
        bigBase += Base
        quality = (stat[1]-stat[0])/10 + stat[0]
        tps1 = time.time()
        
        ls_qual = learning_set_quality(Base, quality)
        mat_qual = init_matrix(len(instance))
        mat_qual = learn(mat_qual, ls_qual)

        e_qual = mat_info_rg(int(len(demand)*crit), mat_qual)
        for e in e_qual:
            if not is_edge_in(e, edges) and not unfeasable_edge(e, edges):
                edges.append(e)
    bigBase.sort()
    param = []
    for b in bigBase:
        param.append(b[2])
    return edges, param

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
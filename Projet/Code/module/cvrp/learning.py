# This module contains functions used for the learning phase

#import cvrp.route as route
#import cvrp.utile as utile
import cvrp.const as const

import random as rd
import cvrp.ClarkeWright as CW
import cvrp.linKernighan as LK
import time




# Initialise a matrix of size nb*nb
def init_matrix(nb):
    return [[0 for i in range(nb)] for j in range(nb)]

# Update the matrix with the edges of sol
def update_matrix(mat, sol):
    for r in sol:
        for i in range(len(r)-1):
            e = [r[i], r[i+1]]
            e.sort()
            mat[e[0]][e[1]] += 1
        mat[0][r[len(r)-1]] += 1
    return mat

# Generate nb tuples (lam,mu,nu), and create a base of CW solution
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
        routes = route.normalize_solution(routes)
        me += route.cost_sol(routes, instance,const.quality_cost)
        Base.append((route.cost_sol(routes, instance,const.quality_cost), routes, (l, m, n)))
    Base.sort()

    return Base, [Base[0][0], Base[len(Base)-1][0], me/nb]

# Build a learning set with quantity criterion
def learning_set_quantity(Base, percent):
    ens = Base[:len(Base)//percent]
    ls = []
    for s in ens:
        ls.append(s[1])
    return ls

# Build a learning set with quality criterion
def learning_set_quality(Base, lim):
    ls = []
    i = 0
    while i < len(Base) and Base[i][0] <= lim:
        ls.append(Base[i][1])
        i += 1
    return ls

# Update the matrix for all solutions in the learning set (ls)
def learn(mat, ls):
    for sol in ls:
        update_matrix(mat, sol)
    return mat

# Return the edges we conserve with the threshold criterion
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

# Return the edges we conserve with the rank criterion
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

# Combine all previous functions to do the learning phase
def learning_results(crit, iterations, generate, instance, demand,capacity, initial, typeBase, percent, learningCriterion):
    edges = []
    bigBase = []
    for lg in range(iterations):
        
        Base, stat = rd_generate(generate, instance, demand,capacity, initial)
        bigBase += Base
        if typeBase == "Quality":
            quality = (stat[1]-stat[0])*percent/100 + stat[0]
            learning_set = learning_set_quality(Base, quality)
        if typeBase == "Quantity":
            learning_set = learning_set_quantity(Base,percent)

        learningMatrix = init_matrix(len(instance))
        learningMatrix = learn(learningMatrix, learning_set)

        if learningCriterion == "Rank":
            learningEdges = mat_info_rg(int(len(demand)*crit), learningMatrix)
        if learningCriterion == "Threshold":
            learningEdges = mat_info_req(int(len(learning_set)*crit), learningMatrix)

        for e in learningEdges:
            if not utile.is_edge_in(e, edges) and not utile.unfeasable_edge(e, edges):
                edges.append(e)

    bigBase.sort()
    param = []
    for b in bigBase:
        param.append(b[2])
    return edges, param

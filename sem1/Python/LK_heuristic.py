import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m

ylim = 200
xlim = 200


def create_instance(n):
    inst = [(0, 0)]
    route = [0]
    for i in range(n):
        x = rd.randint(-xlim, xlim)
        y = rd.randint(-ylim, ylim)
        inst.append((x, y))
        route.append(i+1)
    route.append(0)
    return inst, route


def print_instance(inst):
    dep = inst[0]
    cust = inst[1:]
    py.plot(dep[0], dep[1], color='blue', marker='o')
    for i in cust:
        py.plot(i[0], i[1], color='red', marker='o')


def print_route(route, inst):
    x = []
    y = []
    for i in range(len(route)):
        x.append(inst[route[i]][0])
        y.append(inst[route[i]][1])
    py.plot(x, y, color='black')

def print_costs(costs):
    for i in range(len(costs)):
        py.plot(i,costs[i], color="blue", marker='o')

def decoupe_route(route):
    mini_routes=[]
    for i in range(len(route)//10):
        mini_routes.append(route[i*10:(i+1)*10])
    mini_routes.append(route[10*(len(route)//10):])
    return mini_routes

def distance(p1, p2):
    return m.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def cost_sol(route, inst):
    c = 0
    for r in range(len(route)-1):
        p1 = inst[route[r]][0], inst[route[r]][1]
        p2 = inst[route[r+1]][0], inst[route[r+1]][1]
        c += distance(p1, p2)
    return c


def DeuxOpt(route, inst):
    l = len(route)-1
    best_tuple = (0,0)
    best = 0
    for i in range(l):
        pi = inst[route[i]]
        spi = inst[route[i+1]]
        for j in range(l):
            if j!=i-1 and j!=i and j!=i+1:
                pj = inst[route[j]]
                spj = inst[route[j+1]]
                d = (distance(pi, spi)+distance(pj, spj)) - distance(pi, pj)-distance(spi, spj)
                if d > best:
                    best_tuple = (i+1,j)
                    best = d
    cand = route.copy()
    print_instance(inst)
    py.plot(inst[route[best_tuple[0]]][0],inst[route[best_tuple[0]]][1], color = "green", marker="o")
    py.plot(inst[route[best_tuple[1]]][0],inst[route[best_tuple[1]]][1], color = "yellow",marker="o")
    py.plot(inst[route[best_tuple[0]-1]][0],inst[route[best_tuple[0]-1]][1], color = "green",marker="o")
    py.plot(inst[route[best_tuple[1]+1]][0],inst[route[best_tuple[1]+1]][1], color = "yellow",marker="o")
    cand[best_tuple[0]],cand[best_tuple[1]] = cand[best_tuple[1]],cand[best_tuple[0]]
    return cand


def LK(route, inst):
    next_cand = DeuxOpt(route,inst)
    while next_cand!=route:
        route = next_cand.copy()
        print(cost_sol(next_cand,inst))
        print_route(route,inst)
        py.show()
        next_cand = DeuxOpt(route, inst)
    return next_cand

    '''
    print(possible_cand)
    for i in range(k-1):
        next_possible_cand=[]
        for p in possible_cand:
            next_possible_cand + = DeuxOpt(p,inst)
        possible_cand = next_possible_cand.copy()
    return (possible(cand))
    '''
"""
def divided_lk(lim,routes,inst):
    mini_routes = decoupe_route(route)
    cand = []
    for i in mini_routes:
        cand = cand + LK(lim,i,inst)
    cand = LK(lim,cand,inst)
    return cand
"""

inst, route = create_instance(9)
print_instance(inst)
print_route(route, inst)

py.show()

opt_route = LK(route, inst)




print_instance(inst)
print_route(opt_route, inst)

py.show()


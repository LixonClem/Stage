import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m

global ylim 
global xlim 
global clim 

global instance_test


ylim = 100
xlim = 100
clim = 50


def create_instance(n):
    inst=[(0,0)]
    demand=[]
    for i in range(n):
        x = rd.randint(-xlim,xlim)
        y = rd.randint(-ylim,ylim)
        c = rd.randint(0,clim)
        inst.append((x,y))
        demand.append(c)
    return inst

def print_instance(inst):
    dep = inst[0]
    cust = inst[1:]
    py.plot(dep[0],dep[1], color='blue', marker='o')
    for i in cust:
        py.plot(i[0],i[1], color='red', marker='o')

def print_route(route, inst):
    x=[]
    y=[]
    for i in range(len(route)):
        x.append(inst[route[i]][0])
        y.append(inst[route[i]][1])
    py.plot(x, y, color='black')


def print_routes(routes, inst):
    for i in routes:
        print_route(i, inst)
    py.show()

def print_current_sol(routes, inst):
    print_instance(inst)
    print_routes(routes, inst)
    py.show()

def init_routes(inst):
    routes=[]
    for j in range(1,len(inst)):
        routej = [0,j,0]
        routes.append(routej)
    return routes


def distance(p1,p2):
    return m.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def compute_savings(inst, lam):
    savings = [[0 for j in range(len(inst)-1)] for i in range(len(inst)-1)]
    for i in range (len(inst)-1):
        for j in range(len(inst)-1):
            if (i==j):
                savings[i][j]=0
            else:
                savings[i][j]=distance(inst[i+1], inst[0]) + distance(inst[j+1], inst[0]) - lam*distance(inst[i+1], inst[j+1])
    return savings

def max_savings(n,savings):
    cand = (-1,0,0)
    for i in range(n):
        for j in range(i+1, n):
            if cand[0]<0 or savings[i][j]>cand[2]:
                cand = (i+1,j+1,savings[i][j])
    return cand

def find_route(i,routes): # Trouve la route à laquelle appartient l'usager i
    for k in range(len(routes)):
        if i in routes[k]:
            return routes[k]

def can_merge(i,r1,j,r2):
    if r1==r2:
        return -1
    elif (r1[1]==i and r2[len(r2)-2]==j):
        return 1
    elif (r1[len(r1)-2]==i and r2[1]==j):
        return 2
    else:
        return -1

def merge_routes(cand, routes, savings, inst):
    i,j = cand[0],cand[1]
    r1,r2 = find_route(i,routes),find_route(j,routes)
    mrge = can_merge(i,r1,j,r2)
    if mrge>0:
        routes.remove(r1)
        routes.remove(r2)
        if mrge==1:
            r1.pop()
            r2.remove(0)
            new_road = r1 + r2
        else:
            r2.pop()
            r1.remove(0)
            new_road = r2 + r1
        routes.append(new_road)
    savings[i-1][j-1]=0

def cost_sol(routes, inst):
    c=0
    for r in routes:
        for i in range(len(r)-1):
            (x1,y1) = inst[r[i]][0],inst[r[i]][1]
            (x2,y2) = inst[r[i+1]][0],inst[r[i+1]][1]
            c+=distance((x1,y1),(x2,y2))
    return c


def ClarkeWright(inst, lam):
    routes = init_routes(inst)
    savings = compute_savings(inst, lam)
    (i,j,s) = max_savings(len(inst)-1,savings)
    while s>0:
        merge_routes((i,j,s), routes, savings, inst)
        (i,j,s) = max_savings(len(inst)-1,savings)
    return cost_sol(routes,inst)

def tests_lambda(inst):
    cmin = -1
    lbest = 0
    costs = []
    lam_value = [0.1*k for k in range(1,21)]
    for lam in lam_value:
        c = ClarkeWright(inst, lam)
        costs.append(c)
        if c < cmin or cmin<0:
            cmin = c
            lbest = lam
    py.plot(lam_value,costs)
    return lbest

def mean(t):
    s = 0
    for i in t:
        s+=i
    return s/len(t)

def tests(n,customers):
    bests_lam = []
    for test in range(n):
        inst = create_instance(customers)
        lam = tests_lambda(inst)
        bests_lam.append(lam)
    py.show()
    return(mean(bests_lam))



def customers_influence(n,lim):
    customers = []
    lamMean = []
    for nb_cust in range(10,lim):
        print(nb_cust)
        lamMean.append(tests(n,nb_cust))
        customers.append(nb_cust)
    py.plot(customers, lamMean)
    py.show()
#Tests
"""
inst = create_instance(nb_cust)
print_instance(inst)

routes = init_routes(inst)
print(routes)

savings = compute_savings(inst)
print(savings)

(i,j,s) = max_savings(savings)
print(i,j,s)

k = find_route(i,routes)
print(k)

print(can_merge(i,find_route(i,routes),j, find_route(j,routes)))

print_routes(routes,inst)

merge_routes((i,j,s), routes, savings, inst)


instance_test = create_instance(nb_cust)


routes = ClarkeWright(instance_test, lam)
print(routes)


tests_lambda(instance_test)

"""

customers_influence(10,50)
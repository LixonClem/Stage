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

ylim = 200
xlim = 200
clim = 50
nb_cust = 100
Capacity = 100

lam = 1.4

def create_instance(n):
    inst=[(0,0)]
    demand=[]
    for i in range(n):
        x = rd.randint(-xlim,xlim)
        y = rd.randint(-ylim,ylim)
        c = rd.randint(0,clim)
        inst.append((x,y))
        demand.append(c)
    return inst,demand

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
        print_route(i[0], inst)
    py.show()

def print_current_sol(routes, inst):
    print_instance(inst)
    print_routes(routes, inst)
    py.show()

def init_routes(inst,demand):
    routes=[]
    for j in range(1,len(inst)):
        routej = ([0,j,0],demand[j-1])
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
        if i in routes[k][0]:
            return routes[k]

def can_merge(i,r1,j,r2):
    if r1==r2:
        return -1
    elif (r1[0][1]==i and r2[0][len(r2[0])-2]==j and r2[1]+r1[1]<=Capacity):
        return 1
    elif (r1[0][len(r1[0])-2]==i and r2[0][1]==j and r1[1]+r2[1]<=Capacity):
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
            r1[0].pop()
            r2[0].remove(0)
            new_road = (r1[0] + r2[0],r1[1] + r2[1])
        else:
            r2[0].pop()
            r1[0].remove(0)
            new_road = (r2[0] + r1[0],r1[1] + r2[1])
        routes.append(new_road)
    savings[i-1][j-1]=0

def cost_sol(routes, inst):
    c=0
    for r in routes:
        for i in range(len(r[0])-1):
            (x1,y1) = inst[r[0][i]][0],inst[r[0][i]][1]
            (x2,y2) = inst[r[0][i+1]][0],inst[r[0][i+1]][1]
            c+=distance((x1,y1),(x2,y2))
    return c


def ClarkeWright(inst, demand, lam):
    routes = init_routes(inst, demand)
    savings = compute_savings(inst, lam)
    (i,j,s) = max_savings(len(inst)-1,savings)
    while s>0:
        merge_routes((i,j,s), routes, savings, inst)
        (i,j,s) = max_savings(len(inst)-1,savings)
    print_current_sol(routes,inst)
    return cost_sol(routes,inst)


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
"""

instance_test,demand = create_instance(100)
print(instance_test,demand)
"""
cost = ClarkeWright(instance_test, demand, lam)
print(cost)"""




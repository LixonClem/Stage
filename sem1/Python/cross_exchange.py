import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m

global ylim 
global xlim 
global nb_cust
global kNN

ylim = 200
xlim = 200
nb_cust = 10
kNN = 5

inst_test=[(0, 0), (3, -168), (150, -157), (-195, 68), (105, 4), (-114, -23), (72, -152), (135, -129), (65, -87), (-27, 158), (53, -46)] 
r1_test=[0, 2, 4, 6, 8, 10, 0]
r2_test=[0, 1, 3, 5, 7, 9, 0]
edge_1 = (4,6)
edge_2 = (3,5)
edge_3 = (7,9)

def create_instance(n):
    inst=[(0,0)]
    route1 = [0]
    route2 = [0]
    for i in range(n):
        x = rd.randint(-xlim,xlim)
        y = rd.randint(-ylim,ylim)
        inst.append((x,y))
        if i%2 == 0:
            route1.append(i)
        else:
            route2.append(i)
    return inst,route1,route2

def print_instance(inst):
    dep = inst[0]
    cust = inst[1:]
    py.plot(dep[0],dep[1], color='blue', marker='o')
    for i in cust:
        py.plot(i[0],i[1], color='red', marker='o')

def print_route(route, inst,c):
    x=[]
    y=[]
    for i in range(len(route)):
        x.append(inst[route[i]][0])
        y.append(inst[route[i]][1])
    py.plot(x, y, label = "route " + str(c))


def print_routes(routes, inst):
    c=1
    for i in routes:
        print_route(i, inst,c)
        c+=1

def print_current_sol(routes, inst):
    print_instance(inst)
    print_routes(routes, inst)

def distance(p1,p2):
    return m.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def cost_sol(routes, inst):
    c=0
    for r in routes:
        for i in range(len(r)-1):
            a = inst[r[i]]
            b = inst[r[i+1]]
            c+=distance(a,b)
    return c

#Compute the kNN for each node
def voisins(k,inst):
    v = []
    for i in range(len(inst)):
        vi = []
        couples = []
        for j in range(len(inst)):
            if i!=j :
                vi.append([distance(inst[i],inst[j]),j])
        vi.sort()
        for l in vi:
            couples.append(l[1])
        v.append(couples[:k])
    return v

def find_route(i,routes): # Trouve la route à laquelle appartient l'usager i
    for k in range(len(routes)):
        if i in routes[k]:
            return routes[k]

def another_route(edge, voisins, routes):
    (a,b) = edge
    r1 = find_route(a,routes)
    for i in voisins[a]:
        r2 = find_route(i,routes)
        if r2!=r1 and i!=0:
            return ((r1,r2),i)
    return ("pas assez de voisins")

def cross_exchange(edge, voisins, routes, inst):
    (r1,r2),v = another_route(edge, voisins, routes)
    c_init = cost_sol(routes,inst)
    print(c_init)
    (a,b) = edge
    i_v = r2.index(v)
    i_b = r1.index(b)
    
    r1[i_b],r2[i_v]=v,b
    print(routes)

    cand_r = [r1,r2].copy()
    current_cand = cand_r.copy()
    for i in range (len(r2)-1):
        if i!=i_v-1:
            e1 = (i,i+1)
            for j in range (len(r1)-1):
                if j!=i_b-1:
                    e2 = (j, j+1)
                    print(e1,e2)
                    current_cand[0][j+1],current_cand[1][i+1] = current_cand[1][i+1],current_cand[0][j+1]
                    if cost_sol(current_cand,inst)<c_init:
                        print(cost_sol(current_cand,inst))
                        return current_cand
                current_cand = cand_r.copy()
    return routes

routes = [r1_test,r2_test]



print_current_sol(routes,inst_test)
py.plot([inst_test[edge_1[0]][0],inst_test[edge_1[1]][0]],[inst_test[edge_1[0]][1],inst_test[edge_1[1]][1]], color='red', label='chosen')
py.title("Test de l'opérateur Cross-exchange")
py.legend()
py.show()
v = voisins(kNN,inst_test)
(r1,r2),i = another_route(edge_1,v,routes)


new_routes = cross_exchange(edge_1,v,routes,inst_test)

print_current_sol(new_routes,inst_test)
py.title("Test de l'opérateur Cross-exchange")
py.legend()
py.show()
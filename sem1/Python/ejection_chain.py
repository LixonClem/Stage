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
kNN = 10

# Creation of a test instance
inst_test = [(0, 0), (3, -168), (150, -157), (-195, 68), (105, 4),
             (-114, -23), (72, -152), (135, -129), (65, -87), (-27, 158), (53, -46)]
r1_test = [0, 2, 4, 6, 8, 10]
r2_test = [0, 1, 3, 5, 7, 9]
edge_1 = (4, 6)
edge_2 = (5, 3)
edge_3 = (7, 9)

inst_test2 = [(0, 0), (11, 120), (-142, -149), (-83, 39), (-168, -46), (-83, -146), (4, -99),
              (32, -16), (-117, 12), (-132, 33), (51, 44), (-29, 76), (-98, -33), (-26, -190),
              (-89, 128), (124, -95), (-108, -1), (24, -158), (-115, -106), (80, -160), (-167, 3), (185, -72)]

r1_test2 = [0, 3, 6, 9, 12, 15, 18, 21] 
r2_test2 = [0, 1, 4, 7, 10, 13, 16, 19] 
r3_test3 = [0, 2, 5, 8, 11, 14, 17, 20]

edge2_1 = (9,12)

# Creation of a random instance
def create_instance(n):
    inst = [(0, 0)]
    route1 = [0]
    route2 = [0]
    route3 = [0]
    for i in range(n):
        x = rd.randint(-xlim, xlim)
        y = rd.randint(-ylim, ylim)
        inst.append((x, y))
        if i % 3 == 0:
            route1.append(i)
        elif i % 3 == 1:
            route2.append(i)
        else:
            route3.append(i)
    return inst, route1, route2, route3

# Print the routes


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
    py.plot(x, y, label="route " + str(c))


def print_routes(routes, inst):
    c = 1
    for i in routes:
        print_route(i, inst, c)
        c += 1


def print_current_sol(routes, inst):
    print_instance(inst)
    print_routes(routes, inst)


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

# Find the route, which contains customer i


def find_route(i, routes):
    for k in range(len(routes)):
        if i in routes[k]:
            return routes[k]

# Return the nearest route of the edge given. Can return 0


def another_route(a, voisins, routes):
    r1 = find_route(a, routes)
    for i in voisins[a]:
        r2 = find_route(i, routes)
        if r2 != r1:
            return ((r1, r2), i)
    return ()

#Compute the saving of the new edge
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

# to do: don't violate the capacity constraint


def eval_cand(edge, voisins, routes, inst):
    (a, b) = edge
    if b != 0:
        (r1, r2), v = another_route(b, voisins, routes)
        i_v, i = r2.index(v), r1.index(b)
    else:
        (r1, r2), v = another_route(a, voisins, routes)
        i_v, i = r2.index(v), r1.index(a)
    return (saving(i, r1, i_v, r2, inst), (i, i_v), (r1, r2))


def best_cand(route, np, voisins, routes, inst):
    S = []
    for p in route:
        i = route.index(p)
        if p != np:
            S.append(eval_cand((route[i-1], p), voisins, routes, inst))
    S.sort()
    return S[-1]


def ejection_chain(l, edge, voisins, routes, inst):
    print(cost_sol(routes, inst))
    s, I, R = eval_cand(edge, voisins, routes, inst)
    R[1].insert(I[1]+1, R[0][I[0]])
    R[0].remove(R[0][I[0]])
    print(cost_sol(routes, inst))
    for k in range(l-1):
        
        curr_route = R[1]
        s, I, R = best_cand(curr_route, R[0][I[0]], voisins, routes, inst)
        R[1].insert(I[1]+1, R[0][I[0]])
        R[0].remove(R[0][I[0]])
        print(cost_sol(routes, inst))

    return routes




routes = [r1_test2, r2_test2,r3_test3]

print_current_sol(routes, inst_test2)
py.plot([inst_test2[edge2_1[0]][0], inst_test2[edge2_1[1]][0]], [
        inst_test2[edge2_1[0]][1], inst_test2[edge2_1[1]][1]], color='black', label='chosen')
py.title("Test de l'opérateur ejection_chain")
py.legend()
py.show()
v = voisins(kNN, inst_test2)


new_routes = ejection_chain(14,edge2_1,v,routes,inst_test2)
print(new_routes)

print_current_sol(new_routes,inst_test2)
py.show()

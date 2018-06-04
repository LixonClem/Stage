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

# Creation of a test instance
inst_test = [(0, 0), (3, -168), (150, -157), (-195, 68), (105, 4),
             (-114, -23), (72, -152), (135, -129), (65, -87), (-27, 158), (53, -46)]
r1_test = [0, 2, 4, 6, 8, 10, 0]
r2_test = [0, 1, 3, 5, 7, 9, 0]
edge_1 = (4, 6)
edge_2 = (5, 3)
edge_3 = (7, 9)

# Creation of a random instance


def create_instance(n):
    inst = [(0, 0)]
    route1 = [0]
    route2 = [0]
    for i in range(n):
        x = rd.randint(-xlim, xlim)
        y = rd.randint(-ylim, ylim)
        inst.append((x, y))
        if i % 2 == 0:
            route1.append(i)
        else:
            route2.append(i)
    return inst, route1, route2

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

# Return the nearest route of the edge given


def another_route(a, voisins, routes):
    r1 = find_route(a, routes)
    for i in voisins[a]:
        r2 = find_route(i, routes)
        print(i)
        if r2 != r1 and i != 0:
            return ((r1, r2), i)
    return ("pas assez de voisins")

# Apply the cross-exchange operator


def cross_exchange(edge, voisins, routes, inst):
    (a, b) = edge
    (r1, r2), v = another_route(a, voisins, routes)
    c_init = cost_sol(routes, inst)
    i_v = r2.index(v)
    i_b = r1.index(b)

    r1[i_b], r2[i_v] = v, b

    current_cand = [r1.copy(), r2.copy()]

    for i in range(len(r2)-1):
        if i != i_v-1:
            for j in range(len(r1)-1):
                if j != i_b-1:
                    print(current_cand)
                    current_cand[0][j+1], current_cand[1][i +
                                                          1] = current_cand[1][i+1], current_cand[0][j+1]
                    print(current_cand)
                    if cost_sol(current_cand, inst) < c_init:
                        
                        return current_cand
                current_cand = [r1.copy(), r2.copy()]
    return routes


# tests pour l'opérateur
routes = [r1_test, r2_test]


print_current_sol(routes, inst_test)
py.plot([inst_test[edge_1[0]][0], inst_test[edge_1[1]][0]], [
        inst_test[edge_1[0]][1], inst_test[edge_1[1]][1]], color='red', label='chosen')
py.title("Test de l'opérateur Cross-exchange")
py.legend()
py.show()
v = voisins(kNN, inst_test)



new_routes = cross_exchange(edge_1, v, routes, inst_test)
print(new_routes)

print_current_sol(new_routes, inst_test)
py.title("Test de l'opérateur Cross-exchange")
py.legend()
py.show()

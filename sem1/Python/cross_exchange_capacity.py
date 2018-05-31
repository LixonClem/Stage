import numpy as np
import scipy as sp
import matplotlib.pyplot as py
import random as rd
import math as m

global ylim
global xlim
global nb_cust
global kNN
global clim
global Capacity

ylim = 200
xlim = 200
clim = 20
nb_cust = 10
kNN = 5
Capacity = 75

#Note : instances and voisins will be define globally in a future version

# Creation of a test instance
inst_test = [(0, 0), (3, -168), (150, -157), (-195, 68), (105, 4),
             (-114, -23), (72, -152), (135, -129), (65, -87), (-27, 158), (53, -46)]
demand = [0, 5, 10, 15, 5, 10, 5, 15, 20, 1, 4]
r1_test = [[0, 2, 4, 6, 8, 10], 44]
r2_test = [[0, 1, 3, 5, 7, 9], 46]
edge_1 = (4, 6)
edge_2 = (3, 5)
edge_3 = (7, 9)

# Creation of a random instance


def create_instance(n):
    inst = [(0, 0)]
    route1 = [0]
    route2 = [0]
    demand = []
    for i in range(n):
        x = rd.randint(-xlim, xlim)
        y = rd.randint(-ylim, ylim)
        c = rd.randint(0, clim)
        inst.append((x, y))
        demand.append(c)
        if i % 2 == 0:
            route1.append(i)
        else:
            route2.append(i)
    return inst, route1, route2, demand

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
        print_route(i[0], inst, c)
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
        for i in range(len(r[0])-1):
            a = inst[r[0][i]]
            b = inst[r[0][i+1]]
            c += distance(a, b)
        c += distance(inst[r[0][len(r[0])-1]], inst[r[0][0]])
    return c

# Compute the kNN for each node


def voisins(k, inst):
    neighbours = []
    n = len(inst)
    for i in range(n):
        neighbours_i = []
        couples = []
        for j in range(n):
            if i != j:
                couples.append([distance(inst[i], inst[j]), j])
        couples.sort()
        for l in couples:
            neighbours_i.append(l[1])
        neighbours.append(neighbours_i[:k])
    return neighbours

# Find the route, which contains customer i


def find_route(i, routes):
    for k in range(len(routes)):
        if i in routes[k][0]:
            return routes[k]

# Return the nearest route of the edge given


def another_route(edge, voisins, routes, demand):
    (a, b) = edge
    r1 = find_route(a, routes)
    for i in voisins[a]:
        r2 = find_route(i, routes)
        if r2 != r1 and i != 0 and r1[1]-demand[b]+demand[i] <= Capacity and r2[1]-demand[i]+demand[b] <= Capacity:     #we verify that the future deman on the route won't exceed his capacity
            return ((r1, r2), i)
    return ((r1, r1), -1)       #error case, we haven't found a second route, so no modifications

# Apply the cross-exchange operator


def cross_exchange(edge, voisins, routes, inst, demand):
    (a, b) = edge
    
    (r1, r2), v = another_route(edge, voisins, routes, demand)  # compute the two routes considered, and the NN of the point we remove (a). v is a point 
    if v < 0:
        return (routes)

    c_init = cost_sol(routes, inst)     # for a future comparison
    print(c_init)
    i_v = r2[0].index(v)    
    i_a = r1[0].index(a)
    print(i_a,i_v)
    print(r1[0][i_a], r2[0][i_v])
    r1[0][i_a], r2[0][i_v-1] = r2[0][i_v], a

    r1[1] = r1[1] - demand[a] + demand[v]   #update the demands on each road
    r2[1] = r2[1] - demand[v] + demand[a]

    current_cand = [[r1[0].copy(), r1[1]].copy(), [r2[0].copy(), r2[1]].copy()]     #copy of the current solution 

    for i in range(len(r2[0])-1):
        if i != i_v-1:
            for j in range(len(r1[0])-1):
                if j != i_a-1:
                    p1 = current_cand[0][0][j+1]
                    p2 = current_cand[1][0][i+1]
                    current_cand[0][1] = current_cand[0][1] - \
                        demand[p1] + demand[p2]
                    current_cand[1][1] = current_cand[1][1] - \
                        demand[p2] + demand[p1]

                    current_cand[0][0][j+1], current_cand[1][0][i + 1] = p2, p1

                    
                    if cost_sol(current_cand, inst) < c_init and current_cand[0][1] <= Capacity and current_cand[1][1] <= Capacity:
                        print(cost_sol(current_cand, inst))
                        return current_cand

                current_cand = [[r1[0].copy(), r1[1]].copy(), [r2[0].copy(), r2[1]].copy()]
    return routes


# tests pour l'opérateur
routes = [r1_test, r2_test]


print_current_sol(routes, inst_test)
py.plot([inst_test[edge_3[0]][0], inst_test[edge_3[1]][0]], [
        inst_test[edge_3[0]][1], inst_test[edge_3[1]][1]], color='red', label='chosen')
py.title("Test de l'opérateur Cross-exchange")
py.legend()
py.show()
v = voisins(kNN, inst_test)


new_routes = cross_exchange(edge_3, v, routes, inst_test, demand)
print(new_routes)
print_current_sol(new_routes, inst_test)
py.title("Test de l'opérateur Cross-exchange")
py.legend()
py.show()

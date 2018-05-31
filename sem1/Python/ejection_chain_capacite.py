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
global Error

ylim = 200
xlim = 200
clim = 30
nb_cust = 10
kNN = 5
Capacity = 175

Error = (0, (0, 0), ([[0], 0], [[0], 0]))

# Creation of a test instance


inst_test1 = [(0, 0), (38, 127), (78, -181), (139, -128), (-50, -148), (-60, 83), (-30, -14), (7, -65),
              (-11, 32), (-113, -161), (-166, -11), (-110, -
                                                     134), (-75, 95), (9, 45), (-137, 58),
              (69, 175), (-121, 154), (70, 20), (-68, -73), (-165, 180), (59, 82), (-89, 60)]

r1_test1 = [[0, 3, 6, 9, 12, 15, 18, 21], 128]
r2_test1 = [[0, 1, 4, 7, 10, 13, 16, 19], 147]
r3_test1 = [[0, 2, 5, 8, 11, 14, 17, 20], 129]

demand1 = [0, 25, 19, 26, 19, 22, 10, 7, 8, 14,
           19, 10, 11, 24, 24, 27, 30, 29, 18, 23, 17, 22]

edge1 = (8, 11)

# Creation of a random instance


def create_instance(n):
    inst = [(0, 0)]
    demand = []
    route1 = [0]
    route2 = [0]
    route3 = [0]
    for i in range(n):
        x = rd.randint(-xlim, xlim)
        y = rd.randint(-ylim, ylim)
        c = rd.randint(0, clim)
        inst.append((x, y))
        demand.append(c)
        if i % 3 == 0:
            route1.append(i)
        elif i % 3 == 1:
            route2.append(i)
        else:
            route3.append(i)
    return inst, route1, route2, route3, demand

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
        if i in routes[k][0]:
            return routes[k]

# Return the nearest route of the edge given. Can return -1, if nothing found.


def another_route(a, voisins, routes, demand):
    r1 = find_route(a, routes)
    for i in voisins[a]:
        r2 = find_route(i, routes)
        if r2 != r1 and r2[1]+demand[a] <= Capacity:
            return ((r1, r2), i)
    return (r1, r1), -1

# Compute the saving of the new edge


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

# evalue a possible next edge.

def eval_cand(edge, voisins, routes, inst, demand):
    (a, b) = edge
    if b != 0:
        (r1, r2), v = another_route(b, voisins, routes, demand)
        if v < 0:
            return Error
        i_v, i = r2[0].index(v), r1[0].index(b)

    else:
        (r1, r2), v = another_route(a, voisins, routes, demand)
        if v < 0:
            return Error
        i_v, i = r2[0].index(v), r1[0].index(a)
    return (saving(i, r1[0], i_v, r2[0], inst), (i, i_v), (r1, r2))

# return the best relocation for each point p in the route.
# Return the point to relocate and his neighbour considered.


def best_cand(route, np, voisins, routes, inst, demand):
    S = []
    for p in route:
        i = route.index(p)
        if p != np:
            S.append(eval_cand((route[i-1], p), voisins, routes, inst, demand))
    S.sort()
    return S[-1]


def ejection_chain(l, edge, voisins, routes, inst, demand):
    S = 0  # global cost modification of the current solution
    initial_routes = np.copy(routes)
    s, I, R = eval_cand(edge, voisins, routes, inst, demand)

    if (s, I, R) == Error:
        return routes

    S += s
    relocated_cust = R[0][0][I[0]]

    # update the routes
    R[1][0].insert(I[1]+1, relocated_cust)
    R[1][1] += demand[relocated_cust]
    R[0][1] -= demand[relocated_cust]
    R[0][0].remove(relocated_cust)

    for k in range(l-1):
        curr_route = R[1][0]
        s, I, R = best_cand(curr_route, relocated_cust,
                            voisins, routes, inst, demand)

        if (s, I, R) == Error:
            return routes
        S += s

        relocated_cust = R[0][0][I[0]]
        R[1][0].insert(I[1]+1, relocated_cust)
        R[1][1] += demand[relocated_cust]
        R[0][1] -= demand[relocated_cust]
        R[0][0].remove(relocated_cust)

    if S < 0:  # If the final result is worse than the initial then we don't apply changes
        return initial_routes

    return routes


# Test execution

routes = [r1_test1, r2_test1, r3_test1]
print(cost_sol(routes,inst_test1))
print_current_sol(routes, inst_test1)

py.plot([inst_test1[edge1[0]][0], inst_test1[edge1[1]][0]], [
        inst_test1[edge1[0]][1], inst_test1[edge1[1]][1]], color='black', label='chosen')
py.title("Test de l'opérateur ejection_chain")
py.legend()
py.show()

v = voisins(kNN, inst_test1)


new_routes = ejection_chain(15, edge1, v, routes, inst_test1, demand1)
print(cost_sol(new_routes,inst_test1))
print_current_sol(new_routes, inst_test1)
py.show()

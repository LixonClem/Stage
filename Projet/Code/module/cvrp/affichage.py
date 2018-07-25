# This module contains functions, which allow a display of instances/solutions/edges.
# Don't forget the py.show() after using one of these functions

import matplotlib.pyplot as py

# Display the customers and the depot of the instance


def print_instance(inst):
    dep = inst[0]
    cust = inst[1:]
    py.plot(dep[0], dep[1], color='blue', marker='o')
    for i in cust:
        py.plot(i[0], i[1], color='red', marker='o')

# Display the route given


def print_route(route, inst, c):
    x = []
    y = []
    for i in range(len(route)):
        x.append(inst[route[i]][0])
        y.append(inst[route[i]][1])
    x.append(inst[route[0]][0])
    y.append(inst[route[0]][1])
    py.plot(x, y, label="route" + str(c))

# Display the current solution


def print_solution(solution, inst):
    print_instance(inst)
    c = 1
    for i in solution:
        print_route(i, inst, c)
        c += 1

# Display the edges given


def print_edges(edges, inst, col):
    for e in edges:
        x = [inst[e[0]][0], inst[e[1]][0]]
        y = [inst[e[0]][1], inst[e[1]][1]]
        py.plot(x, y, color=col)

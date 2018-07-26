# This module contains functions, which allow a display of instances/solutions/edges.
# Don't forget the py.show() after using one of these functions

import matplotlib.pyplot as py
import cvrp.const as const
# Display the customers and the depot of the instance


def print_instance():
    dep = const.instance[0]
    cust = const.instance[1:]
    py.plot(dep[0], dep[1], color='blue', marker='o')
    for i in cust:
        py.plot(i[0], i[1], color='red', marker='o')

# Display the route given


def print_route(route, color):
    x = []
    y = []
    for i in range(len(route)):
        x.append(const.instance[route[i]][0])
        y.append(const.instance[route[i]][1])
    x.append(const.instance[route[0]][0])
    y.append(const.instance[route[0]][1])
    py.plot(x, y, label="route" + str(color))

# Display the current solution


def print_solution(solution):
    print_instance()
    c = 1
    for i in solution:
        print_route(i, c)
        c += 1

# Display the edges given


def print_edges(edges, col):
    for e in edges:
        x = [const.instance[e[0]][0], const.instance[e[1]][0]]
        y = [const.instance[e[0]][1], const.instance[e[1]][1]]
        py.plot(x, y, color=col)

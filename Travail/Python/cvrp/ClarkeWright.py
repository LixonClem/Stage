# Code for the CW heuristic with (lambda, mu, nu) paramaters.

import cvrp.utile as utile
import cvrp.route as route

def init_routes(inst, demand):
    routes = []
    for j in range(1, len(inst)):
        routej = [0, j, 0]
        routes.append(routej)
    return routes

def mean_demand(demand):
    n = len(demand)
    d = 0
    for i in demand:
        d += i
    return d/(n-1)

def compute_savings(inst, demand, lam, mu, nu):
    savings = []
    d_bar = mean_demand(demand)
    for i in range(len(inst)-1):
        s = 0
        for j in range(i+1, len(inst)-1):
            if (i == j):
                savings.append([0, (i+1, j+1)])
            else:
                s = utile.distance(inst[i+1], inst[0]) + utile.distance(inst[j+1], inst[0]) - lam*utile.distance(inst[i+1], inst[j+1]) + mu*abs(
                    utile.distance(inst[i+1], inst[0]) - utile.distance(inst[j+1], inst[0])) + (nu*(demand[i+1] + demand[j+1])/d_bar)
                if s >= 0:
                    savings.append([s, (i+1, j+1)])
    savings.sort()
    return savings


def max_savings(savings, cpt):
    maximum = savings[-cpt]
    return maximum


def can_merge(i, r1, j, r2, demand, capacity):
    if r1 == r2:
        return -1
    elif (r1[1] == i and r2[len(r2)-2] == j and route.route_demand(r1, demand)+route.route_demand(r2, demand) <= capacity):
        return 1
    elif (r1[len(r1)-2] == i and r2[1] == j and route.route_demand(r1, demand)+route.route_demand(r2, demand) <= capacity):
        return 2
    else:
        return -1


def merge_routes(i, j, routes, inst, demand, capacity,detailed_cust):
    ir1, ir2 = detailed_cust[i-1], detailed_cust[j-1]
    r1, r2 = routes[ir1].copy(), routes[ir2].copy()
    mrge = can_merge(i, r1, j, r2, demand,capacity)
    new_road = []
    if mrge > 0:
        if mrge == 1:
            r1.pop()
            r2.remove(0)
            new_road = r1 + r2
        else:
            r2.pop()
            r1.remove(0)
            new_road = r2 + r1
        routes.append(new_road)
        routes[ir1] = []
        routes[ir2] = []
        detailed_cust[i-1] = len(routes)-1
        detailed_cust[j-1] = len(routes)-1
        for k in new_road:
            detailed_cust[k-1] = len(routes)-1


def ClarkeWright(routes, inst, demand,capacity, lam, mu, nu, detailed_cust):
    new_routes = []
    cpt = 1

    savings = compute_savings(inst, demand, lam, mu, nu)
    [s, (i, j)] = max_savings(savings, cpt)
    while s > 0 and cpt < len(savings):

        savings[-cpt][0] = 0
        cpt += 1
        merge_routes(i, j, routes, inst, demand,capacity, detailed_cust)
        [s, (i, j)] = max_savings(savings, cpt)

    for i in range(len(routes)):
        if routes[i] != []:
            routes[i].pop()
            new_routes.append(routes[i])

    return new_routes
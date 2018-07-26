# Optimisation algorithm

import math as m
import time
import cvrp.const as const
import cvrp.utile as utile
import cvrp.learning as learn
import cvrp.route as route
import cvrp.linKernighan as LK
import cvrp.ejectionChain as EC
import cvrp.crossExchange as CE
import cvrp.ClarkeWright as CW

# Compute the gravity center of a route


def gravity_center(route):
    xg = 0
    yg = 0
    c = 0
    for i in route:
        pi = const.instance[i]
        xg += pi[0]
        yg += pi[1]
        c += 1
    return (xg/c, yg/c)

# Compute the width of an edge (i,j). G is the gravity center


def width(i, j, G):
    theta = m.acos(G[1]/utile.distance(G, (0, 0)))
    proj_i = (i[0]*m.sin(theta), i[1]*m.cos(theta))
    proj_j = (j[0]*m.sin(theta), j[1]*m.cos(theta))
    return abs(utile.distance(i, proj_i)-utile.distance(j, proj_j))

# Compute the cost of an edge (i,j). p is the penality of the edge


def cost(i, j, p):
    return utile.distance(i, j)*(1 + 0.1*p)

# Compute the depth of an edge (i,j).


def depth(i, j):
    return max(utile.distance(i, (0, 0)), utile.distance(j, (0, 0)))


# Return a penalization function with lw, lc and ld parameters


def penalization_function(lw, lc, ld):
    return lambda i, j, G, p: ((lw * width(i, j, G) + lc * cost(i, j, p))*(depth(i, j)/const.maxDepth)**(ld/2))/(1 + p)

# Compute the worst edge of the solution.
# b is the penalization function, p is a matrix of penalties


def bad_edge(b, p, solution,  fixed):
    cand = [0, (0, 0)]
    for r in solution:
        G = gravity_center(r)
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            b_ij = b(const.instance[pi], const.instance[pj], G, p[pi][pj])
            if b_ij > cand[0] and pi != 0 and pj != 0 and (pi, pj) not in fixed and (pj, pi) not in fixed:
                cand[0] = b_ij
                cand[1] = (pi, pj)
    return cand

# Try to find a new solution by doing a global optimisation
# (ie apply the neighborhood operators to each edges and conserve the best result)


def global_opti(solution):
    edges = route.all_edges(solution)
    fixed_edges = []
    c_init = route.cost_sol(solution, const.quality_cost)

    routes = route.copy_sol(solution)
    new_solution = route.copy_sol(routes)
    for e in edges:
        cp = utile.rd_point(e, solution)

        # apply determinist ejection-chain
        routes = EC.ejection_chain(
            cp, const.neighbors, routes, fixed_edges, "DE")

        for i in routes:
            if len(i) == 2:
                routes = EC.reject(i, routes, const.neighbors)

        # apply LK
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i])

        # apply determinist cross-exchange
        routes = CE.cross_exchange(
            cp, const.neighbors, routes, fixed_edges, "DE")

        # apply LK
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i])

        c_final = route.cost_sol(routes, const.quality_cost)
        if c_init - c_final > 0:
            c_init = c_final
            new_solution = route.copy_sol(routes)

    return new_solution

# Compute a solution of the given instance


def optimisation_heuristic(initial_routes, lam, mu, nu, fixed_edges):

    tps1 = time.time()
    B = [penalization_function(1, 0, 0),
         penalization_function(1, 1, 0),
         penalization_function(1, 0, 1),
         penalization_function(1, 1, 1),
         penalization_function(0, 1, 0),
         penalization_function(0, 1, 1)]

    b_i = 0
    b = B[b_i]
    p = [[0 for j in range(len(const.instance))]
         for i in range(len(const.instance))]
    all_solution = []

    detailed_cust = [0 for i in range(len(const.instance))]
    for r in range(len(initial_routes)):
        for i in initial_routes[r]:
            detailed_cust[i-1] = r
    initial_routes = CW.ClarkeWright(
        initial_routes, lam, mu, nu, detailed_cust)

    routes = route.copy_sol(initial_routes)
    routes2 = route.copy_sol(routes)

    c_init = route.cost_sol(routes, const.quality_cost)

    tps2 = time.time()
    tpsGS = time.time()
    tpsCH = time.time()

    while tps2-tps1 < const.limitTime:

        # find the worst edge
        worst = bad_edge(b, p, routes, fixed_edges)[1]

        p[worst[0]][worst[1]] += 1
        p[worst[1]][worst[0]] += 1

        # apply ejection-chain
        cp = utile.rd_point(worst, routes)
        routes = EC.ejection_chain(
            cp, const.neighbors, routes, fixed_edges, "RD")
        for i in routes:
            if len(i) == 2:
                routes = EC.reject(
                    i, routes, const.neighbors,)

        # apply LK
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i])

        # apply cross-exchange
        routes = CE.cross_exchange(
            cp, const.neighbors, routes, fixed_edges, "RD")

        # apply LK
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i])

        c_final = route.cost_sol(routes, const.quality_cost)

        if c_final < c_init:
            routes2 = route.copy_sol(routes)  # new optimum

            # Try to delete routes with only one customer
            for i in routes2:
                if len(i) == 2:
                    routes2 = EC.reject(i, routes2, const.neighbors)
            c_init = route.cost_sol(routes2, const.quality_cost)

            print(round(tps2-tps1, 2), round(c_init, 3))
            all_solution.append([c_init, route.copy_sol(routes2)])

            tps1 = time.time()
            tpsCH = time.time()
            tpsGS = time.time()

        # Compute a new solution near from current best
        if tps2-tpsGS > const.restartTime:

            edges = utile.fixed_alea(route.all_edges(routes2), const.conserved)
            routes = route.complete(utile.destruction(utile.ignore_0(edges)))

            detailed_cust = [0 for i in range(len(const.instance))]
            for r in range(len(routes)):
                for i in routes[r]:
                    detailed_cust[i-1] = r
            routes = CW.ClarkeWright(routes, lam, mu, nu, detailed_cust)

            tpsGS = time.time()

        # Change the penalization function and reset penalities
        if tps2-tpsCH > const.resetTime:
            b_i += 1

            if b_i < len(B):
                b = B[b_i]
                p = [[0 for j in range(len(const.instance))]
                     for i in range(len(const.instance))]
            else:
                b_i = 0
                b = B[b_i]
                p = [[0 for j in range(len(const.instance))]
                     for i in range(len(const.instance))]

            tpsCH = time.time()

        # Update the time of execution
        tps2 = time.time()

    for i in (routes2):
        if len(i) == 2:
            routes2 = EC.reject(
                i, routes2, const.neighbors)
        if len(i) == 1:
            routes2.remove(i)

    for i in range(len(routes2)):
        routes2[i] = LK.LK(routes2[i])

    all_solution.append(
        [route.cost_sol(routes2, const.quality_cost), route.copy_sol(routes2)])
    all_solution.sort()
    return all_solution

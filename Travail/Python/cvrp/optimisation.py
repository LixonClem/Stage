#Algorithme d'optimisation 

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

def gravity_center(route, inst):
    xg = 0
    yg = 0
    c = 0
    for i in route:
        pi = inst[i]
        xg += pi[0]
        yg += pi[1]
        c += 1
    return (xg/c, yg/c)


def width(i, j, G):
    theta = m.acos(G[1]/utile.distance(G, (0, 0)))
    proj_i = (i[0]*m.sin(theta), i[1]*m.cos(theta))
    proj_j = (j[0]*m.sin(theta), j[1]*m.cos(theta))
    return abs(utile.distance(i, proj_i)-utile.distance(j, proj_j))


def cost(i, j, p):
    return utile.distance(i, j)*(1 + 0.1*p)


def depth(i, j):
    return max(utile.distance(i, (0, 0)), utile.distance(j, (0, 0)))


def max_depth(inst):
    d = 0
    for i in inst:
        di = utile.distance(i, (0, 0))
        if di > d:
            d = di
    return d


def penalization_function(lw, lc, ld, max_d):
    return lambda i, j, G, p: ((lw * width(i, j, G) + lc * cost(i, j, p))*(depth(i, j)/max_d)**(ld/2))/(1 + p)


def bad_edge(b, p, routes, inst, fixed):
    cand = [0, (0, 0)]
    for r in routes:
        G = gravity_center(r, inst)
        for i in range(len(r)-1):
            pi = r[i]
            pj = r[i+1]
            b_ij = b(inst[pi], inst[pj], G, p[pi][pj])
            if b_ij > cand[0] and pi != 0 and pj != 0 and (pi, pj) not in fixed and (pj, pi) not in fixed:
                cand[0] = b_ij
                cand[1] = (pi, pj)
    return cand


def global_opti(solution, inst, demand,capacity, v, l):
    edges = learn.all_edges(solution)
    fixed_edges = []
    c_init = route.cost_sol(solution, inst, const.quality_cost)
    
    routes = route.copy_sol(solution)
    new_solution = route.copy_sol(routes)
    for e in edges:
        cp = utile.rd_point(e, solution, inst)

        routes = EC.ejection_chain(l, cp, v, routes, inst,
                                demand,capacity, fixed_edges, "DE")
        for i in routes:
            if len(i) == 2:
                routes = EC.reject(i, routes, v, inst, demand,capacity)
        
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i], inst)
        # apply cross-exchange

        routes = CE.cross_exchange(cp, v, routes, inst, demand,capacity, fixed_edges, "DE")

        # apply LK
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i], inst)

        c_final = route.cost_sol(routes, inst, const.quality_cost)
        if c_init - c_final > 0:
            c_init = c_final
            new_solution = route.copy_sol(routes)
    
    return new_solution


def optimisation_heuristic(initial_routes, inst, demand,capacity, lam, mu, nu, l, max_d, v, fixed_edges):
    tps1 = time.time()
    B = [penalization_function(1, 0, 0, max_d), penalization_function(1, 1, 0, max_d), penalization_function(
        1, 0, 1, max_d), penalization_function(1, 1, 1, max_d), penalization_function(0, 1, 0, max_d), penalization_function(0, 1, 1, max_d)]

    b_i = 0
    b = B[b_i]
    p = [[0 for j in range(len(inst))] for i in range(len(inst))]

    detailed_cust = [0 for i in range(len(inst))]
    for r in range(len(initial_routes)):
        for i in initial_routes[r]:
            detailed_cust[i-1] = r
    initial_routes = CW.ClarkeWright(
        initial_routes, inst, demand,capacity, lam, mu, nu, detailed_cust)

    routes = route.copy_sol(initial_routes)
    routes2 = route.copy_sol(routes)
    
    c_init = route.cost_sol(routes, inst,const.quality_cost)
    
    tps2 = time.time()
    tpsGS = time.time()
    tpsCH = time.time()

    while tps2-tps1 < len(demand)/8:
        
        # find the worst edge
        worst = bad_edge(b, p, routes, inst, fixed_edges)[1]

        p[worst[0]][worst[1]] += 1
        p[worst[1]][worst[0]] += 1

        # apply ejection-chain
        cp = utile.rd_point(worst, routes, inst)

        routes = EC.ejection_chain(l, cp, v, routes, inst,
                                demand,capacity, fixed_edges, "RD")
        for i in routes:
            if len(i) == 2:
                routes = EC.reject(i, routes, v, inst, demand,capacity)

        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i], inst)
        # apply cross-exchange

        routes = CE.cross_exchange(cp, v, routes, inst,
                                demand,capacity, fixed_edges, "RD")

        # apply LK
        for i in range(len(routes)):
            routes[i] = LK.LK(routes[i], inst)
        
        #routes = global_opti(routes,inst,demand,v,l)
        c_final = route.cost_sol(routes, inst,const.quality_cost)
        

        if c_final < c_init:
            routes2 = route.copy_sol(routes)  # new optimum
            
            for i in routes2:
                if len(i) == 2:
                    routes2 = EC.reject(i, routes2, v, inst, demand,capacity)
            c_init = route.cost_sol(routes2,inst,const.quality_cost)
            print(round(tps2-tps1,2), round(c_init,3))


            tps1 = time.time()
            tpsCH = time.time()
            tpsGS = time.time()

        if tps2-tpsGS > 10:
            # return to the last best solution, for gs iterations

            routes = route.copy_sol(routes2)

            tpsGS = time.time()

        if tps2-tpsCH > 5:
            tpsCH = time.time()
            b_i += 1

            if b_i < len(B):
                b = B[b_i]
                p = [[0 for j in range(len(inst))]
                     for i in range(len(inst))]

            else:
                b_i = 0
                b = B[b_i]
                p = [[0 for j in range(len(inst))]
                     for i in range(len(inst))]

        tps2 = time.time()

    for i in (routes2):
        if len(i) == 2:
            routes2 = EC.reject(i, routes2, v, inst, demand,capacity)
        if len(i) == 1:
            routes2.remove(i)

    for i in range(len(routes2)):
        routes2[i] = LK.LK(routes2[i], inst)

    return initial_routes, routes2
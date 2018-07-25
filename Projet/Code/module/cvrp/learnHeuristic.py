# Learning heuristic

import math as m
import time
import cvrp.const as const
import cvrp.ReadWrite as rw
import cvrp.utile as utile
import cvrp.learning as learn
import cvrp.route as route
import cvrp.linKernighan as LK
import cvrp.ejectionChain as EC
import cvrp.crossExchange as CE
import cvrp.ClarkeWright as CW
#import cvrp.optimisation as opt
#import cvrp.execution as execute


def learning_heuristic(instance, demand, capacity):
    # compute global variables
    namefile = execute.namefile

    costs = 0
    all_sol = []
    fixed_edges = []
    BaseSolution = []
    tps_deb = time.time()

    # learning
    initial = CW.init_routes(instance, demand)
    edges, param = learn.learning_results(
        0.7, 2, 50, instance, demand, capacity, initial,const.typeBase,const.percent,const.learningCriterion)
    initial_routes = utile.complete(utile.destruction(
        utile.ignore_0(edges)), instance, demand, capacity)

    tps_learn = time.time()

    rw.writef(namefile, 'Learning time = ' + str(tps_learn-tps_deb))

    # start

    cpt = 0
    for i in range(const.NbIterations):
        print(i)
        edges = []
        (lam, mu, nu) = param[0]  # best tuple of the learning phase
        BaseSolution = opt.optimisation_heuristic(
            route.copy_sol(initial_routes), instance, demand, capacity, lam, mu, nu, fixed_edges)

        all_sol += BaseSolution
        # conserve best and worst costs
        stat = [BaseSolution[0][0], BaseSolution[-1][0]]

        # New learning phase
        quality = (stat[1]-stat[0])/10 + stat[0]
        crit = max(const.upBound-cpt/10, const.lowBound)
        cpt += 1
        if crit == const.lowBound:
            cpt = 0
        ls_qual = learn.learning_set_quality(BaseSolution, quality)
        mat_qual = learn.init_matrix(len(instance))
        mat_qual = learn.learn(mat_qual, ls_qual)
        e_qual = learn.mat_info_rg(int(len(demand)*crit), mat_qual)

        initial_routes = utile.complete(utile.destruction(
            utile.ignore_0(e_qual)), instance, demand, capacity)

    # write results in a file

    all_sol.sort()
    tps_fin = time.time()
    print(tps_fin-tps_deb)
    costs = 0
    for i in range(10):
        c_sol, sol = all_sol[i]
        costs += c_sol

        rw.writef(namefile, '')
        rw.writef(namefile, 'res = ' + str(round(c_sol, 3)))
        rw.writef(namefile, 'res_int = ' +
                  str(round(route.cost_sol(sol, instance, "Int"))))
        rw.writef(namefile, 'solution = ' + str(sol))

    rw.writef(namefile, '')
    rw.writef(namefile, 'Mean = ' + str(costs/10))
    rw.writef(namefile, 'Execution = ' + str(tps_fin-tps_deb))
    rw.writef(namefile, '')

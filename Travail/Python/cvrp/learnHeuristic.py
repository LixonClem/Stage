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
import cvrp.optimisation as opt

def learning_heuristic(instance, demand,capacity, l):
    # compute global variables
    namefile = "resultats/Heuristic_results/Values/all/golden7.txt"
    all_sol = []
    tps_deb = time.time()
    max_d = opt.max_depth(instance)
    v = utile.voisins(const.KNN, instance)
    initial = CW.init_routes(instance, demand)
    edges, param = learn.learning_results(0.5, 2, 100, instance, demand,capacity, initial)
    initial_routes = learn.complete(learn.destruction(learn.ignore_0(edges)), instance, demand,capacity)
    tps_learn = time.time()

    rw.writef(namefile, 'Time = ' + str(tps_learn-tps_deb))

    base = []
    costs = 0
    fixed_edges = []
   
    best_cost = route.cost_sol(initial_routes, instance,const.quality_cost)
    for i in range(40):
        print(i)

        (lam, mu, nu) = param[0]
        init, sol = opt.optimisation_heuristic(
            route.copy_sol(initial_routes), instance, demand,capacity, lam, mu, nu, l, max_d, v,fixed_edges)
        base.append(sol)
        c_sol = route.cost_sol(sol, instance,const.quality_cost)
        all_sol.append((c_sol, sol))
        
        if c_sol < best_cost:
            
            best_sol = sol
            best_cost = c_sol


        if i%4 == 0 and i!=0 :
            print("learn")
            edges = []
            fixed_edges = []
            base = []
            mat_qual = learn.init_matrix(len(instance))
            mat_qual = learn.learn(mat_qual, base)
            e_qual = learn.mat_info_rg(int(len(demand)*0.8), mat_qual)
            for e in e_qual:
                if not learn.is_edge_in(e, edges) and not learn.unfeasable_edge(e, edges):
                    edges.append(e)
            initial_routes = learn.complete(learn.destruction(
                learn.ignore_0(edges)), instance, demand,capacity)
            edges, param = learn.learning_results(
                0.8, 2, 100, instance, demand,capacity, initial_routes)
            initial_routes = learn.complete(learn.destruction(
                learn.ignore_0(edges)), instance, demand,capacity)
        
        else :
            print("best learn")
            edges = utile.fixed_alea(learn.all_edges(best_sol),0.95)
            initial_routes = learn.complete(learn.destruction(
                learn.ignore_0(edges)), instance, demand,capacity)
            """
            edges,param = learning_results(0.95,2,100,instance,demand,initial_routes)
            initial_routes = complete(destruction2(
                ignore_0(edges)), instance, demand)
            """    
                

    all_sol.sort()
    tps_fin = time.time()
    print(tps_fin-tps_deb)
    costs = 0
    for i in range(10):
        c_sol, sol = all_sol[i]
        costs += c_sol

        rw.writef(namefile, '')
        rw.writef(namefile, 'res = ' + str(round(c_sol, 3)))
        rw.writef(namefile, 'res_int = ' + str(round(route.cost_sol(sol, instance, "Int"))))
        rw.writef(namefile, 'solution = ' + str(sol))

    rw.writef(namefile, '')
    rw.writef(namefile, 'Mean = ' + str(costs/10))
    rw.writef(namefile, 'Execution = ' + str(tps_fin-tps_deb))
    rw.writef(namefile, '')
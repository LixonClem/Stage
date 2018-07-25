# Code for the Clarke and Wright (CW) heuristic with (lambda, mu, nu) paramaters.

#import cvrp.utile as utile
#import cvrp.route as route

# Compute the classical initial solution of CW


def init_routes(inst, demand):
    solution = []
    for j in range(1, len(inst)):
        routej = [0, j, 0]
        solution.append(routej)
    return solution

# Compute the savings between each couple of customers


def compute_savings(inst, demand, lam, mu, nu):
    savings = []
    d_bar = utile.mean_demand(demand)
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

# Return -1, if we can't merge routes of i and j.
# Return 1 if we can merge routes of i and j ([...,i,j,...])
# Return 2 if we can merge routes of j and i ([...,j,i,...])


def can_merge(i, r1, j, r2, demand, capacity):
    if r1 == r2:
        return -1
    elif (r1[1] == i and r2[len(r2)-2] == j and route.route_demand(r1, demand)+route.route_demand(r2, demand) <= capacity):
        return 1
    elif (r1[len(r1)-2] == i and r2[1] == j and route.route_demand(r1, demand)+route.route_demand(r2, demand) <= capacity):
        return 2
    else:
        return -1

# Merge the routes of i and j, if possible


def merge_routes(i, j, solution, inst, demand, capacity, detailed_cust):
    ir1, ir2 = detailed_cust[i-1], detailed_cust[j-1]
    r1, r2 = solution[ir1].copy(), solution[ir2].copy()
    mrge = can_merge(i, r1, j, r2, demand, capacity)
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
        solution.append(new_road)
        solution[ir1] = []
        solution[ir2] = []
        detailed_cust[i-1] = len(solution)-1
        detailed_cust[j-1] = len(solution)-1
        for k in new_road:
            detailed_cust[k-1] = len(solution)-1

# Apply the CW algorithm


def ClarkeWright(solution, inst, demand, capacity, lam, mu, nu, detailed_cust):
    new_solution = []
    cpt = 1

    savings = compute_savings(inst, demand, lam, mu, nu)
    [s, (i, j)] = savings[-cpt]
    while s > 0 and cpt < len(savings):

        savings[-cpt][0] = 0
        cpt += 1
        merge_routes(i, j, solution, inst, demand, capacity, detailed_cust)
        [s, (i, j)] = savings[-cpt]

    for i in range(len(solution)):
        if solution[i] != []:
            solution[i].pop()
            new_solution.append(solution[i])

    return new_solution

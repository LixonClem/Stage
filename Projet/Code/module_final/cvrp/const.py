# This module contains the constants of the program

import cvrp.ReadWrite as rw
import cvrp.utile as utile


global Error
global KNN
global relocation

global quality_cost  # are distances Int or Float ?
# proportion of edges conserved during the restart phase in the optimisation heuristic
global conserved

# learning heuristic parameters
global NbIterations
global upBound
global lowBound

# learning parameters
global typeBase  # Quality or Quantity ?
global percent
global learningCriterion  # Rank or Threshold ?

global namefile
global limitTime
global restartTime
global resetTime
global maxDepth
global neighbors
global meanDemand

global instance
global demand
global capacity

Error = (0, (0, 0), ([[0], 0], [[0], 0]))
KNN = 30
relocation = 3
quality_cost = "Float"
conserved = 0.7
NbIterations = 25
upBound = 0.8
lowBound = 0.2
typeBase = "Quality"
percent = 10
learningCriterion = "Rank"


# Compute the k nearest neighbors (kNN) for each customer


def voisins(k):
    v = []
    for i in range(len(instance)):
        vi = []
        couples = []
        for j in range(len(instance)):
            if i != j:
                vi.append([utile.distance(instance[i], instance[j]), j])
        vi.sort()
        for l in vi:
            couples.append(l[1])
        v.append(couples[:k])
    return v

# Compute the maximal depth of the instance given


def max_depth():
    d = 0
    for i in instance:
        di = utile.distance(i, (0, 0))
        if di > d:
            d = di
    return d

# Compute the mean of demands of the instance


def mean_demand():
    n = len(demand)
    d = 0
    for i in demand:
        d += i
    return d/(n-1)


def define(fileinstance, startFolder, arrivalFolder, dfile):
    global namefile
    global limitTime
    global restartTime
    global resetTime
    global maxDepth
    global neighbors
    global instance
    global demand
    global capacity
    global meanDemand

    namefile = arrivalFolder+"/"+dfile+".txt"
    print(fileinstance)
    rw.writef(namefile, 'Instance : ' + fileinstance)

    instance, demand, capacity = rw.read(startFolder+'/'+fileinstance)

    limitTime = len(demand)/4
    restartTime = len(demand)/40
    resetTime = len(demand)/100
    maxDepth = max_depth()
    neighbors = voisins(KNN)
    meanDemand = mean_demand()

    return instance, demand, capacity

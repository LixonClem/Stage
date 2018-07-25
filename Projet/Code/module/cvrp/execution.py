# This module executes the algorithm on all instances in a repertory 'toExecute' (or what you want)

import cvrp.utile as utile
import cvrp.ReadWrite as rw
import cvrp.const as const
import cvrp.learnHeuristic as LH
import os.path



def execute(startFolder,arrivalFolder,dfile):
    global namefile
    global limitTime
    global restartTime
    global resetTime
    global maxDepth
    global neighbors
    allinstances = os.listdir(startFolder)
    allinstances.sort()
    print(allinstances)

    for fileinstance in allinstances:
        namefile = arrivalFolder+"/"+dfile+".txt"
        print(fileinstance)
        rw.writef(namefile, 'Instance : ' + fileinstance)
        instance, demand, capacity = rw.read(startFolder+'/'+fileinstance)

        limitTime = len(demand)/4
        restartTime = len(demand)/40
        resetTime = len(demand)/100
        maxDepth = utile.max_depth(instance)
        neighbors = utile.voisins(const.KNN, instance)
        LH.learning_heuristic(instance, demand, capacity)

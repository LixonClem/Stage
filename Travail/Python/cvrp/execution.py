import cvrp.ReadWrite as rw
import cvrp.learnHeuristic as LH
import cvrp.const as const
import os.path


allinstances = os.listdir('toExecute')
allinstances.sort()
print(allinstances)

for fileinstance in allinstances:
    namefile = "resultats/Heuristic_results/Values/all/golden7.txt"
    print(fileinstance)
    rw.writef(namefile, 'Instance : ' + fileinstance)
    instance, demand, capacity = rw.read('toExecute/'+fileinstance)
    print(capacity)
    print("")
    LH.learning_heuristic(instance, demand,capacity, const.relocation)
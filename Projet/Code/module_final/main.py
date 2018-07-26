import cvrp.const as const
import cvrp.learnHeuristic as LH
import os.path

startFolder = 'toExecute2'
arrivalFolder = 'resultats'
dFile = 'GoldenEye'

allinstances = os.listdir(startFolder)
allinstances.sort()

for fileInstance in allinstances:
    instance,demand,capacity = const.define(fileInstance,startFolder,arrivalFolder,dFile)
    LH.learning_heuristic()


# This module contains the constants of the program

import cvrp.ReadWrite as ReadWrite

global Error
global KNN
global relocation

global quality_cost # are distances Int or Float ?
global conserved # proportion of edges conserved during the restart phase in the optimisation heuristic

# learning heuristic parameters
global NbIterations
global upBound
global lowBound

# learning parameters
global typeBase # Quality or Quantity ?
global percent
global learningCriterion # Rank or Threshold ?


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
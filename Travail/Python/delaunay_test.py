import numpy as np
import scipy as sp
from scipy.spatial import Delaunay
import matplotlib.pyplot as py
import random as rd
import math as m

global ylim
global xlim

ylim = 200
xlim = 200

def create_instance(n):
    inst=[[0,0]]
    for i in range(n):
        x = rd.randint(-xlim,xlim)
        y = rd.randint(-ylim,ylim)
        inst.append([x,y])
    return inst

def distance(p1,p2):
    return m.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def print_instance(inst):
    dep = inst[0]
    cust = inst[1:]
    py.plot(dep[0],dep[1], color='blue', marker='o')
    for i in cust:
        py.plot(i[0],i[1], color='red', marker='o')

def aretes(triangle):
    a = []
    a.append((triangle[0],triangle[1]))
    a.append((triangle[1],triangle[2]))
    a.append((triangle[2],triangle[0]))
    return a

def all_aretes(triangles):
    A=[]
    for i in triangles:
        A  = A + aretes(i)
    return A

def suppr_aretes(points,aretes,lim):
    new_aretes=[]
    for i in aretes:
        if (distance(points[i[0]],points[i[1]])<lim) and (i not in new_aretes):
            new_aretes.append(i)
    return new_aretes


points = create_instance(10)
pointsArray = np.array(points)
print_instance(points)

tri = Delaunay(pointsArray)

py.triplot(pointsArray[:,0], pointsArray[:,1], tri.simplices)
py.show()

triangles = [list(tri.simplices[i]) for i in range (len(tri.simplices))]

A = all_aretes(triangles)

faretes = suppr_aretes(points,A,distance([0,0],[200,200])/3)
print(faretes)
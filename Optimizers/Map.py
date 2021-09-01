import numpy as np
from Optimizers.Parameter import *
import matplotlib.pyplot as plt
import copy, math, sys


class Map:
    def __init__(self, fileName) -> None:
        self.__fileName = fileName.split('/')[-1].replace('.txt', '')
        self.__nodes, self.__numNodes = self.readFile(fileName) 
        self.__ttime, self.__dtime = self.create_time_matrix()
    
    @property
    def fileName(self):
        return self.__fileName
    
    @property
    def numNodes(self):
        return self.__numNodes

    @property
    def nodes(self):
        return self.__nodes

    @property
    def ttime(self, i, j):
        return self.ttime[i, j]
    
    @property
    def dtime(self, i, j):
        return self.dtime[i, j]

    def readFile(self, fileName):
        # print("Reading file ...")
        f = open(fileName, "r")

        t = 0
        data = [{'x': 0, 'y': 0}]
        num_nodes = 0
        for x in f:
            if t > 1:
                ys = x.split()
                node = {
                    'x': float(ys[0]),
                    'y': float(ys[1])
                }
                data.append(node)
                #print(f"Node {t-1} is added {ys[0] - ys[1]}")
                num_nodes +=1
            t +=1 
        
        num_nodes += 1
        nodes = np.zeros((num_nodes, 2))

        for i in range(num_nodes):
            nodes[i] = [data[i]['x'], data[i]['y']]

        return nodes, num_nodes
    
    def create_time_matrix(self):
        # print("Calculating time travel...")
        t_time = np.full(( self.numNodes,  self.numNodes), INFINITY) # khoi tao lai ve 0 tai ii 
        d_time = np.full(( self.numNodes,  self.numNodes), INFINITY)

        for i in range( self.__numNodes):
            for j in range( self.__numNodes):
                distance =  math.sqrt(((self.nodes[i, 0] - self.nodes[j, 0]) ** 2 ) + ((self.nodes[i, 1] - self.nodes[j, 1]) ** 2))
                t_time[i][j] = float(distance / T_VEL)
                d_time[i][j] = float(distance / D_VEL)
        
        return t_time, d_time

        


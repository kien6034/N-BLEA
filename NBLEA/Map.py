import numpy as np

from NBLEA.Parameter import *
import matplotlib.pyplot as plt
import copy, math


class Map:
    def __init__(self, fileName) -> None:
        self.nodes = dict() #x,y 
        
        self.base = {'x': 0, 'y': 0}
        self.t_time = None
        self.d_time = None

        self.readFile(fileName)
        self.create_time_matrix()
  
    def readFile(self, fileName):
        # print("Reading file ...")
        f = open(fileName, "r")

        t = 0
        for x in f:
            if t > 1:
                ys = x.split()
                node = {
                    'x': float(ys[0]),
                    'y': float(ys[1])
                }
                self.nodes[t-1] = node
                #print(f"Node {t-1} is added {ys[0] - ys[1]}")
            t +=1 
        
    
    def create_time_matrix(self):
        # print("Calculating time travel...")
        num_length = len(self.nodes) + 2 #node 0 + customer node + node 0

        nodes = copy.deepcopy(self.nodes)

        
        
        #add base to dict
        nodes[0] = {'x': 0, 'y': 0}
        nodes[num_length - 1] = {'x': 0, 'y': 0} 

        
        t_time = np.full((num_length, num_length), INFINITY) # khoi tao lai ve 0 tai ii 
        d_time = np.full((num_length, num_length), INFINITY)


       
        for i in range(num_length):
            for j in range(num_length):
                
                distance =  math.sqrt(((nodes[i]['x'] - nodes[j]['x']) ** 2 )+ ((nodes[i]['y'] - nodes[j]['y']) ** 2))
                t_time[i][j] = float(distance / T_VEL)
                d_time[i][j] = float(distance / D_VEL)
        
        self.t_time = t_time
        self.d_time = d_time

        
    
    def draw(self):
        for i in range(len(self.nodes) + 1):
            if i == 0:
                plt.plot(self.base['x'], self.base['y'], "ro", markersize = 10)
                plt.annotate(i, (self.base['x'], self.base['y']))
                
            else:
                plt.plot(self.nodes[i]['x'], self.nodes[i]['y'], "bo", markersize = 5)
                plt.annotate(i, (self.nodes[i]['x'], self.nodes[i]['y']))
            
     
    


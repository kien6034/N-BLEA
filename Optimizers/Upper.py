import random
from Optimizers.Parameter import *
import numpy as np
import sys
from os import path, mkdir

class Upper:
    def __init__(self, graph) -> None:
        self.graph = graph 

   
    
    def init_Pop(self, popSize, technican_num, create_sample):
        pop = np.zeros((popSize,(self.graph.numNodes + technican_num - 2)), dtype=np.int16)

        for i in range(popSize):
            gen = np.arange(1, (self.graph.numNodes + technican_num - 1))
            np.random.shuffle(gen)
            pop[i] = gen
         
        if create_sample:
            expected_dir = f"low_data/{self.graph.fileName}"

        if not path.exists(expected_dir):
            mkdir(expected_dir)
        
        print(pop[0])
        sys.exit()
        np.save(f"{expected_dir}/route", pop[0])
        np.save(f"{expected_dir}/techican_num", technican_num)

        return pop  

    def run(self, popSize, eliteSize, mutationRate, generations, technican_num, create_sample):
        pop = self.init_Pop(popSize, technican_num, create_sample)
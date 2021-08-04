import sys
import matplotlib.pyplot as plt

from NBLEA import Map as map
from NBLEA import GA, GA_test 
import os

inputFolder = "Instances/"

data = ['6.5.4.txt', '6.10.1.txt', '6.20.1.txt', '6.5.2.txt', '6.20.3.txt', '6.5.1.txt', '6.20.2.txt', '6.10.4.txt', '6.10.3.txt', '6.20.4.txt', '6.10.2.txt']





def nbleaRun(fileDir):
    graph = map.Map(fileDir)

    # sys.exit()

    if 0:
        graph.draw()
        plt.axis('equal')
        plt.show()
        
        sys.exit()

    # ga = GA.GA(graph)
    ga = GA_test.GA(graph)

    ga.run()

for fileName in data:

    fileDir = os.path.join(inputFolder, fileName )
    nbleaRun(fileDir)


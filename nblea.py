import sys
import matplotlib.pyplot as plt

from NBLEA import Map as map
from NBLEA import GA, GA_test 


graph = map.Map("Instances/12.20.1.txt")

# sys.exit()

if 0:
    graph.draw()
    plt.axis('equal')
    plt.show()
    
    sys.exit()

# ga = GA.GA(graph)
ga = GA_test.GA(graph)

ga.run()


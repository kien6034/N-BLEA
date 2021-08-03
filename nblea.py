import sys
import matplotlib.pyplot as plt

from NBLEA import Map as map
from NBLEA import GA, GA_test 


<<<<<<< HEAD

graph = map.Map("Instances/6.5.3.txt")
# graph.draw()
# plt.show()
=======
graph = map.Map("Instances/6.5.3.txt")
>>>>>>> origin/kien_low

# sys.exit()

if 0:
    graph.draw()
    plt.axis('equal')
    plt.show()
    
    sys.exit()

# ga = GA.GA(graph)
ga = GA_test.GA(graph)

ga.run()


import sys
import matplotlib.pyplot as plt

from NBLEA import Map as map
from NBLEA import GA, GA_test 



graph = map.Map("Instances/6.5.1.txt")
# graph.draw()
# plt.show()

# sys.exit()

# ga = GA.GA(graph)
ga = GA_test.GA(graph)

ga.run()
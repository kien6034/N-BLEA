
from NBLEA import Map as map
from NBLEA import GA 

graph = map.Map("Instances/20.10.1.txt")
#graph.draw()

ga = GA.GA(graph)

ga.run()
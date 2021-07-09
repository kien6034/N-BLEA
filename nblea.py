
from NBLEA import Map as map
from NBLEA import GA, GA_test 

graph = map.Map("Instances/20.10.1.txt")
#graph.draw()

# ga = GA.GA(graph)
ga = GA_test.GA(graph)

ga.run()
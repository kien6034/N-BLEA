from NBLEA.Low import *

graph = Map("Instances/20.10.1.txt")
#graph.draw()

t_route = [18, 14, 6, 8, 1, 5, 16, 4, 15, 11, 3, 7, 9, 17, 12, 13, 10, 19, 2]

solver(graph, t_route)

#plt.show()
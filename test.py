from operator import le
import random 
from NBLEA import Low_GA, Map

graph = Map.Map("Instances/6.5.3.txt")

ga = Low_GA.GA(graph, [4, 3 ,6 ,1 ,5 ,2])

ga.run()
# print(2*4 -1)

a = [1, 2, 3 ,4 , 5, 6]
b = [3, 1, 5 ,2 , 4]
c = [a, b] 
# for i in range(len(a)-1):
#     a.insert(2*i + 1, 0)
# print(a)

# c.reverse()
# print(crossover(a, b))

# for i in range(1, 2):
#     print(i)
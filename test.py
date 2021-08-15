from operator import le
from pprint import pprint
import random 
from NBLEA import Low_GA, Map, GA_upper

graph = Map.Map("Instances/6.10.4.txt")
t_route = [2,7,6,1,5,3,4] 
# t_route = [52, 3, 5, 19, 36, 23, 48, 20, 30, 10, 37, 51, 12, 42, 47, 39, 38, 9, 13, 27, 33, 40, 14, 8, 7, 17, 6, 32, 22, 18, 16, 43, 45, 28, 46, 1, 35, 50, 29, 21, 41, 2, 26, 34, 15, 24, 49, 25, 44, 31, 4, 11] 

ga = Low_GA.GA(graph, t_route)

cost, uav_tour, route_detail = ga.run()


pprint(route_detail)

print(f't route: {t_route}')
print(f'uav_tour: {uav_tour}')
print(cost)



# ga = GA_upper.GA(graph)
# ga.run()

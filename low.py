from NBLEA.Low import *

graph = Map("Instances/6.5.1.txt")
#graph.draw()

t_route = [6, 5, 1, 4 ,3 ,2]

fitness, max_cost, best_u_tour = solver(graph, t_route)  

#fitness = 1 - (cost khi có drone hỗ trợ / max_cost)
#max_cost: Tổng thời gian chờ của mẫu ứng với hành trình t_route khi không có drone hỗ trợ 
#best_u_tour: hành trình của uav 

print("=====================================")
print(fitness, max_cost)

cost = (1 - fitness) * max_cost
print(cost)

print(best_u_tour)

#plt.show()
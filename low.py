from NBLEA.Low import *

graph = Map("Instances/20.10.1.txt")
#graph.draw()

t_route = [18, 14, 6, 8, 1, 5, 16, 4, 15, 11, 3, 7, 9, 17, 12, 13, 10, 19, 2]

fitness, max_cost, best_u_tour = solver(graph, t_route)  

#fitness = 1 - (cost khi có drone hỗ trợ / max_cost)
#max_cost: Tổng thời gian chờ của mẫu ứng với hành trình t_route khi không có drone hỗ trợ 
#best_u_tour: hành trình của uav 

print(fitness, max_cost, best_u_tour)
#plt.show()
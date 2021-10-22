import pprint
import matplotlib.pyplot
import time

from Optimizers.Lower import Lower
from main import l_params
from Optimizers.Map import Map

graph = Map("Instances/6.5.4.txt")
if 0:
    graph.draw()
    plt.show()
    sys.exit()

t_route = [3, 6, 5, 2, 4, 1, 7]

start = time.time()

lower_ga = Lower(graph, t_route, l_params, 2)
cost, best_u_tour, best_route_details = lower_ga.run(popSize=l_params['pop_size'], 
                                                        eliteSize=l_params['elite_size'], 
                                                        mutationRate=l_params['mutation_rate'], 
                                                        generations=l_params['generations'])

# from NBLEA.Low import *

# graph = Map("Instances/6.5.3.txt")
# if 0:
#     graph.draw()
#     plt.show()
#     sys.exit()

# t_route = [7, 4, 3, 6, 1, 5, 2] 

# cost, best_route_details, best_u_tour = solver(graph, t_route)  

end = time.time()

pprint.pprint(best_route_details)

#fitness = 1 - (cost khi có drone hỗ trợ / max_cost)
#max_cost: Tổng thời gian chờ của mẫu ứng với hành trình t_route khi không có drone hỗ trợ 
#best_u_tour: hành trình của uav 

print("=====================================")
print(f't route: {t_route}')
print(f'uav_tour: {best_u_tour}')
print(cost)
print(f'lower Ga run in {end - start}')

#plt.show()
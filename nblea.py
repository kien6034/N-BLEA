import sys
import matplotlib.pyplot as plt

from NBLEA import Map as map
from NBLEA import GA, GA_test, Low_GA, utils
import pprint

graph = map.Map("Instances/12.20.1.txt")

# sys.exit()

if 0:
    graph.draw()
    plt.axis('equal')
    plt.show()
    
    sys.exit()

# ga = GA.GA(graph)
ga = GA_test.GA(graph)
# for _ in range(20):
time_diff, best_fitness, best_t_tour, best_uav_tour, best_route_details = ga.run()
# instance, run_time, number_of_tech, total_wait_time, total_work_time, tech_tour, uav_tour = utils.get_result(graph, time_diff, best_fitness, best_t_tour, best_uav_tour, best_route_details)
# utils.save_solution(instance, run_time, number_of_tech, total_wait_time, total_work_time, tech_tour, uav_tour)



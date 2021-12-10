import pprint
import time
import random

from Optimizers.Lower import Lower
from Optimizers.Utils import save_stats
from main import l_params, run_type
from Optimizers.Map import Map

graph = Map("Instances/200.20.3.txt")
run_num = 3
print(graph.numNodes)

def gen_routes(Node_num, tech_num, graph, sample_num):
    routes = []
    for _ in range(sample_num):
        new_route = list(range(1, graph.numNodes + tech_num - 1))
        random.shuffle(new_route)
        routes += [new_route]
    return routes

t_routes = gen_routes(graph.numNodes, run_type['technican_num'], graph, 10)
for r in t_routes:
    print('running low GA.....')
    start = time.time()

    lower_ga = Lower(graph, r, l_params, run_type['technican_num'])
    cost, best_u_tour, best_route_details, record = lower_ga.run(popSize=l_params['pop_size'], 
                                                            eliteSize=l_params['elite_size'], 
                                                            mutationRate=l_params['mutation_rate'], 
                                                            generations=l_params['generations'],
                                                            save_stats=True)
    runtime = time.time() - start
    save_stats(instance=graph.fileName, 
            version=run_type['run_version'],
            run_time=runtime, 
            tech_num=run_type['technican_num'], 
            work_time=l_params['work_time'], 
            level='lower', 
            record=record)

# t_route = [3, 6, 5, 2, 4, 1, 7] -> 3652417

{
    '3652417': 100
}

# start = time.time()

# lower_ga = Lower(graph, t_route, l_params, 2)
# cost, best_u_tour, best_route_details = lower_ga.run(popSize=l_params['pop_size'], 
#                                                         eliteSize=l_params['elite_size'], 
#                                                         mutationRate=l_params['mutation_rate'], 
#                                                         generations=l_params['generations'])

# from NBLEA.Low import *
# graph = Map("Instances/6.5.3.txt")
# t_route = [7, 4, 3, 6, 1, 5, 2] 
# cost, best_route_details, best_u_tour = solver(graph, t_route)  

# end = time.time()

# pprint.pprint(best_route_details)

#fitness = 1 - (cost khi có drone hỗ trợ / max_cost)
#max_cost: Tổng thời gian chờ của mẫu ứng với hành trình t_route khi không có drone hỗ trợ 
#best_u_tour: hành trình của uav 

# print("=====================================")
# print(f't route: {t_route}')
# print(f'uav_tour: {best_u_tour}')
# print(cost)
# print(f'lower Ga run in {end - start}')

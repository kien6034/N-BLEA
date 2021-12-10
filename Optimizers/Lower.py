import copy
import operator
import random
import numpy as np
from Optimizers.Parameter import INFINITY

import sys


class Lower:
    def __init__(self, graph, t_route, l_params, technican_num) -> None:
        self.graph = graph
        self.t_route = t_route
        self.l_params = l_params
        self.technican_num = technican_num

    def get_specific_route(self, t_route):

        routes = {}

        num_nodes = len(t_route) - (self.technican_num - 1)
        sub_end_points = list(range(num_nodes + 1, num_nodes + self.technican_num))


        for tid in range(self.technican_num):
            routes[tid] = []

        k = 0
        for index in range(len(t_route)):  
            if t_route[index] in sub_end_points:
                k += 1
            else:
                routes[k].append(t_route[index]) 
        
        new_routes = {}
        for tid in routes:
            if routes[tid]: 
                new_routes[len(new_routes)] = routes[tid]
            else:
                self.technican_num -= 1
                
        return new_routes

    def sort_by_time(self, specific_routes):
        routes = {}
        t_back_time = {}

        for tid in range(self.technican_num):
            path = specific_routes[tid]

            travel_time = 0 
            for index, node in enumerate(path):
                if index == 0:
                    travel_time += self.graph.ttime[0][node]
                    routes[node] = travel_time
                else:
                    travel_time += self.graph.ttime[path[index - 1]][path[index]]
                    routes[node] = travel_time
        
            t_back_time[tid] = travel_time + self.graph.ttime[path[-1]][0] #time that each technican gets back to base {'0': 132.3, '1': 212.23}

        sorted_routes = dict(sorted(routes.items(), key= lambda item: item[1])) #route sorted in time order
    
        #find max cost 
        max_cost = 0 #total cost when there are no uav support
        for tid in range(self.technican_num):
            for node in specific_routes[tid]:
                max_cost += t_back_time[tid] - sorted_routes[node] 

        return sorted_routes, t_back_time, max_cost

    def get_fitness(self, idv, sorted_routes, specific_routes):
        routes = list(sorted_routes.keys())
    

        search_space = copy.deepcopy(sorted_routes)
        # search_space = sorted_routes[:]
       

        C = []

        for index in range(len(routes) -1):
            C.append(routes[index])
            C.append(0)
        
        C.append(routes[-1])
        # print(C)


        k = 0 # so luong hanh trinh cua uav
        uav_tour = {} 
        uav_tour[k] = []
        uav_tour[k].append(0)

        u_time = 0
        endurance = 0

        #route detail
        route_details = {}
        route_details['time_at_node'] = {}
        route_details['uav_route'] = []

        #[4, 0, 3, 0, 6, 0, 1, 0, 5, 0, 2]
        test_chosen_idx = [0, 1, 2, 3, 4,5,6,7,8,9]

        # cost = {}

        for i in range(0, len(C)):
            src = uav_tour[k][-1] #get last elemetn of uav tour k
            #for debug
            next_node = C[i] 
            # t_time = search_space[next_node]

            if C[i] == src: #if 2 node 0 in a row 
                continue

            if idv[i]: 
            #if i in test_chosen_idx:
                #expected destination and expected travel time 
                e_des = C[i]
                e_travel_time = self.graph.dtime[src][e_des]

                if e_des == 0:
                    u_time += e_travel_time
                    endurance += e_travel_time
                else:
                    #constrain 1: T
                    if endurance + e_travel_time + self.graph.dtime[e_des][0] > self.l_params['drone_time']: 
                        #if uav cannot flyback to base when chosing the e_des as next node 
                        route_details['time_at_node'][next_node] = (search_space[next_node], -1)
                        continue
                    else:
                        route_details['time_at_node'][next_node] = (search_space[next_node], u_time + e_travel_time)
                    
                    e_uav_arrive_time = u_time + e_travel_time #expected uav arrival time 
                     

                    if self.l_params['technican_can_wait']:
                        if e_uav_arrive_time > search_space[e_des]: #if uav come after technican
                            #TODO: make small func additional time that technican have to wait for uav at edes
                            for sid in specific_routes:
                                subtour = specific_routes[sid]

                                if e_des in subtour:
                                    index = subtour.index(e_des)           
                                    for updating_node in subtour[index+1 :(len(subtour))]:
                                        search_space[updating_node] += e_uav_arrive_time - search_space[e_des]

                            u_time = e_uav_arrive_time
                            endurance += e_travel_time
                            
                        else: #uav come before
                            u_time = search_space[e_des]
                            endurance += e_travel_time + (search_space[e_des]- e_uav_arrive_time) 

                    else:
                        # UAV arrive before techincan
                        if e_uav_arrive_time  > search_space[e_des]:
                            continue
                        
                        u_time = search_space[e_des]
                        endurance += e_travel_time

                uav_tour[k].append(e_des)
 
                #check if close subtour  
                if uav_tour[k][-1] == 0:
                    sub_route_data = {
                        'k': k,
                        'route': uav_tour[k],
                        'endurance': endurance
                    }
                    route_details['uav_route'].append(sub_route_data)

                    #create new subtour 
                    k = k + 1
                    uav_tour[k] = []
                    uav_tour[k].append(0)

                    #reset endurance 
                    endurance = 0
            else:
                if next_node != 0:
                    route_details['time_at_node'][next_node] = (search_space[next_node], -1)
            
        

        if uav_tour[k][-1] != 0:
            uav_tour[k].append(0)

            u_time += self.graph.dtime[uav_tour[k][-1]][0]

        if len(uav_tour[k]) == 1: #case when there are only start node 0
            del uav_tour[k]

        
        
        lattest_node = list(search_space.keys())[-1]
        work_time = search_space[lattest_node] + self.graph.ttime[lattest_node][0]
        
        if work_time > self.l_params['work_time']:
            cost = INFINITY
        else:
            cost, wait_times = self.find_cost(time_at_nodes=route_details['time_at_node'], uav_tour=uav_tour, specific_routes=specific_routes)
            route_details['wait_times'] = wait_times
        return cost, route_details, uav_tour, work_time
    
    def find_cost(self, time_at_nodes, uav_tour, specific_routes):

        t_back_time = {}
        
        specific_route = copy.deepcopy(specific_routes)
        # specific_route = specific_routes[:]
        wait_times = {}
       
        # print(time_at_nodes)
        for tid in specific_routes:
            
            path = specific_routes[tid]
            last_node = path[-1]
            
            
            time_leave_last_node = max(time_at_nodes[last_node][0],time_at_nodes[last_node][1] )
            t_back_time[tid] = time_leave_last_node + self.graph.ttime[last_node][0]

        
        #iterate over each subtour
        for stid in uav_tour:
            subtour = uav_tour[stid]
            
            #caculate time when uav back to base 
            last_sub_node_idx = len(subtour) - 2 #last node's index of uav subtour (except 0)
            back_time = max(time_at_nodes[subtour[last_sub_node_idx]][0],time_at_nodes[subtour[last_sub_node_idx]][1] )  + self.graph.dtime[subtour[last_sub_node_idx]][0] 


            #iterate over each non-zero node in above subtour
            for unid in range(1, (len(subtour) - 1)):
                #iterate over each technican route 
                for tid in specific_route:
                    index = -1

                    #if subtour node in route of tid technican 
                    if subtour[unid] in specific_route[tid]:
                        index = specific_route[tid].index(subtour[unid])

                        for finished_node in specific_route[tid][0:(index+1)]:
                            wait_times[finished_node] = (back_time - time_at_nodes[finished_node][0], 'uav')
                        
                        del specific_route[tid][0:(index+1)]

        #node that are brought back by technican
        for tid in specific_route:
            for node in specific_route[tid]:
                wait_times[node] = ((t_back_time[tid] - max(time_at_nodes[node][0],time_at_nodes[node][1] )), 'technican')
        
        cost = 0
        for idx in wait_times:
            cost += wait_times[idx][0]
        
        return cost, wait_times

    def init_pop(self, popSize, sorted_routes, preOptima): 
        populations = np.zeros((popSize, 2*len(sorted_routes)-1), dtype=np.int8)
        for i in range(len(preOptima)):
            populations[i] = preOptima[i]
        for i in range(len(preOptima), popSize):
            new_idv = np.zeros(2*len(sorted_routes)-1, dtype=np.int8)
            for j in range(len(new_idv)):
                new_idv[j] = random.randint(0, 1)
            while new_idv.tolist() in populations.tolist():
                for j in range(len(new_idv)):
                    new_idv[j] = random.randint(0, 1)
            populations[i] = new_idv
        return populations

    def mutate(self, c, prob):
        new_child = c[:]
        if random.random() < prob:
            idx = random.randint(0, len(c) - 1)
            
            if new_child[idx]:
                new_child[idx] = 0
            else:
                new_child[idx] = 1
        return new_child

    def crossover(self, p1, p2):
        c1, c2 = np.zeros(len(p1), dtype=np.int8), np.zeros(len(p2), dtype=np.int8)
        for i in range(len(p1)):
            idx = random.randint(0, 1)
            if idx:
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]

        return c1, c2

    def tournament_selection(self, pop, pop_fitness):
        # Selecting randomly 4 individuals to select 2 parents by a binary tournament
        parentIds = set()
        while len(parentIds) < 4:
            parentIds |= {random.randint(0, len(pop) - 1)}
        parentIds = list(parentIds)
        # Selecting 2 parents with the binary tournament
        parent1 = pop[parentIds[0]] if pop_fitness[parentIds[0]] < pop_fitness[parentIds[1]] else pop[parentIds[1]]
        parent2 = pop[parentIds[2]] if pop_fitness[parentIds[2]] < pop_fitness[parentIds[3]] else pop[parentIds[3]]
        return parent1, parent2

    def rank_pop(self, pop, pop_fitness):
        fitness_results = {}
        
        for i in range(0, len(pop)):
            fitness_results[i] = pop_fitness[i]

        pop_ranked = sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = False)
        return list(map(lambda x: x[0], pop_ranked))

    def select_elite(self, pop_ranked, pop, eliteSize):
        elite_pop = np.zeros((eliteSize, len(pop[0])), dtype=np.int8)
        for i, idx in enumerate(pop_ranked[:eliteSize]):
            elite_pop[i] = pop[idx]
        return elite_pop

    def run(self, popSize, eliteSize, mutationRate, generations, preOptima=[], check_dup = True, save_stats=False):
        specific_route = self.get_specific_route(self.t_route)
        sorted_routes, t_back_time, max_cost = self.sort_by_time(specific_route)
        pop = self.init_pop(popSize, sorted_routes, preOptima)

        # print(specific_route)
        # print(self.sort_by_time(specific_route))
        # print(pop)
        # print(pop[0], pop[1])
        # print(self.crossover(pop[0], pop[1]))
        # print(self.mutate(pop[0], mutationRate))

        record = []
        fitness_results = []

        for i in range(generations):

            pop_fitness = []
            uav_tours = []
            pop_details = []

            # cal_pop = {}

            for idv in pop:

                # idv = ''.join(map(str, idv))
                # if idv in cal_pop:
                #     pop_fitness += [cal_pop[idv][0]]
                #     uav_tours += [cal_pop[idv][1]]
                #     pop_details += [cal_pop[idv][2]]
                # else:

                cost, route_details, uav_tour, work_time = self.get_fitness(idv, sorted_routes, specific_route)
                route_details['number_of_tech'] = self.technican_num
                route_details['work_time'] = work_time
            
                pop_fitness += [cost]
                uav_tours += [uav_tour]
                pop_details += [route_details]

                    # cal_pop[idv] = (cost, uav_tour, route_details)
            
            best_idx = self.rank_pop(pop, pop_fitness)[0]
            best_cost = pop_fitness[best_idx]
            # best_tour = uav_tours[best_idx]
            best_tour = pop[best_idx]
            best_route_detail = pop_details[best_idx]
            
            # print('best cost:', best_cost)
            # print('uav_tour:', best_tour)
            fitness_results += [best_cost]
            if fitness_results.count(best_cost) != len(fitness_results):
                fitness_results.clear()
            if save_stats:
                record += [best_cost]

            pop_ranked = self.rank_pop(pop, pop_fitness)
            next_pop = self.select_elite(pop_ranked, pop, eliteSize)
            while len(next_pop) < popSize - eliteSize:
                p1, p2 = self.tournament_selection(pop, pop_fitness)
                c1, c2 = self.crossover(p1, p2)
                if check_dup:
                    while c1.tolist() in next_pop.tolist() and c2.tolist() in next_pop.tolist():
                        # print("l", c1, c2)
                        p1, p2 = self.tournament_selection(pop, pop_fitness)
                        c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                # next_pop += [c1, c2]
                next_pop = np.concatenate((next_pop, np.array([c1, c2], dtype=np.int8)))
            # elite_pop = self.select_elite(pop_ranked, pop, eliteSize)
            # next_pop += elite_pop
            pop = next_pop
            if len(fitness_results) >= 10:
                # print(f'lower {i}')
                return best_cost, best_tour, best_route_detail

        if save_stats:
            return best_cost, best_tour, best_route_detail, record
        return best_cost, best_tour, best_route_detail

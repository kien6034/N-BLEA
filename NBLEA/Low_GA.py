import sys
from NBLEA.Map import *
from NBLEA.Parameter import *
import copy, random
import operator

ACTIVE_TECH = None

def get_specific_route(t_route):
    global ACTIVE_TECH
    ACTIVE_TECH = TECHNICAN_NUMS
    # routes = dict()

    # for tid in range(TECHNICAN_NUMS):
    #     routes[tid] = list()

    # k = 0
    # for index in range(len(t_route)):        
        
    #     routes[k].append(t_route[index])
    
    #     k +=1
    #     if k == TECHNICAN_NUMS:
    #         k = 0
    # return routes

    routes = dict()

    num_nodes = len(t_route) - 1
    sub_end_points = list(range(num_nodes + 1, num_nodes + TECHNICAN_NUMS))


    for tid in range(TECHNICAN_NUMS):
        routes[tid] = list()

    k = 0
    for index in range(len(t_route)):  
        if t_route[index] in sub_end_points:
            k += 1
        else:
            routes[k].append(t_route[index]) 
    
    new_routes = dict()
    for tid in routes:
        if routes[tid]: 
            new_routes[len(new_routes)] = routes[tid]
        else:
            ACTIVE_TECH -= 1
            
    return new_routes

def sort_by_time(graph, specific_routes):
    # routes = dict()
    # t_back_time = dict()
    
    # for tid in range(TECHNICAN_NUMS):
    #     path = specific_routes[tid]

    #     travel_time = 0 
    #     for index, node in enumerate(path):
    #         if index == 0:
    #             travel_time += graph.t_time[0][node]
    #             routes[node] = travel_time
    #         else:
    #             travel_time += graph.t_time[path[index - 1]][path[index]]
    #             routes[node] = travel_time

        
    #         t_back_time[tid] = travel_time + graph.t_time[path[-1]][0] #time that each technican gets back to base {'0': 132.3, '1': 212.23}

    # sorted_routes = dict(sorted(routes.items(), key= lambda item: item[1])) #route sorted in time order
 
    # #find max cost 
    # max_cost = 0 #total cost when there are no uav support
    # for tid in range(TECHNICAN_NUMS):
    #     for node in specific_routes[tid]:
    #         max_cost += t_back_time[tid] - sorted_routes[node] 

    # return sorted_routes, t_back_time, max_cost
    routes = dict()
    t_back_time = dict()

    for tid in range(ACTIVE_TECH):
        path = specific_routes[tid]

        travel_time = 0 
        for index, node in enumerate(path):
            if index == 0:
                travel_time += graph.t_time[0][node]
                routes[node] = travel_time
            else:
                travel_time += graph.t_time[path[index - 1]][path[index]]
                routes[node] = travel_time
       
        t_back_time[tid] = travel_time + graph.t_time[path[-1]][0] #time that each technican gets back to base {'0': 132.3, '1': 212.23}

    sorted_routes = dict(sorted(routes.items(), key= lambda item: item[1])) #route sorted in time order
 
    #find max cost 
    max_cost = 0 #total cost when there are no uav support
    for tid in range(ACTIVE_TECH):
        for node in specific_routes[tid]:
            max_cost += t_back_time[tid] - sorted_routes[node] 

    return sorted_routes, t_back_time, max_cost 

class GA():
    def __init__(self, graph, t_route) -> None:
        self.t_route = t_route
        self.graph = graph


    def fitness(self, sorted_routes, uav_tour, ispecific_routes, t_back_time, max_cost): 

        specific_routes = copy.deepcopy(ispecific_routes)

        cost = max_cost 

    
        for stid in uav_tour:
            #iterate each uav subtour 
            subtour = uav_tour[stid]
            
            last_sub_node_idx = len(subtour) - 2 #last node's index of uav subtour
    
            #caculate time when uav back to base 
         
            back_time = sorted_routes[subtour[last_sub_node_idx]] + self.graph.d_time[subtour[last_sub_node_idx]][0] 

            #iterate each node in uav subtour
            for unid in range(1, len(subtour) -1):

                #iterate over each route of technican
                for tid in specific_routes:
                    index = -1
                    if subtour[unid] in specific_routes[tid]: #if uav take sample from technican tid
                        index = specific_routes[tid].index(subtour[unid])

                        #reduce cost 
                        cost -= ((index +1) * (t_back_time[tid] - back_time))
                        # for finished_node in specific_routes[tid][0:(index+1)]:
                        #     cost -= t_back_time[tid] - back_time
                        
                        #delete finished node 
                        del specific_routes[tid][0:(index+1)]

        #TODO: T

        #print(sorted_routes)
        #print(uav_tour)
        

        # return (1 - cost / max_cost)
        return cost

    def get_fitness(self, idv, sorted_routes, specific_routes):
        routes = list(sorted_routes.keys())

        search_space = copy.deepcopy(sorted_routes)
        search_space[0] = 0

        C = list()

        for index in range(len(routes) -1):
            C.append(routes[index])
            C.append(0)
        
        C.append(routes[-1])
        # print(C)


        k = 0 # so luong hanh trinh cua uav
        uav_tour = dict() 
        uav_tour[k] = list()
        uav_tour[k].append(0)

        u_time = 0
        endurance = 0

        #route detail
        route_details = dict()
        route_details['time_at_node'] = dict()
        route_details['uav_route'] = list()

        #[4, 0, 3, 0, 6, 0, 1, 0, 5, 0, 2]
        test_chosen_idx = [0, 1, 2, 3, 4,5,6,7,8,9]

        # cost = dict()

        for i in range(0, len(C)):
            src = uav_tour[k][-1] #get last elemetn of uav tour k
            #for debug
            next_node = C[i] 
            t_time = search_space[next_node]

            if C[i] == src: #if 2 node 0 in a row 
                continue

            if idv[i]: 
            #if i in test_chosen_idx:
                #expected destination and expected travel time 
                e_des = C[i]
                e_travel_time = self.graph.d_time[src][e_des]

                if e_des == 0:
                    u_time += e_travel_time
                    endurance += e_travel_time
                else:
                    #constrain 1: T
                    #route details
                    route_details['time_at_node'][next_node] = (search_space[next_node], u_time + e_travel_time)

                    if endurance + e_travel_time + self.graph.d_time[e_des][0] > T: #make sure uav can fly back to base 
                        
                        continue
                    
                    e_uav_arrive_time = u_time + e_travel_time #expected uav arrival time 
                     

                    if TECHNICAN_CAN_WAIT:
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
                    uav_tour[k] = list()
                    uav_tour[k].append(0)

                    #reset endurance 
                    endurance = 0
            else:
                if next_node != 0:
                    route_details['time_at_node'][next_node] = (search_space[next_node], -1)
            
        

        if uav_tour[k][-1] != 0:
            uav_tour[k].append(0)

            u_time += self.graph.d_time[uav_tour[k][-1]][0]

        if len(uav_tour[k]) == 1: #case when there are only start node 0
            del uav_tour[k]

        cost, wait_times = self.find_cost(time_at_nodes=route_details['time_at_node'], uav_tour=uav_tour, specific_routes=specific_routes)
        route_details['wait_times'] = wait_times
        return cost, route_details, uav_tour
    
    def find_cost(self, time_at_nodes, uav_tour, specific_routes):

        t_back_time = dict()
        
        specific_route = copy.deepcopy(specific_routes)
        wait_times = dict()
       
        
        for tid in specific_routes:
            
            path = specific_routes[tid]
            last_node = path[-1]
            
            
            time_leave_last_node = max(time_at_nodes[last_node][0],time_at_nodes[last_node][1] )
            t_back_time[tid] = time_leave_last_node + self.graph.t_time[last_node][0]

        
        #iterate over each subtour
        for stid in uav_tour:
            subtour = uav_tour[stid]
            
            #caculate time when uav back to base 
            last_sub_node_idx = len(subtour) - 2 #last node's index of uav subtour (except 0)
            back_time = max(time_at_nodes[subtour[last_sub_node_idx]][0],time_at_nodes[subtour[last_sub_node_idx]][1] )  + self.graph.d_time[subtour[last_sub_node_idx]][0] 


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

    def create_individual(self, sorted_routes):
        idv = []
        # for _ in range(len(self.t_route)):
        for _ in range(2*len(sorted_routes)-1):
            idv += [random.randint(0, 1)]
        return idv

    def init_pop(self, popSize, sorted_routes): 
        populations = []
       
        while len(populations) < popSize:
            new_idv = self.create_individual(sorted_routes)
            if new_idv not in populations:
                populations.append(new_idv)

        return populations  

    def rank_pop(self, pop, pop_fitness):
        fitness_results = {}
        
        for i in range(0, len(pop)):
            fitness_results[i] = pop_fitness[i]

        return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = False)

    def select_elite(self, pop_ranked, pop, eliteSize):
        elite_pop = []
        for i in pop_ranked[:eliteSize]:
            elite_pop.append(pop[i[0]])
        return elite_pop

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

    def mutate(self, c, prob):
        new_child = c
        if random.random() < prob:
            idx = random.randint(0, len(c) - 1)
            if new_child[idx]:
                new_child[idx] = 0
            else:
                new_child[idx] = 1
        return new_child

    def crossover(self, p1, p2):
        c1, c2 = [], []
        for i in range(len(p1)):
            idx = random.randint(0, 1)
            if idx:
                c1 += [p1[i]]
                c2 += [p2[i]]
            else:
                c1 += [p2[i]]
                c2 += [p1[i]]

        return c1, c2

    def get_uav_route(self, idv, sorted_routes, specific_routes):
        uav_tour = dict()
        route = list(sorted_routes.keys())
        for i in range(len(route)-1):
            route.insert(2*i+1, 0)

        sub_tour = []
        sub_tech = []
        t_fly = 0
        for i in range(0, len(idv), 2):
        # for i in range(len(idv)):
            # print(sub_tour)

            t_fly = 0 if len(sub_tour) == 0 else self.graph.d_time[0][sub_tour[0]]
            for k in range(1, len(sub_tour)):
                t_fly += self.graph.d_time[sub_tour[k-1]][sub_tour[k]]
            if idv[i] and idv[i-1] == 0:
                pre_node = 0 if len(sub_tour) == 0 else sub_tour[len(sub_tour)-1]
                t_fly += self.graph.d_time[pre_node][route[i]]
                if t_fly > T:
                    idv[i-1] = 1
                    t_fly = 0
            
            if idv[i]:
                # uav_tour[len(uav_tour)] = [0, route[i], 0]
                if i != 0 and idv[i-1] and len(sub_tour) > 0:
                    uav_tour[len(uav_tour)] = [0] + sub_tour + [0]
                    sub_tour = []
                sub_tour += [route[i]]
                if i == len(idv)-1 and idv[i-1] and len(sub_tour) > 0:
                    uav_tour[len(uav_tour)] = [0] + sub_tour + [0]
                    sub_tour = []


            
        return uav_tour

    def run(self, popSize = 50, eliteSize = 2, mutationRate = 0.2, generations = 50):
        specific_routes = get_specific_route(self.t_route)
        sorted_routes, t_back_time, max_cost = sort_by_time(self.graph, specific_routes)
        pop = self.init_pop(popSize, sorted_routes)

        # print(specific_routes)
        # print(sorted_routes)
        # print(self.get_uav_route(pop[0], sorted_routes, specific_routes))
        # print(pop[0])

        for i in range(generations):

            pop_fitness = []
            uav_tours = []
            pop_details = []
            for idv in pop:
                # uav_tour = self.get_uav_route(idv, sorted_routes, specific_routes)
                cost, route_details, uav_tour = self.get_fitness(idv, sorted_routes, specific_routes)
                # cost = self.fitness(sorted_routes, uav_tour, specific_routes, t_back_time, max_cost)
                pop_fitness += [cost]
                uav_tours += [uav_tour]
                pop_details += [route_details]

            # print(pop)
            best_idx = self.rank_pop(pop, pop_fitness)[0][0]
            best_cost = pop_fitness[best_idx]
            best_tour = uav_tours[best_idx]
            best_route_detail = pop_details[best_idx]
            # best_tour = self.get_uav_route(pop[best_idx], sorted_routes, specific_routes)
            # print('best cost:', best_cost)
            # print('uav_tour:', best_tour)
            
            #chosing parent 
            #cross over with prob pc 
            # mutation with prob pc
            #decode and fitness calculation 
            #survivor selection
            #find best

            next_pop = []
            while len(next_pop) < popSize - eliteSize:
                p1, p2 = self.tournament_selection(pop, pop_fitness)
                c1, c2 = self.crossover(p1, p2)
                # while c1 in pop and c2 in pop:
                #     p1, p2 = self.tournament_selection(pop, pop_fitness)
                #     c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                next_pop += [c1, c2]
            pop_ranked = self.rank_pop(pop, pop_fitness)
            elite_pop = self.select_elite(pop_ranked, pop, eliteSize)
            next_pop += elite_pop
            pop = next_pop

        return best_cost, best_tour, best_route_detail

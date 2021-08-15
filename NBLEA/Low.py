from operator import sub
from numpy.lib.twodim_base import tril_indices_from

from numpy.lib.type_check import common_type
from NBLEA.Map import *
from NBLEA.Parameter import *
import copy, random

import matplotlib.pyplot as plt
import numpy as np
import sys, pprint

graph = None
# graph = Map("Instances/6.5.1.txt")

ACTIVE_TECH = None

def draw(routes):
    colors = ["green", "black", "red", "orange"]
        

    for tid in range(ACTIVE_TECH):
        path = routes[tid]

        for i in range(len(path)):
            if i == 0:
                x = (graph.base['x'], graph.nodes[path[i]]['x'])
                y = (graph.base['y'], graph.nodes[path[i]]['y'])

                plt.plot(x, y, colors[tid], linewidth= 1) 
            else:
                x = (graph.nodes[path[i-1]]['x'], graph.nodes[path[i]]['x'])
                y = (graph.nodes[path[i-1]]['y'], graph.nodes[path[i]]['y'])

                plt.plot(x, y, colors[tid], linewidth= 1)
            

def get_specific_route(t_route):
    global ACTIVE_TECH
    ACTIVE_TECH = TECHNICAN_NUMS
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

def sort_by_time(specific_routes):
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


def init_pheromones(it_route):
    pheromones = dict()
    
    t_route = copy.deepcopy(it_route)
    t_route.append(0)
    # t_route.append(len(t_route))
    # print(t_route)

    for nodei in t_route:
        for nodej in t_route:
            pheromones[(nodei, nodej)] = INIT_PHEROMONE

    return pheromones


def global_pheromone_update(best_u_tour, pheromones, fitness):
    g_pheromones = pheromones
    for stid in best_u_tour:
        tour = best_u_tour[stid]

        for i in range(1, len(tour)):   
            g_pheromones[(tour[i-1], tour[i])] +=   DEPOSIT_RATE
            
    
    for p in g_pheromones:
        g_pheromones[p] -= GLOBAL_EVAPORATION_RATE

    return g_pheromones


###########################################################
###########################################################
########################################################### 
def solver(sgraph, t_route):
    #update graph var
    global graph
    graph = sgraph
    
    specific_routes = get_specific_route(t_route)  #specific route: route corresponding to technicans {'0': path, '1': path} 

    sorted_routes, t_back_time, max_cost = sort_by_time(specific_routes)
    #sorted routes: time order of visted nodes 
    #  {5: 3.5293766715800725, 4: 4.3167745828325685, 3: 8.169203692638696, 2: 9.147779280116103, 6: 16.245709046965185, 1: 20.170591373731998}
    #t_back_time: time when each technicans get back to base 
    # {0: 24.37122391772349, 1: 11.874545343055747}
    #max_cost: cost when there are no uav : 

    #Ant run 
    global_pheromones = init_pheromones(t_route)
    ant = Ant(sorted_routes, specific_routes) 
    #test bo anh manh 
    # ant.fitness() 
    # sys.exit()
   
    x = list()
    y = list()   
    
    for iter in range(MAX_ITERATION):
        iterO = INFINITY 
        pheromones = copy.deepcopy(global_pheromones)
        
        best_u_tour = None
        best_route_details = None
        for antIter in range(NUM_LANTS):   
            #find route and fitness of uav 
        
            uav_tour, search_space, route_details, cost = ant.find_route(pheromones)
           
            #local pheromone update 
            pheromones = ant.local_pheromone_update(uav_tour, pheromones)
        
            if cost < iterO:
                iterO = cost
                best_u_tour = uav_tour
                best_route_details = route_details

            #TODO: local search 
        
        #global 
        global_pheromones = global_pheromone_update(best_u_tour, global_pheromones, iterO)
        
        
        if iter % 5 == 0:
            x.append(iter) 
            y.append(iterO)
        
        # if iter % 100 == 0:
        #     print(f"iter {iter}")

        if iter == (MAX_ITERATION - 1):
            # plt.plot(x, y)
            # plt.show()

            #print fly time of uav 
            # for st in best_u_tour:
            #     path = best_u_tour[st]

            #     for i in range(len(path)-1):
            #         print(f"fly time from {path[i]} to {path[i+1]} is: {graph.d_time[path[i]][path[i+1]]}") 

            # print(best_su_tour)
            
            return iterO, best_route_details, best_u_tour

    return None

###########################################################
###########################################################
###########################################################    


class Ant():
    def __init__(self, search_space, specific_routes) -> None:
        self.search_space = search_space
        self.specific_routes = specific_routes


    def find_route(self,pheromones):
        routes = list(self.search_space.keys())

        search_space = copy.deepcopy(self.search_space)
        search_space[0] = 0
        
        
        C = list()

        for index in range(len(routes) -1):
            C.append(routes[index])
            C.append(0)
        
        C.append(routes[-1])
     
        #[5, 0, 4, 0, 3, 0, 2, 0, 6, 0, 1]
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

        #[4, 0, 6, 0, 3, 0, 5, 0, 2, 0, 1]
        #[4, 0, 6, 0, 3, 0, 0, 1]
       
        test_chosen_idx = [0, 2, 3, 10]

        cost = dict()
        chosen_idx = list()
        for i in range(0, len(C)):
            src = uav_tour[k][-1] #get last elemetn of uav tour k
            #for debug
            next_node = C[i] 
            t_time = search_space[next_node]

            if C[i] == src: #if 2 node 0 in a row 
                continue

            if random.random() < (1 /(1+ np.exp(- pheromones[(src, C[i])]))): 
            #if i in test_chosen_idx:
                #expected destination and expected travel time 
                e_des = C[i]
                e_travel_time = graph.d_time[src][e_des]

                if e_des == 0:
                    u_time += e_travel_time
                    endurance += e_travel_time
                else:
                    #constrain 1: T
                    if endurance + e_travel_time + graph.d_time[e_des][0] > T: 
                        #if uav cannot flyback to base when chosing the e_des as next node 
                        route_details['time_at_node'][next_node] = (search_space[next_node], -1)
                        continue
                    else:
                        route_details['time_at_node'][next_node] = (search_space[next_node], u_time + e_travel_time)
                    
                    e_uav_arrive_time = u_time + e_travel_time #expected uav arrival time 

                    if TECHNICAN_CAN_WAIT:
                        if e_uav_arrive_time > search_space[e_des]: #if uav come after technican
                            #TODO: make small func additional time that technican have to wait for uav at edes
                            for sid in self.specific_routes:
                                subtour = self.specific_routes[sid]

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
            sub_route_data = {
                        'k': k,
                        'route': uav_tour[k],
                        'endurance': endurance,
                    }
            route_details['uav_route'].append(sub_route_data)
            u_time += graph.d_time[uav_tour[k][-1]][0]

        if len(uav_tour[k]) == 1: #case when there are only start node 0
            del uav_tour[k]

        print(search_space[-1])
        if search_space[-1] > WORK_TIME:
            print("exceed work time")

        cost, wait_time = self.find_cost(time_at_nodes=route_details['time_at_node'], uav_tour=uav_tour, search_space=search_space)

        route_details['wait_times'] = wait_time
        return uav_tour, search_space, route_details, cost

    def find_cost(self, time_at_nodes, uav_tour, search_space):
        t_back_time = dict()
        
        specific_route = copy.deepcopy(self.specific_routes)
        wait_times = dict()
        
        for tid in self.specific_routes:
            
            path = self.specific_routes[tid]
            last_node = path[-1]
            
            time_leave_last_node = max(time_at_nodes[last_node][0],time_at_nodes[last_node][1] )
            t_back_time[tid] = time_leave_last_node + graph.t_time[last_node][0]

        
        #iterate over each subtour
        for stid in uav_tour:
            subtour = uav_tour[stid]
            
            #caculate time when uav back to base 
            last_sub_node_idx = len(subtour) - 2 #last node's index of uav subtour (except 0)
            back_time = max(time_at_nodes[subtour[last_sub_node_idx]][0],time_at_nodes[subtour[last_sub_node_idx]][1] )  + graph.d_time[subtour[last_sub_node_idx]][0] 


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

        #node that are brought back by  truck
        for tid in specific_route:
            for node in specific_route[tid]:
                wait_times[node] = ((t_back_time[tid] - max(time_at_nodes[node][0],time_at_nodes[node][1] )), 'truck')
        
        cost = 0
        for idx in wait_times:
            cost += wait_times[idx][0]
       
        return cost, wait_times

    def fitness(self, uav_tour,search_space, ispecific_routes, t_back_time, max_cost): 
 
        # uav_tour = {0: [0, 4, 0], 1: [0,3, 0], 2: [0, 6, 0], 3: [0, 1, 0], 4: [0, 5, 0]}
        # ispecific_routes= {0: [4, 3, 6, 1, 5, 2]}
        # t_back_time = {0: 30.096458996052014}
        # max_cost=  82.55549202245493

        specific_routes = copy.deepcopy(ispecific_routes)
        cost = max_cost 
    
        for stid in uav_tour:
            #iterate each uav subtour 
            subtour = uav_tour[stid]
            
            #caculate time when uav back to base 
            last_sub_node_idx = len(subtour) - 2 #last node's index of uav subtour (except 0)
            back_time = search_space[subtour[last_sub_node_idx]] + graph.d_time[subtour[last_sub_node_idx]][0] 

            #iterate each node in uav subtour
            for unid in range(1, len(subtour) -1):
                #iterate over each route of technican
                for tid in specific_routes:
                    index = -1
                    if subtour[unid] in specific_routes[tid]: #if uav take sample from technican tid
                        index = specific_routes[tid].index(subtour[unid]) #index of current uav node in technican specific route 

                        #reduce cost 
                        cost -= ((index +1) * (t_back_time[tid] - back_time))
                        # for finished_node in specific_routes[tid][0:(index+1)]:
                        #     cost -= t_back_time[tid] - back_time
                        
                        #delete finished node (from 0 to index)
                        del specific_routes[tid][0:(index+1)]

        #TODO: T

        #print(self.search_space)
        #print(uav_tour)
        #print(cost)
        return (1 - cost / max_cost)


    def local_pheromone_update(self, uav_tour, pheromones):
        l_pheromones = pheromones

        for stid in uav_tour:
            tour = uav_tour[stid]

            for i in range(1, len(tour)):   
                l_pheromones[(tour[i-1], tour[i])] -= LOCAL_EVAPORATION_RATE
        
        return l_pheromones

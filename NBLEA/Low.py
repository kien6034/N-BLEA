from operator import sub
from NBLEA.Map import *
from NBLEA.Parameter import *
import copy, random

import matplotlib.pyplot as plt
import numpy as np
import sys

graph = None
# graph = Map("Instances/6.5.1.txt")

def draw(routes):
    colors = ["green", "black", "red", "orange"]
        
    
    for tid in range(TECHNICAN_NUMS):
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
    routes = dict()

    for tid in range(TECHNICAN_NUMS):
        routes[tid] = list()

    k = 0
    for index in range(len(t_route)):        
        
        routes[k].append(t_route[index])
    
        k +=1
        if k == TECHNICAN_NUMS:
            k = 0
    return routes

def sort_by_time(specific_routes):
    routes = dict()
    t_back_time = dict()
    
    for tid in range(TECHNICAN_NUMS):
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
    for tid in range(TECHNICAN_NUMS):
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
            g_pheromones[(tour[i-1], tour[i])] +=  fitness * DEPOSIT_RATE
            
    
    for p in g_pheromones:
        g_pheromones[p] -= GLOBAL_EVAPORATION_RATE

    return g_pheromones


        ###########################################################
        ###########################################################
        ###########################################################
#solver 
def solver(sgraph, t_route):
    #update graph var
    global graph
    graph = sgraph
    
    specific_routes = get_specific_route(t_route)  #specific route: route corresponding to technicans {'0': path, '1': path} 
    #draw(specific_routes)


    sorted_routes, t_back_time, max_cost = sort_by_time(specific_routes)

    #Ant run 
    global_pheromones = init_pheromones(t_route)
    
    ant = Ant(sorted_routes)
   
    x = list()
    y = list()   
    
    for iter in range(MAX_ITERATION):
        iterO = INFINITY * (-1)
    
        pheromones = copy.deepcopy(global_pheromones)
        
        best_u_tour = None
        for antIter in range(NUM_LANTS):
            
            uav_tour = ant.find_route(pheromones)
          
            fitness = ant.fitness(uav_tour, specific_routes, t_back_time, max_cost)

            #local pheromone update 
            pheromones = ant.local_pheromone_update(uav_tour, pheromones)

            
            if fitness > iterO:

                iterO = fitness
                best_u_tour = uav_tour
            
  
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
            for st in best_u_tour:
                path = best_u_tour[st]

                # for i in range(len(path)-1):
                #     print(f"fly time from {path[i]} to {path[i+1]} is: {graph.d_time[path[i]][path[i+1]]}") 

            return iterO, max_cost, best_u_tour


        ###########################################################
        ###########################################################
        ###########################################################
        
    return None


class Ant():
    def __init__(self, search_space) -> None:
        self.search_space = search_space


    def find_route(self,pheromones):
        uav_tour = dict()

        routes = list(self.search_space.keys())

        search_space = copy.deepcopy(self.search_space)
        search_space[0] = 0
        
        C = list()

        for index in range(len(routes) -1):
            C.append(routes[index])
            C.append(0)
        
        C.append(routes[-1])

        #[6, 0, 5, 0, 1, 0, 4, 0, 3, 0, 2]
        k = 0 # so luong hanh trinh cua uav 
        uav_tour[k] = list()
        uav_tour[k].append(0)

        u_time = 0
        endurance = 0

        for i in range(0, len(C)):
            src = uav_tour[k][-1] #get last elemetn of uav tour 

            if C[i] == src: #if 2 node 0 in a row 
                continue
        
            if random.random() < (1 /(1+ np.exp(- pheromones[(src, C[i])]))): 
                e_des = C[i]
                e_travel_time = graph.d_time[src][e_des]

                #constrain 1: uav must arrive before technican
                if (u_time + e_travel_time) > search_space[e_des] and e_des != 0: #neu uav den sau drone  -> khong chon 
                    continue
                else:
                    if e_des == 0:
                        u_time += e_travel_time
                    else:
                        u_time = search_space[e_des]

                #constrain 2: T, uav energy level 
                uav_tour[k].append(e_des)
                
                #check if close subtour 
                if uav_tour[k][-1] == 0:
                    k +=1 
                    uav_tour[k] = list()
                    uav_tour[k].append(0)
        
        if uav_tour[k][-1] != 0:
            uav_tour[k].append(0)

        if len(uav_tour[k]) == 1: #case when there are only start node 0
            del uav_tour[k]
       

        return uav_tour

    def fitness(self, uav_tour, ispecific_routes, t_back_time, max_cost): 

       

        specific_routes = copy.deepcopy(ispecific_routes)

        cost = max_cost 

    
        for stid in uav_tour:
            #iterate each uav subtour 
            subtour = uav_tour[stid]
            
            last_sub_node_idx = len(subtour) - 2 #last node's index of uav subtour
    
            #caculate time when uav back to base 
         
            back_time = self.search_space[subtour[last_sub_node_idx]] + graph.d_time[subtour[last_sub_node_idx]][0] 

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

        #print(self.search_space)
        #print(uav_tour)
        

        return (1 - cost / max_cost)


    def local_pheromone_update(self, uav_tour, pheromones):
        l_pheromones = pheromones

        for stid in uav_tour:
            tour = uav_tour[stid]

            for i in range(1, len(tour)):   
                l_pheromones[(tour[i-1], tour[i])] -= LOCAL_EVAPORATION_RATE
        
        return l_pheromones

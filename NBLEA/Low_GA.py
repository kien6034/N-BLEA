from NBLEA.Map import *
from NBLEA.Parameter import *
import copy, random
import operator



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

def sort_by_time(graph, specific_routes):
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

    def create_individual(self):
        idv = []
        # for _ in range(len(self.t_route)):
        for _ in range(2*len(self.t_route)-1):
            idv += [random.randint(0, 1)]
        return idv

    def init_pop(self, popSize): 
        populations = []
       
        while len(populations) < popSize:
            new_idv = self.create_individual()
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
                if t_fly > T_MAX:
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

    def run(self, popSize = 50, eliteSize = 2, mutationRate = 0.01, generations = 50):
        specific_routes = get_specific_route(self.t_route)
        sorted_routes, t_back_time, max_cost = sort_by_time(self.graph, specific_routes)
        pop = self.init_pop(popSize)

        # print(specific_routes)
        # print(sorted_routes)
        # print(self.get_uav_route(pop[0], sorted_routes, specific_routes))
        # print(pop[0])

        for i in range(generations):

            pop_fitness = []
            for idv in pop:
                uav_tour = self.get_uav_route(idv, sorted_routes, specific_routes)
                cost = self.fitness(sorted_routes, uav_tour, specific_routes, t_back_time, max_cost)
                pop_fitness += [cost]

            # print(pop)
            best_idx = self.rank_pop(pop, pop_fitness)[0][0]
            best_cost = pop_fitness[best_idx]
            best_tour = self.get_uav_route(pop[best_idx], sorted_routes, specific_routes)
            print('best cost:', best_cost)
            print('uav_tour:', best_tour)
            
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

        return best_cost, best_tour

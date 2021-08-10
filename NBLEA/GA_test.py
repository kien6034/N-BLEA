from NBLEA.Map import *
from NBLEA.Parameter import *
from NBLEA.Low import *
from NBLEA import Low_GA
import random
import operator
import copy


class GA:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.search_space = graph.nodes


    def create_individual(self, searchSpace):
        ids = list(searchSpace.keys())
        ids.extend(list(range(len(ids) + 1, len(ids) + TECHNICAN_NUMS)))
        random.shuffle(ids)
        return ids
    
    def init_Pop(self, popSize, searchSpace):
        populations = []
       
        for i in range(0, popSize):
            populations.append(self.create_individual(searchSpace))

        return populations        

    def rank_pop(self, pop, low_fitness):
        fitness_results = {}
        
        for i in range(0, len(pop)):
            fitness_results[i] = low_fitness[i]

        return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = False)


    def idv_fitness(self, idv):
        #calculate travel distance 
        cost =0 
        for i in range(0, len(idv) - 1):
            cost += self.graph.t_time[idv[i]][idv[i+1]]
        
        fitness = 1 / cost 
        return fitness

    def pop_fitness(self, low_fitness):
        #calculate average fitness of population
        fitness = 0
        for i in range(0, len(low_fitness)):
            fitness += low_fitness[i]

        return fitness / len(low_fitness)

    def selection(self, pop_ranked, pop, eliteSize):
        elite_pop = []
        for i in pop_ranked[:eliteSize]:
            elite_pop.append(pop[i[0]])
        return elite_pop

    def tournament_selection(self, pop, low_fitness):
        # Selecting randomly 4 individuals to select 2 parents by a binary tournament
        parentIds = set()
        while len(parentIds) < 4:
            parentIds |= {random.randint(0, len(pop) - 1)}
        parentIds = list(parentIds)
        # Selecting 2 parents with the binary tournament
        parent1 = pop[parentIds[0]] if low_fitness[parentIds[0]] < low_fitness[parentIds[1]] else pop[parentIds[1]]
        parent2 = pop[parentIds[2]] if low_fitness[parentIds[2]] < low_fitness[parentIds[3]] else pop[parentIds[3]]
        return parent1, parent2

    def mutate(self, c, prob):
        new_child = c
        if random.random() < prob:
            # i1 = random.randint(0, len(c) - 1)
            # i2 = random.randint(0, len(c) - 1)
            # new_child[i1], new_child[i2] = new_child[i2], new_child[i1]

            i1 = random.randint(0, len(c) - 1)
            i2 = random.randint(i1, len(c) - 1)
            c_mid = new_child[i1:i2]
            c_mid.reverse()
            new_child = new_child[:i1] + c_mid + new_child[i2:]

        return new_child

    def crossover(self, p1, p2):
        # implement partially mapped crossover (PMX)
        size = min(len(p1), len(p2))
        c1, c2 = ['x'] * size, ['x'] * size

        cutIdx = set()
        while len(cutIdx) < 2:
            cutIdx |= {random.randint(1, size - 1)}
        cutIdx = list(cutIdx)
        cutIdx1, cutIdx2 = cutIdx[0], cutIdx[1]
        cutIdx1, cutIdx2 = min(cutIdx1, cutIdx2), max(cutIdx1, cutIdx2)
        c1[cutIdx1:cutIdx2] = p2[cutIdx1:cutIdx2]
        c2[cutIdx1:cutIdx2] = p1[cutIdx1:cutIdx2]

        p1_dict={}
        p2_dict={}
        for i in range(size):
            if c1[i] == 'x' or c2[i] == 'x':
                if p1[i] not in c1:
                    c1[i] = p1[i]
                if p2[i] not in c2:
                    c2[i] = p2[i]

            p1_dict[p1[i]] = p2[i]
            p2_dict[p2[i]] = p1[i]

        for i in range(size):
            if c1[i] == 'x':
                new_gen = p1[i]
                while True:
                    new_gen = p2_dict[new_gen]
                    if new_gen not in c1:
                        c1[i] = new_gen
                        break
            if c2[i] == 'x':
                new_gen = p1[i]
                while True:
                    new_gen = p1_dict[new_gen]
                    if new_gen not in c2:
                        c2[i] = new_gen
                        break

        return c1, c2

    def run(self, popSize = POP_SIZE, eliteSize = ELITE_SIZE, mutationRate = MUTATION_RATE, generations = GENERATIONS):
        # init pop 
        pop = self.init_Pop(popSize, self.search_space)
        # print(pop)
        # print(pop[0])

        for i in range(0, generations):
            #chosing parent 
            #cross over with prob pc 
            # mutation with prob pc
            #decode and fitness calculation 
            #survivor selection
            #find best
            print("=====================================")
            print(f'generation {i+1}')

            low_fitness = []
            u_tours = []
            route_details = []
            for j in range(len(pop)):
                # print(f'calculating fitness {j+1}')
                t_route = copy.deepcopy(pop[j]) 
                
                # low ant
                # cost, route_detail, u_tour = solver(self.graph, t_route) 

                # low GA
                low_GA = Low_GA.GA(self.graph, t_route)
                cost, u_tour, route_detail = low_GA.run()

                ### technican fitness: tf ^ alpha * df ^ beta.

                low_fitness += [cost]
                u_tours += [u_tour]
                route_details += [route_detail]

            # print('average fitness:', self.pop_fitness(low_fitness))
            # pprint.pprint(route_detail)
            print('best fitness:', low_fitness[self.rank_pop(pop, low_fitness)[0][0]])
            print('t_tour', pop[self.rank_pop(pop, low_fitness)[0][0]])
            print('UAV tour:', u_tours[self.rank_pop(pop, low_fitness)[0][0]])

            next_pop = []
            while len(next_pop) < popSize - eliteSize:
                p1, p2 = self.tournament_selection(pop, low_fitness)
                c1, c2 = self.crossover(p1, p2)
                while c1 in pop and c2 in pop:
                    # print(c1, c2)
                    p1, p2 = self.tournament_selection(pop, low_fitness)
                    c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                next_pop += [c1, c2]
            pop_ranked = self.rank_pop(pop, low_fitness)
            elite_pop = self.selection(pop_ranked, pop, eliteSize)
            next_pop += elite_pop
            pop = next_pop
            
            
        
        #find fitness of pop 

        """
            Init pop 
            Find fitness ò pop

            While (termination critera is reached) do 
                parent selection 
                cross over with prob pc 
                mutation with prob pm 
                decode and fitness calculation 
                survivor selection 
                find best
            endwhile 

            return best
    """



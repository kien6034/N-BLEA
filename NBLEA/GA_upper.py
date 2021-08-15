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

    def rank_pop(self, pop):
        fitness_results = {}
        
        for i in range(0, len(pop)):
            fitness_results[i] = self.idv_fitness(pop[i])

        return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = False)


    def idv_fitness(self, idv):
        #calculate wait time 

        specific_route = Low_GA.get_specific_route(idv)
        sorted_route, t_back_time, max_cost = Low_GA.sort_by_time(self.graph, specific_route)

        return max_cost

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

    def tournament_selection(self, pop):
        # Selecting randomly 4 individuals to select 2 parents by a binary tournament
        parentIds = set()
        while len(parentIds) < 4:
            parentIds |= {random.randint(0, len(pop) - 1)}
        parentIds = list(parentIds)
        # Selecting 2 parents with the binary tournament
        parent1 = pop[parentIds[0]] if self.idv_fitness(pop[parentIds[0]]) < self.idv_fitness(pop[parentIds[1]]) else pop[parentIds[1]]
        parent2 = pop[parentIds[2]] if self.idv_fitness(pop[parentIds[2]]) < self.idv_fitness(pop[parentIds[3]]) else pop[parentIds[3]]
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


            next_pop = []
            while len(next_pop) < popSize - eliteSize:
                p1, p2 = self.tournament_selection(pop)
                c1, c2 = self.crossover(p1, p2)
                while c1 in pop and c2 in pop:
                    # print(c1, c2)
                    p1, p2 = self.tournament_selection(pop)
                    c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                next_pop += [c1, c2]
            pop_ranked = self.rank_pop(pop)
            elite_pop = self.selection(pop_ranked, pop, eliteSize)
            next_pop += elite_pop
            pop = next_pop
            
            print("=====================================")
            print(f'generation {i+1}')

            t_route = pop[self.rank_pop(pop)[0][0]]
            best_fitness = self.rank_pop(pop)[0][1]

            print(f'best t_route: {t_route}')
            print(f'best fitness: {best_fitness}')
            
        return t_route, best_fitness
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


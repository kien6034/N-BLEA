from NBLEA.Map import *
from NBLEA.Parameter import *
import random
import operator


class GA:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.search_space = graph.nodes


    def create_individual(self, searchSpace):
        
        ids = list(searchSpace.keys())
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
        
        return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = True)


    def idv_fitness(self, idv):
        #calculate travel distance 
        cost =0 
        for i in range(0, len(idv) - 1):
            cost += self.graph.t_time[idv[i]][idv[i+1]]
        
        fitness = 1 / cost 
        return fitness


    def selection(self, pop_ranked, eliteSize):
        pass

    def run(self, popSize = POP_SIZE, eliteSize = ELITE_SIZE, mutationRate = MUTATION_RATE, generations =GENERATIONS):
        # init pop 
        pop = self.init_Pop(popSize, self.search_space)

        print(pop[-1])
        #Calculate fitness 
        pop_ranked = self.rank_pop(pop)

        

        for i in range(0, generations):
            #chosing parent 
            self.selection(pop_ranked, eliteSize)
            #cross over with prob pc 

            # mutation with prob pc

            #decode and fitness calculation 

            #survivor selection

            #find best
            pass
        
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



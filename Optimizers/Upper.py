import random
from Optimizers.Parameter import *
from Optimizers.Lower import Lower
from Optimizers.Utils import hamming_distance
import numpy as np
import sys
import operator
from os import path, mkdir
import time
# import distance
import pprint


class Upper:
    def __init__(self, graph) -> None:
        self.graph = graph

    def init_Pop(self, popSize, technican_num, create_sample):
        pop = np.zeros((popSize, (self.graph.numNodes + technican_num - 2)), dtype=np.int16)

        for i in range(popSize):
            gen = np.arange(1, (self.graph.numNodes + technican_num - 1))
            np.random.shuffle(gen)
            while gen.tolist() in pop.tolist():
                np.random.shuffle(gen)
            pop[i] = gen

        # if create_sample:
        #     expected_dir = f"low_data/{self.graph.fileName}"

        #     if not path.exists(expected_dir):
        #         mkdir(expected_dir)
        #     np.save(f"{expected_dir}/route", pop[0])
        #     np.save(f"{expected_dir}/techican_num", technican_num)


        return pop

    def mutate(self, c, prob):
        new_child = c[:]
        if random.random() < prob:
            # i1 = random.randint(0, len(c) - 1)
            # i2 = random.randint(0, len(c) - 1)
            # new_child[i1], new_child[i2] = new_child[i2], new_child[i1]

            cutIdx = set()
            while len(cutIdx) < 2:
                cutIdx |= {random.randint(0, len(c))}
            cutIdx = list(cutIdx)
            i1, i2 = min(cutIdx[0], cutIdx[1]), max(cutIdx[0], cutIdx[1])

            # reverse chromesomes i1 -> i2
            c_mid = new_child[i1:i2][::-1]
            new_child[i1:i2] = c_mid

        return new_child

    def crossover(self, p1, p2):
        # implement partially mapped crossover (PMX)
        size = min(len(p1), len(p2))
        c1, c2 = np.array([-1] * size, dtype=np.int16), np.array([-1] * size, dtype=np.int16)

        cutIdx = set()
        while len(cutIdx) < 2:
            cutIdx |= {random.randint(1, size - 1)}
        cutIdx = list(cutIdx)
        cutIdx1, cutIdx2 = cutIdx[0], cutIdx[1]
        cutIdx1, cutIdx2 = min(cutIdx1, cutIdx2), max(cutIdx1, cutIdx2)
        c1[cutIdx1:cutIdx2] = p2[cutIdx1:cutIdx2]
        c2[cutIdx1:cutIdx2] = p1[cutIdx1:cutIdx2]

        p1_dict = {}
        p2_dict = {}
        for i in range(size):
            if c1[i] == -1 or c2[i] == -1:
                if p1[i] not in c1:
                    c1[i] = p1[i]
                if p2[i] not in c2:
                    c2[i] = p2[i]

            p1_dict[p1[i]] = p2[i]
            p2_dict[p2[i]] = p1[i]

        for i in range(size):
            if c1[i] == -1:
                new_gen = p1[i]
                while True:
                    new_gen = p2_dict[new_gen]
                    if new_gen not in c1:
                        c1[i] = new_gen
                        break
            if c2[i] == -1:
                new_gen = p1[i]
                while True:
                    new_gen = p1_dict[new_gen]
                    if new_gen not in c2:
                        c2[i] = new_gen
                        break

        return c1, c2

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

    def fitness(self, idv, l_params, technican_num):
        #calculate wait time 
        lower_GA = Lower(self.graph, idv, l_params, technican_num)
        specific_route = lower_GA.get_specific_route(idv)
        sorted_route, t_back_time, max_cost = lower_GA.sort_by_time(specific_route)

        return max_cost

    def fitness1(self, idv, l_params, technican_num):
        #calculate wait time 
        lower_GA = Lower(self.graph, idv, l_params, technican_num)
        specific_route = lower_GA.get_specific_route(idv)
        sorted_route, t_back_time, max_cost = lower_GA.sort_by_time(specific_route)
        cost = 0
        for i in t_back_time:
            if t_back_time[i] > cost:
                cost = t_back_time[i]
        return cost

    def select_elite(self, pop_ranked, pop, eliteSize, technican_num):
        
        def is_diff(idv, chosen_pop, count):
            for i in range(count):
                # if distance.hamming(idv.tostring(), chosen_pop[i].tostring()) < len(idv) / 4:
                #     return False
                if hamming_distance(idv, chosen_pop[i], self.graph.numNodes) < len(idv) / 4:
                    return False
            return True

        elite_pop = np.zeros((eliteSize, (self.graph.numNodes + technican_num - 2)), dtype=np.int16)
        i = 0
        left_pop = []
        for idx in pop_ranked:
            if is_diff(pop[idx], elite_pop, i) or i == 0:
                elite_pop[i] = pop[idx]
                i+=1
            else:
                left_pop += [pop[idx]]
            if np.all(elite_pop[eliteSize-1] != 0):
                break
        if i < eliteSize:
            elite_pop[i:eliteSize] = random.sample(left_pop, eliteSize - i)
        # print(i)
        # print(elite_pop)
        return elite_pop
    
    def select_elite1(self, pop_ranked, pop, eliteSize):
        elite_pop = []
        for i in pop_ranked[:eliteSize]:
            elite_pop.append(pop[i])
        return np.array(elite_pop, dtype=np.int16)

    def rank_pop(self, pop, low_fitness):
        fitness_results = {}
        
        for i in range(0, len(low_fitness)):
            fitness_results[i] = low_fitness[i]

        pop_ranked = sorted(fitness_results.items(), key = operator.itemgetter(1), reverse = False)
        # print(pop)
        # print(pop_ranked)
        # sorted_pop = list(map(lambda x: pop[x[0]], pop_ranked))
        # pop = np.array(sorted_pop, dtype=np.int16)
        # print(len(sorted_pop))
        # print(pop)
        # sys.exit()
        # return np.array(list(map(lambda x: x[0], pop_ranked)), dtype=np.int16)
        return list(map(lambda x: x[0], pop_ranked))

    def get_diff_pop(self, pop, distance_rate, diff_size, cal_pop, cal_times):

        def is_diff(idv, chosen_pop, count):
            for i in range(count):
                # if distance.hamming(idv.tostring(), chosen_pop[i].tostring()) < len(idv) / 4:
                #     return False
                if hamming_distance(idv, pop[chosen_pop[i]], self.graph.numNodes) < len(idv) * distance_rate:
                    return False
            return True

        chosen_pop = []        
        offset = 0
        for i in range(len(pop)):
            if i ==0 or is_diff(pop[i], chosen_pop, len(chosen_pop)):
                chosen_pop += [i]
                idv = '-'.join(map(str, pop[i]))
                if idv in cal_pop and cal_pop[idv][3] >= cal_times:
                    offset +=1
            if len(chosen_pop) >= diff_size + offset:
                break
        return chosen_pop

    def run0(self, u_params, l_params, technican_num, save_stats=False):
        popSize= u_params['pop_size']
        generations = u_params['generations']
        eliteSize = u_params['elite_size']
        mutationRate =u_params['mutation_rate']
        create_sample = u_params['create_sample']
        # diff_size=u_params['diff_size']
        cal_times=u_params['cal_times']
        pop = self.init_Pop(popSize, technican_num, create_sample)

        cal_pop = {}
        record = {
            'convergence': [],
            'diff': []
        }
        optima_id = None
        start = time.time()
        for i in range(0, generations):
            print("=====================================")
            print(f'generation {i+1}')

            # low_fitness = np.zeros((popSize//2), dtype=np.float32)
            low_fitness = []
            u_tours = []
            route_details = []

            count = 0
            for j in range(popSize):
                t_route = pop[j][:]

                idv = '-'.join(map(str, pop[j]))
                if idv not in cal_pop:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.

                    cal_pop[idv] = (cost, u_tour, route_detail, 1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                elif idv in cal_pop and cal_pop[idv][3] < cal_times:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        preOptima=[cal_pop[idv][1]],
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.
                    cal_pop[idv] = (cost, u_tour, route_detail, cal_pop[idv][3]+1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                else:
                    low_fitness += [cal_pop[idv][0]]
                    u_tours += [cal_pop[idv][1]]
                    route_details += [cal_pop[idv][2]]
            
            print(len(low_fitness))
            print(count)
            # print(len(pop))
                    
            best_idv = pop[self.rank_pop(pop, low_fitness)[0]]
            best_id = '-'.join(map(str, best_idv))
            best_idv_detail = cal_pop[best_id]
            print('cost:', best_idv_detail[0])

            if optima_id != None:
                if best_idv_detail[0] < cal_pop[optima_id][0]:
                    optima_id = best_id
            else:
                optima_id = best_id

            if save_stats:
                record['convergence'] +=[best_idv_detail[0]]
                record['diff'].append(f'{len(low_fitness)}|{count}')

            pop_ranked = self.rank_pop(pop, low_fitness)
            # next_pop = self.select_elite(pop_ranked, pop, eliteSize, technican_num)
            next_pop = self.select_elite1(pop_ranked, pop, eliteSize)
            while len(next_pop) < popSize:
                p1, p2 = self.tournament_selection(pop, low_fitness)
                c1, c2 = self.crossover(p1, p2)
                while c1.tolist() in next_pop.tolist() and c2.tolist() in next_pop.tolist():
                    print(c1, c2)
                    p1, p2 = self.tournament_selection(pop, low_fitness)
                    c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                # next_pop += [c1, c2]
                next_pop = np.concatenate((next_pop, np.array([c1, c2], dtype=np.int16)))
            # elite_pop = self.selection(pop_ranked, pop, eliteSize)
            # next_pop += elite_pop
            pop = next_pop[:popSize]

        run_time = time.time() - start
        print(run_time)
        print("total idv:", len(cal_pop))

        if cal_pop[optima_id][2]['work_time'] > l_params['work_time']:
            cal_pop[optima_id] = (INFINITY, *cal_pop[optima_id][1:])

        return cal_pop[optima_id], run_time, record, optima_id

    def run1(self, u_params, l_params, technican_num, save_stats=False):
        popSize= u_params['pop_size']
        generations = u_params['generations']
        eliteSize = u_params['elite_size']
        mutationRate =u_params['mutation_rate']
        create_sample = u_params['create_sample']
        diff_size=u_params['diff_size']
        cal_times=u_params['cal_times']
        pop = self.init_Pop(popSize, technican_num, create_sample)

        cal_pop = {}
        record = {
            'convergence': [],
            'diff': []
        }
        optima_id = None
        start = time.time()
        for i in range(0, generations):
            print("=====================================")
            print(f'generation {i+1}')

            # low_fitness = np.zeros((popSize//2), dtype=np.float32)
            low_fitness = []
            u_tours = []
            route_details = []

            upper_fitness = []
            
            for j in range(len(pop)):
                # cal upper fitness
                upper_fitness += [self.fitness(pop[j], l_params, technican_num)]

            pop_ranked = self.rank_pop(pop, upper_fitness)
            pop = np.array(list(map(lambda x: pop[x], pop_ranked)), dtype=np.int16)
            upper_fitness = list(map(lambda x: upper_fitness[x], pop_ranked))
            # print(pop)
            count = 0
            for j in range(popSize):
                t_route = pop[j][:]

                idv = '-'.join(map(str, pop[j]))
                if idv not in cal_pop:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.

                    cal_pop[idv] = (cost, u_tour, route_detail, 1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                elif idv in cal_pop and cal_pop[idv][3] < cal_times:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        preOptima=[cal_pop[idv][1]],
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.
                    cal_pop[idv] = (cost, u_tour, route_detail, cal_pop[idv][3]+1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                else:
                    low_fitness += [cal_pop[idv][0]]
                    u_tours += [cal_pop[idv][1]]
                    route_details += [cal_pop[idv][2]]
                if count >= diff_size:
                    break
            
            print(len(low_fitness))
            print(count)
                    
            best_idv = pop[self.rank_pop(pop, low_fitness)[0]]
            best_id = '-'.join(map(str, best_idv))
            best_idv_detail = cal_pop[best_id]
            print('cost:', best_idv_detail[0])

            if optima_id != None:
                if best_idv_detail[0] < cal_pop[optima_id][0]:
                    optima_id = best_id
            else:
                optima_id = best_id

            if save_stats:
                record['convergence'] +=[best_idv_detail[0]]
                record['diff'].append(f'{len(low_fitness)}|{count}')

            pop_ranked = self.rank_pop(pop, low_fitness)
            # next_pop = self.select_elite(pop_ranked, pop, eliteSize, technican_num)
            next_pop = self.select_elite1(pop_ranked, pop, eliteSize)
            while len(next_pop) < popSize:
                p1, p2 = self.tournament_selection(pop, upper_fitness)
                c1, c2 = self.crossover(p1, p2)
                while c1.tolist() in next_pop.tolist() and c2.tolist() in next_pop.tolist():
                    print(c1, c2)
                    p1, p2 = self.tournament_selection(pop, upper_fitness)
                    c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                # next_pop += [c1, c2]
                next_pop = np.concatenate((next_pop, np.array([c1, c2], dtype=np.int16)))
            # elite_pop = self.selection(pop_ranked, pop, eliteSize)
            # next_pop += elite_pop
            pop = next_pop

        run_time = time.time() - start
        print(run_time)
        print("total idv:", len(cal_pop))

        if cal_pop[optima_id][2]['work_time'] > l_params['work_time']:
            cal_pop[optima_id] = (INFINITY, *cal_pop[optima_id][1:])
        
        return cal_pop[optima_id], run_time, record, optima_id

    def run2(self, u_params, l_params, technican_num, save_stats=False):
        popSize= u_params['pop_size']
        generations = u_params['generations']
        eliteSize = u_params['elite_size']
        mutationRate =u_params['mutation_rate']
        create_sample = u_params['create_sample']
        distance_rate=u_params['distance_rate']
        diff_size=u_params['diff_size']
        cal_times=u_params['cal_times']

        pop = self.init_Pop(popSize, technican_num, create_sample)

        cal_pop = {}
        record = {
            'convergence': [],
            'diff': []
        }
        start = time.time()
        for i in range(0, generations):
            print("=====================================")
            print(f'generation {i+1}')

            # low_fitness = np.zeros((popSize), dtype=np.float32)
            low_fitness = []
            u_tours = []
            route_details = []

            run_list = []
            # if i == 0:
            #     run_list = range(len(pop))
            # else:
            #     run_list = self.get_diff_pop(pop, distance_rate, diff_size)
            #     print(len(run_list))
            run_list = self.get_diff_pop(pop, distance_rate, diff_size, cal_pop, cal_times)
            print(len(run_list))
            
            count = 0
            s = time.time()
            for j in run_list:
                # print(f'calculating fitness {j+1}')
                
                t_route = pop[j][:]
                idv = '-'.join(map(str, pop[j]))
                if idv not in cal_pop:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.

                    cal_pop[idv] = (cost, u_tour, route_detail, 1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                elif idv in cal_pop and cal_pop[idv][3] < cal_times:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        preOptima=[cal_pop[idv][1]],
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.
                    cal_pop[idv] = (cost, u_tour, route_detail, cal_pop[idv][3]+1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                else:
                    low_fitness += [cal_pop[idv][0]]
                    u_tours += [cal_pop[idv][1]]
                    route_details += [cal_pop[idv][2]]

            print(count)
            print("lower time:", time.time()-s)

            if len(run_list) < len(pop):
            #     pop_ranked = self.rank_pop(pop, low_fitness)
            #     pop = np.array(list(map(lambda x: pop[x], pop_ranked)), dtype=np.int16)
            #     low_fitness = list(map(lambda x: low_fitness[x], pop_ranked))
            # else: 
                adjusted_pop = []
                for i in run_list:
                    adjusted_pop.append(pop[i])
                pop_ranked = self.rank_pop(pop, low_fitness)
                adjusted_pop = list(map(lambda x: adjusted_pop[x], pop_ranked))
                low_fitness = list(map(lambda x: low_fitness[x], pop_ranked))
                for i in range(len(pop)):
                    if i not in run_list:
                        adjusted_pop.append(pop[i])
                pop = np.array(adjusted_pop, dtype=np.int16)

            best_idv = pop[self.rank_pop(pop, low_fitness)[0]]
            best_id = '-'.join(map(str, best_idv))
            best_idv_detail = cal_pop[best_id]

            print('cost:', best_idv_detail[0])

            if save_stats:
                record['convergence'] +=[best_idv_detail[0]]
                record['diff'].append(f'{len(run_list)}|{count}')
            
            pop_ranked = self.rank_pop(pop, low_fitness)
            # next_pop = self.select_elite(pop_ranked, pop, eliteSize, technican_num)
            next_pop = self.select_elite1(pop_ranked, pop, eliteSize)
            while len(next_pop) < popSize:
                if len(next_pop) < popSize//2 + popSize//3:
                    p1, p2 = self.tournament_selection(pop[:popSize//2], low_fitness)
                    c1, c2 = self.crossover(p1, p2)
                    while c1.tolist() in next_pop.tolist() and c2.tolist() in next_pop.tolist():
                        # print(1, c1, c2)
                        p1, p2 = self.tournament_selection(pop[:popSize//2], low_fitness)
                        c1, c2 = self.crossover(p1, p2)
                    c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                elif len(next_pop) < popSize:
                    p_i1, p_i2 = random.randint(0, popSize//2), random.randint(popSize//2 + 1, popSize-1)
                    c1, c2 = self.crossover(pop[p_i1], pop[p_i2])
                    while c1.tolist() in next_pop.tolist() and c2.tolist() in next_pop.tolist():
                        # print(2, c1, c2)
                        p_i1, p_i2 = random.randint(0, popSize//2), random.randint(popSize//2 + 1, popSize-1)
                        c1, c2 = self.crossover(pop[p_i1], pop[p_i2])
                    c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                # else:
                #     p_i1, p_i2 = random.randint(popSize//2 + 1, popSize-1), random.randint(popSize//2 + 1, popSize-1)
                #     c1, c2 = self.crossover(pop[p_i1], pop[p_i2])
                #     while c1.tolist() in next_pop.tolist() and c2.tolist() in next_pop.tolist():
                #         print(3, c1, c2)
                #         p_i1, p_i2 = random.randint(0, popSize//2), random.randint(popSize//2 + 1, popSize-1)
                #         c1, c2 = self.crossover(pop[p_i1], pop[p_i2])
                #     c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                next_pop = np.concatenate((next_pop, np.array([c1, c2], dtype=np.int16)))
            pop = next_pop

        run_time = time.time() - start
        print(run_time)
        print(len(cal_pop))

        if cal_pop[optima_id][2]['work_time'] > l_params['work_time']:
            cal_pop[optima_id] = (INFINITY, *cal_pop[optima_id][1:])

        return best_idv_detail, run_time, record, best_id

    def run3(self, u_params, l_params, technican_num, save_stats=False):
        popSize= u_params['pop_size']
        generations = u_params['generations']
        eliteSize = u_params['elite_size']
        mutationRate =u_params['mutation_rate']
        create_sample = u_params['create_sample']
        distance_rate=u_params['distance_rate']
        diff_size=u_params['diff_size']
        cal_times=u_params['cal_times']
        
        pop = self.init_Pop(popSize, technican_num, create_sample)

        record = {
            'convergence': [],
            'diff': []
        }
        cal_pop = {}
        optima_id = None
        start = time.time()
        for i in range(0, generations):
            print("=====================================")
            print(f'generation {i+1}')

            # low_fitness = np.zeros((popSize//2), dtype=np.float32)
            low_fitness = []
            u_tours = []
            route_details = []

            upper_fitness = []

            
            for j in range(len(pop)):
                # cal upper fitness
                upper_fitness += [self.fitness(pop[j], l_params, technican_num)]

            pop_ranked = self.rank_pop(pop, upper_fitness)
            pop = np.array(list(map(lambda x: pop[x], pop_ranked)), dtype=np.int16)
            upper_fitness = list(map(lambda x: upper_fitness[x], pop_ranked))

            run_list = []
            run_list = self.get_diff_pop(pop, distance_rate, diff_size, cal_pop, cal_times)
            print(len(run_list))
            
            count = 0
            s = time.time()
            for j in run_list:
                t_route = pop[j][:]

                idv = '-'.join(map(str, pop[j]))
                if idv not in cal_pop:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.

                    cal_pop[idv] = (cost, u_tour, route_detail, 1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                elif idv in cal_pop and cal_pop[idv][3] < cal_times:
                    count += 1
                    # lower GA
                    lower_GA = Lower(self.graph, t_route, l_params, technican_num)
                    cost, u_tour, route_detail = lower_GA.run(l_params= l_params,
                                                        preOptima=[cal_pop[idv][1]],
                                                        check_dup=False)

                    ### technican fitness: tf ^ alpha * df ^ beta.
                    cal_pop[idv] = (cost, u_tour, route_detail, cal_pop[idv][3]+1)

                    low_fitness += [cost]
                    u_tours += [u_tour]
                    route_details += [route_detail]
                else:
                    low_fitness += [cal_pop[idv][0]]
                    u_tours += [cal_pop[idv][1]]
                    route_details += [cal_pop[idv][2]]
                    
            print(count)
            print("lower time:", time.time()-s)

            if len(run_list) < len(pop):
                pop_ranked = self.rank_pop(pop, low_fitness)
                low_fitness = list(map(lambda x: low_fitness[x], pop_ranked))
                adjusted_pop = []
                for i in run_list:
                    adjusted_pop.append(i)
                adjusted_pop = list(map(lambda x: adjusted_pop[x], pop_ranked))
                for i in range(len(pop)):
                    if i not in run_list:
                        adjusted_pop.append(i)
                pop = np.array(list(map(lambda x: pop[x], adjusted_pop)), dtype=np.int16)
                upper_fitness = list(map(lambda x: upper_fitness[x], adjusted_pop))

            best_idv = pop[self.rank_pop(pop, low_fitness)[0]]
            best_id = '-'.join(map(str, best_idv))
            best_idv_detail = cal_pop[best_id]
            print('cost:', best_idv_detail[0])

            if optima_id != None:
                if best_idv_detail[0] < cal_pop[optima_id][0]:
                    optima_id = best_id
            else:
                optima_id = best_id

            if save_stats:
                record['convergence'] +=[best_idv_detail[0]]
                record['diff'] += [f'{len(run_list)}|{count}']

            pop_ranked = self.rank_pop(pop, low_fitness)
            # next_pop = self.select_elite(pop_ranked, pop, eliteSize, technican_num)
            next_pop = self.select_elite1(pop_ranked, pop, eliteSize)
            while len(next_pop) < popSize:
                p1, p2 = self.tournament_selection(pop, upper_fitness)
                c1, c2 = self.crossover(p1, p2)
                while c1.tolist() in next_pop.tolist() and c2.tolist() in next_pop.tolist():
                    # print(c1, c2)
                    p1, p2 = self.tournament_selection(pop, upper_fitness)
                    c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1, mutationRate), self.mutate(c2, mutationRate)
                # next_pop += [c1, c2]
                next_pop = np.concatenate((next_pop, np.array([c1, c2], dtype=np.int16)))
            # elite_pop = self.selection(pop_ranked, pop, eliteSize)
            # next_pop += elite_pop
            pop = next_pop

        run_time = time.time() - start
        print("total idv:", len(cal_pop))
        print("run_time:", run_time)

        if cal_pop[optima_id][2]['work_time'] > l_params['work_time']:
            cal_pop[optima_id] = (INFINITY, *cal_pop[optima_id][1:])
            
        return cal_pop[optima_id], run_time, record, optima_id

        
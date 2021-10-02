import sys
import matplotlib.pyplot as plt

from Optimizers import Map as map
from Optimizers.Upper import Upper

def run(run_type, file_inputs, u_params, l_params, exports):
    input_dirs = []
    for num_node in file_inputs['num_nodes']:
        for r in file_inputs['range']:
            for file_num in file_inputs['file_num']:
                input_dir = file_inputs['input_folder'] + str(num_node) + '.' + str(r) + '.' + str(file_num) + '.txt'
                input_dirs.append(input_dir)
    
    
    #run
    for input_dir in input_dirs:
        for nrun in range(run_type['num_of_runs']):
            graph = map.Map(input_dir)
            
            if run_type['level'] == "upper":
                optimizer = Upper(graph)
                optimizer.run(popSize= u_params['pop_size'],
                            generations = u_params['generations'], 
                            eliteSize = u_params['elite_size'], 
                            mutationRate =u_params['mutation_rate'],
                            technican_num = run_type['technican_num'],
                            create_sample = u_params['create_sample'],
                            l_params=l_params),
            elif run_type['level'] == "lower":
                #read data
                #call low level 
                pass 

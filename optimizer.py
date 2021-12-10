import sys
import time
import matplotlib.pyplot as plt

from Optimizers import Map as map
from Optimizers.Upper import Upper
from Optimizers.Utils import save_solution, save_stats

def run(run_type, file_inputs, u_params, l_params, exports):
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
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
                if run_type['run_version'] == 1:
                    best_solution, run_time, record = optimizer.run(u_params=u_params,
                                                                l_params=l_params,
                                                                technican_num = run_type['technican_num'],
                                                                save_stats=(exports['save_stats'] or exports['diff_stats']))
                elif run_type['run_version'] == 2:
                    best_solution, run_time, record = optimizer.run1(u_params=u_params,
                                                                l_params=l_params,
                                                                technican_num = run_type['technican_num'],
                                                                save_stats=(exports['save_stats'] or exports['diff_stats']))
                elif run_type['run_version'] == 3:
                    best_solution, run_time, record = optimizer.run2(u_params=u_params,
                                                                l_params=l_params,
                                                                technican_num = run_type['technican_num'],
                                                                save_stats=(exports['save_stats'] or exports['diff_stats']))

                total_tech = str(best_solution[2]['number_of_tech']) + '/' + str(run_type['technican_num'])
                params = str(u_params['pop_size']) + '|' + str(u_params['cal_times']) + '|' + str(u_params['distance_rate'])
                if exports['to_excel']:
                    save_solution(instance=graph.fileName, 
                                run_time=run_time, 
                                number_of_tech=total_tech, 
                                cost=best_solution[0], 
                                work_time=best_solution[2]['work_time'], 
                                version=run_type['run_version'],
                                params=params,
                                current_time=current_time)

                if exports['save_stats']:
                    save_stats(instance=graph.fileName,
                            version=run_type['run_version'],
                            run_time=run_time,
                            tech_num=total_tech,
                            work_time=l_params['work_time'],
                            level='upper',
                            record=record['convergence'],
                            params=params,
                            current_time=current_time)
                if exports['diff_stats']:
                    save_stats(instance=graph.fileName,
                            version=run_type['run_version'],
                            run_time=run_time,
                            tech_num=total_tech,
                            work_time=l_params['work_time'],
                            level='diff',
                            record=record['diff'],
                            params=params,
                            current_time=current_time)
            elif run_type['level'] == "lower":
                #read data
                #call low level 
                pass 

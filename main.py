import optimizer

run_type = {
    'level': 'upper',
    'technican_num': 3,
    'num_of_runs': 10,
    'run_version': 2
}

file_inputs = {
    'input_folder': 'Instances/',
    # 'num_nodes': [6, 10, 12, 20],
    # 'range': [5, 10],
    # 'file_num': [3]
    'num_nodes': [6],
    'range': [5],
    'file_num': [3]
}

u_params = {
    'create_sample': 'txt',
    'generations': 30,
    'pop_size': 60,
    'elite_size': 15,
    'mutation_rate':  0.1,
}

l_params = {
    'generations': 50,
    'pop_size': 50,
    'elite_size': 2,
    'mutation_rate': 0.2,
    'technican_can_wait': True,
    'work_time': 120,
    'drone_time': 30,
}

exports= {
    'to_excel': False
}

if __name__=="__main__":
    optimizer.run(run_type=run_type, file_inputs= file_inputs, u_params=u_params, l_params=l_params, exports= exports)
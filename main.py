import optimizer

run_type = {
    'level': 'upper',
    'technican_num': 3,
    'num_of_runs': 1
}

file_inputs = {
    'input_folder': 'Instances/',
    'num_nodes': [6, 10, 12, 20],
    'range': [5, 10, 20],
    'file_num': [1, 2, 3, 4]
}

u_params = {
    'create_sample': 'txt',
    'generations': 50,
    'pop_size': 50,
    'elite_size': 2,
    'mutation_rate':  0.1,
}

l_params = {

}

exports= {

}

optimizer.run(run_type=run_type, file_inputs= file_inputs, u_params=u_params, l_params=l_params, exports= exports)
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import brian2 as b2
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import process_prior, process_simulator
import priors
from simulators import Simulator
import os
from datetime import datetime
import tables
import platform

# Define Hyperparameters of the inference
num_rounds = 100
num_simulations = 20
"""LOAD ALL AMORTIZE SAMPLES IN batch_directory"""
if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
elif platform.system() == 'Linux':
    results_dir = '/flash/FukaiU/danielmk/sbiemd/'

# Create the simulator and prepare for sbi
prior_dict = priors.baseline

sim = Simulator(0.1 * b2.ms,
                1 * b2.second,
                prior_dict['constants'],
                prior_dict)

prior_tensor = priors.prior_dict_to_tensor(prior_dict)

prior, theta_numel, prior_returns_numpy = process_prior(prior_tensor)

simulator = process_simulator(sim.run, prior, is_numpy_simulator=False)

# simulator, prior = prepare_for_sbi(sim.run, prior_tensor)

# Check the file path for already existing file under the same name
# if data already exists, load the inference instance and start at the last
# iteration. Otherwise start from scratch
output_filename = f'amortized_non-restricted_npe_network_{Simulator.name}_TESTING.pickle'

output_path = os.path.join(results_dir, output_filename)

for idx in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations, num_workers=4)

    c = datetime.now()

    with tables.open_file(output_path, mode='a') as output:
      output.create_group('/', str(c))
      output.create_array(f'/{c}', 'i', obj=idx)
      output.create_array(f'/{c}', 'x', obj=x.numpy())
      output.create_array(f'/{c}', 'theta', obj=theta.numpy())


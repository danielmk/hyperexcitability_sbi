# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import brian2 as b2
import torch
import sbi
from sbi.inference import SNPE
from sbi.utils.user_input_checks import process_prior, process_simulator
from sbi import utils as utils
import priors
from simulators_epilepsy import Simulator
import outcomes
import os
import pickle
from datetime import datetime
import tables
import numpy as np
from itertools import chain
import sys
from sbi import analysis as analysis
from sbi.inference import SNPE, simulate_for_sbi
import pdb
import platform

# Define Hyperparameters of the inference
num_rounds = 6
num_simulations = 20000
if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
    f = os.path.join(results_dir, "amortized_inference_with_nan_sbi-main.pickle")
elif platform.system() == 'Linux':
    results_dir = r'/flash/FukaiU/danielmk/sbiemd/'
    f = os.path.join(results_dir, "amortized_inference_with_nan_sbi-main.pickle")

# Load the armortized inference and use it to create the initial proposal
def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

"""IF THE OUTPUT FILE ALREADY EXIST, START INFERENCE FROM THE LAST SAVE"""
output_filename = f'truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_hyperexcitability_01.pickle'
output_path = os.path.join(results_dir, output_filename)

if os.path.isfile(output_path):
    # IF THE OUTPUT FILE EXISTS, START INFERENCE FROM ITS LAST SAVE
    inference = list(loadall(output_path))
    key = list(inference[-1].keys())[0]
    curr_idx = inference[-1][key]['idx'] + 1
    inference = inference[-1][key]['inference']
else:
    # IF NO OUTPUT FILE EXISTS YET, START FROM THE AMORTIZED INFERENCE
    inference = list(loadall(f))[0]
    curr_idx = 0

print(curr_idx)


# Create the simulator and prepare it for SBI
prior_dict = priors.baseline_epilepsy

sim = Simulator(0.1 * b2.ms,
                1 * b2.second,
                prior_dict['constants'],
                prior_dict)

prior_tensor = priors.prior_dict_to_tensor(prior_dict)

prior, theta_numel, prior_returns_numpy = process_prior(prior_tensor)

simulator = process_simulator(sim.run, prior, is_numpy_simulator=False)

# if not type(amortized_inference[-1]) == sbi.inference.snpe.snpe_c.SNPE_C:
#     amortized_inference = amortized_inference[-1][list(amortized_inference[-1].keys())[-1]]['inference']

# Define the outcome
outcome_name = 'hyperexcitable'

x_o = outcomes.x_hyperexcitable[outcome_name]

# Create posterior and truncate
# posterior = amortized_inference.build_posterior(amortized_estimator)

posterior = inference.build_posterior(sample_with='direct').set_default_x(x_o)

accept_reject_fn = utils.get_density_thresholder(posterior, quantile=1e-4)

proposal = utils.RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")

# Check the file path for already existing file under the same name
# if data already exists, load the inference instance and start at the last
# iteration. Otherwise start from scratch

# sys.exit()

output_path = os.path.join(results_dir, output_filename)

for idx in range(curr_idx, curr_idx + num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, num_workers=8)

    x = torch.nan_to_num(x, nan=0)
    
    _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
    
    posterior = inference.build_posterior(sample_with='direct').set_default_x(x_o)
    
    accept_reject_fn = utils.get_density_thresholder(posterior, quantile=1e-4)

    proposal = utils.RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
    
    c = datetime.now()

    data = {str(c):{
        'idx': idx,
        'theta': theta,
        'x': x,
        'inference': inference}
        }

    with open(output_path, 'ab') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

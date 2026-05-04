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
from simulators import Simulator
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
import io

# Define Hyperparameters of the inference
num_rounds = 4
num_simulations = 20000
if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
    estimator_path = os.path.join(results_dir, "amortized_inference_original_replicate_one_cpu.pickle")
elif platform.system() == 'Linux':
    results_dir = r'/flash/FukaiU/danielmk/sbiemd/'
    estimator_path = os.path.join(results_dir, "amortized_inference_original_replicate_one_cpu.pickle")

# Load the armortized inference and use it to create the initial proposal
def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

"""IF THE OUTPUT FILE ALREADY EXIST, START INFERENCE FROM THE LAST SAVE"""
output_filename = 'truncated_sequential_npe_network_baseline_conductance_based_01_baseline_replicate_02_nan_zero.pickle'
output_path = os.path.join(results_dir, output_filename)

# Check the file path for already existing file under the same name
# if data already exists, load the inference instance and start at the last
# iteration. Otherwise start from scratch
if os.path.isfile(output_path):
    # IF THE OUTPUT FILE EXISTS, START INFERENCE FROM ITS LAST SAVE
    inference = list(loadall(output_path))
    key = list(inference[-1].keys())[0]
    curr_idx = inference[-1][key]['idx'] + 1
    inference = inference[-1][key]['inference']
else:
    # IF NO OUTPUT FILE EXISTS YET, START FROM THE AMORTIZED INFERENCE
    print("STARTING FROM SCRATCH")
    with open(estimator_path, "rb") as f:
        inference = CPU_Unpickler(f).load()
    curr_idx = 0

print(curr_idx)

# Create the simulator and prepare it for SBI
prior_dict = priors.baseline

sim = Simulator(0.1 * b2.ms,
                1 * b2.second,
                prior_dict['constants'],
                prior_dict)

prior_tensor = priors.prior_dict_to_tensor(prior_dict)

prior, theta_numel, prior_returns_numpy = process_prior(prior_tensor)

simulator = process_simulator(sim.run, prior, is_numpy_simulator=False)

# Define the outcome
outcome_name = 'baseline'

x_o = outcomes.x_baseline[outcome_name]

posterior = inference.build_posterior(inference._neural_net.to("cpu"), sample_with='direct').set_default_x(x_o)

accept_reject_fn = utils.get_density_thresholder(posterior, quantile=1e-4)

proposal = utils.RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")

output_path = os.path.join(results_dir, output_filename)

for idx in range(curr_idx, curr_idx + num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, num_workers=16)

    x = torch.nan_to_num(x, nan=0)

    _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
    
    posterior = inference.build_posterior(inference._neural_net.to("cpu"), sample_with='direct').set_default_x(x_o)
    
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

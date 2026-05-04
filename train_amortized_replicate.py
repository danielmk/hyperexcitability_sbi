# -*- coding: utf-8 -*-
"""
Train an amortized estimator as in the ELife paper.
"""

import torch
import priors
import tables
import pickle
import numpy as np
from itertools import chain
from sbi.inference import NPE_C
import os
import platform
import sys
from sbi.neural_nets import posterior_nn
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp
from sbi.analysis.plot import sbc_rank_plot
import matplotlib.pyplot as plt
import pdb

"""LOAD ALL AMORTIZE SAMPLES IN batch_directory"""
if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
elif platform.system() == 'Linux':
    results_dir = '/flash/FukaiU/danielmk/sbips_sparsity/'

batch_directory = 'amortized_samples'

filepath = os.path.join(results_dir, batch_directory)

filenames = os.listdir(filepath)

files = [os.path.join(filepath, f) for f in filenames]  # Full paths to all files

all_theta = []
all_x = []
for file in files:
    f = tables.open_file(file, mode='r')
    runs = list(f.root._v_children)
    for k in runs:
        x = f.root[k].x.read()
        theta = f.root[k].theta.read()
        all_x.append(x)
        all_theta.append(theta)

all_x_flat = torch.Tensor(np.array(list(chain.from_iterable(all_x))))
all_theta_flat = torch.Tensor(np.array(list(chain.from_iterable(all_theta))))

all_x_flat = torch.nan_to_num(all_x_flat, nan=0.0)

pdb.set_trace()

# all_x_flat = torch.nan_to_num(all_x_flat, nan=0)  # Convert NaNs to zero
n_validation = 10000

n_simulations = 1122586  # As in eLife paper

all_x_flat = all_x_flat[:n_simulations + n_validation, :]

all_theta_flat = all_theta_flat[:n_simulations + n_validation, :]


invalid_simulations = torch.isnan(all_x_flat).any(axis=1)

all_x_flat = all_x_flat[~invalid_simulations]

all_theta_flat = all_theta_flat[~invalid_simulations]

torch.manual_seed(155)

shuffled_indices = torch.randperm(all_x_flat.shape[0])

all_x_flat = all_x_flat[shuffled_indices]

all_theta_flat = all_theta_flat[shuffled_indices]

training_x = all_x_flat[:-n_validation, :]

training_theta = all_theta_flat[:-n_validation, :]

training_x = training_x.to(device='cpu')

training_theta = training_theta.to(device='cpu')

validation_x = all_x_flat[-n_validation:,:]

validation_theta = all_theta_flat[-n_validation:,:]

validation_x = validation_x.to(device='cpu')

validation_theta = validation_theta.to(device='cpu')

prior_dict = priors.baseline

prior = priors.prior_dict_to_tensor(prior_dict)

inference = NPE_C(prior=prior, device='cpu')

inference = inference.append_simulations(training_theta, training_x)

print("Starting Training")

inference.train(training_batch_size=2000)

"""LOADING SAVING CODE!"""
with open(os.path.join(results_dir, 'amortized_inference_original_replicate_one_cpu.pickle'), 'wb') as f:
    pickle.dump(inference, f, pickle.HIGHEST_PROTOCOL)
    
""" 
file_path = os.path.join(results_dir, 'amortized_inference_06_80000_20_1e-4_nsf.pickle')

# Load the object
with open(file_path, 'rb') as f:
    inference = pickle.load(f)
"""

"""SBC"""
posterior = inference.build_posterior()

num_posterior_samples = 1000
num_workers = 1

ranks, dap_samples = run_sbc(
    validation_theta[-1000:, :], validation_x[-1000:, :], posterior, num_posterior_samples=num_posterior_samples, num_workers=10,
    reduce_fns=lambda theta, x: -posterior.log_prob(theta, x),
    # reduce_fns=posterior.log_prob
)

# theta_delete_nan = validation_theta[~torch.any(torch.isnan(validation_x), dim=1)]

check_stats = check_sbc(
    ranks, validation_theta[-1000:, :], dap_samples, num_posterior_samples=num_posterior_samples
)

plt.rcParams['svg.fonttype'] = 'none'

f, ax = sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=num_posterior_samples,
    plot_type="cdf",
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)

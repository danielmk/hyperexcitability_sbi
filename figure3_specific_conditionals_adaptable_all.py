# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import brian2 as b2
import torch
import sbi
import priors
from simulators import Simulator
import outcomes
import os
import pickle
import tables
import numpy as np
from sbi.utils.user_input_checks import prepare_for_sbi
from copy import deepcopy
from scipy.stats import ks_2samp
from metadata import Metadata
import conditionals

"""HYPERPARAMTERS"""
n_samples = 100000

np.random.seed(321)

torch.manual_seed(45234567)

f = os.path.join(Metadata.results_dir,
                 "truncated_sequential_npe_network_baseline_conductance_based_01_baseline.pickle")


def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


data = list(loadall(f))

x_o = outcomes.x_baseline['baseline']

inference_round = 3

inference = data[inference_round][list(data[inference_round].keys())[0]]['inference']

# posterior = inference.build_posterior().set_default_x(x_o)

prior_dict = priors.baseline

prior = priors.prior_dict_to_tensor(prior_dict)

simulator = Simulator(0.1 * b2.ms,
                      1.0 * b2.second,
                      prior_dict['constants'],
                      prior_dict)

simulator, prior = prepare_for_sbi(simulator.run, prior)

# posterior = inference.build_posterior().set_default_x(x_o)

# marginal_samples = posterior.sample((n_samples,), x=x_o)

posterior_estimator = inference._neural_net

conditional_normal = conditionals.synapse_only_normal_conditional

conditional_patho = conditionals.synapse_only_sprouting_conditional

samples = {"Normal": None, "Patho": None}

potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
    posterior_estimator, prior=prior, x_o=x_o
)

conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
    potential_fn=potential_fn,
    theta_transform=theta_transform,
    prior=prior,
    condition=torch.as_tensor(conditional_normal.condition, dtype=torch.float),
    dims_to_sample=conditional_normal.dims_to_sample,
)

mcmc_posterior = sbi.inference.MCMCPosterior(
    potential_fn=conditioned_potential_fn,
    theta_transform=restricted_tf,
    proposal=restricted_prior,
    method="slice_np_vectorized",
    num_chains=20,
).set_default_x(x_o)

samples['Normal'] = mcmc_posterior.sample((n_samples,))

conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
    potential_fn=potential_fn,
    theta_transform=theta_transform,
    prior=prior,
    condition=torch.as_tensor(conditional_patho.condition, dtype=torch.float),
    dims_to_sample=conditional_patho.dims_to_sample,
)

mcmc_posterior = sbi.inference.MCMCPosterior(
    potential_fn=conditioned_potential_fn,
    theta_transform=restricted_tf,
    proposal=restricted_prior,
    method="slice_np_vectorized",
    num_chains=20,
).set_default_x(x_o)

samples['Patho'] = mcmc_posterior.sample((n_samples,))

ks_test = []

for idx in range(len(conditional_normal.unconditioned_labels)):
    curr_ks = ks_2samp(samples['Normal'][:, idx], samples['Patho'][:, idx])
    ks_test.append(curr_ks)

ks_test = np.array(ks_test)

prior = deepcopy(priors.baseline)

n_simulations = 100

theta_baseline = conditional_normal.make_theta(samples['Normal'])[:n_simulations]
x_baseline = simulator(theta_baseline)

theta_hyperexcitable = conditional_patho.make_theta(samples['Patho'])[:n_simulations]
x_hyperexcitable = simulator(theta_hyperexcitable)

output_path = os.path.join(Metadata.results_dir, 'conditionals_output_data.h5')

condition = 'sprouting_only_v4'
with tables.open_file(output_path, mode='a') as output:
    output.create_group('/', f'{condition}')
    output.create_array(f'/{condition}', 'x_healthy', obj=x_baseline.numpy())
    output.create_array(f'/{condition}', 'x_sprouted', obj=x_hyperexcitable.numpy())
    output.create_array(f'/{condition}', 'theta_healthy', obj=samples['Normal'].numpy())
    output.create_array(f'/{condition}', 'theta_sprouted', obj=samples['Patho'].numpy())
    output.create_array(f'/{condition}', 'ks_test', obj=ks_test)

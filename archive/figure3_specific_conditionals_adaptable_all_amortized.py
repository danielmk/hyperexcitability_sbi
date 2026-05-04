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
import io
from sbi.analysis import conditional_potential

"""HYPERPARAMTERS"""
n_samples = 100000

np.random.seed(321)

torch.manual_seed(45234567)

results_dir = os.path.join(Metadata.results_dir,
                 "amortized_inference_one.pickle")



x_o = outcomes.x_baseline['baseline'].to('cuda')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


with open(results_dir, "rb") as f:
    # inference = CPU_Unpickler(f).load()
    inference = pickle.load(f)

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

conditional_normal = conditionals.intrinsics_normal_conditional

conditional_patho = conditionals.intrinsics_depolarized_conditional

samples = {"Normal": None, "Patho": None}

potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
    posterior_estimator, prior=prior, x_o=x_o
)

conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
    potential_fn=potential_fn,
    theta_transform=theta_transform,
    prior=prior,
    condition=torch.as_tensor(conditional_normal.condition, dtype=torch.float).to('cuda'),
    dims_to_sample=torch.Tensor(conditional_normal.dims_to_sample).to('cuda'),
)

mcmc_posterior = sbi.inference.MCMCPosterior(
    potential_fn=conditioned_potential_fn,
    theta_transform=restricted_tf,
    proposal=restricted_prior,
    method="hmc_pyro",
    num_chains=10,
    num_workers=12,
    device='cuda',
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
    num_chains=10,
    num_workers=12,
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

output_path = os.path.join(Metadata.results_dir, 'conditionals_output_data_amortized.h5')

condition = 'hyperexcitable_v4'
with tables.open_file(output_path, mode='a') as output:
    output.create_group('/', f'{condition}')
    output.create_array(f'/{condition}', 'x_healthy', obj=x_baseline.numpy())
    output.create_array(f'/{condition}', 'x_sprouted', obj=x_hyperexcitable.numpy())
    output.create_array(f'/{condition}', 'theta_healthy', obj=samples['Normal'].numpy())
    output.create_array(f'/{condition}', 'theta_sprouted', obj=samples['Patho'].numpy())
    output.create_array(f'/{condition}', 'ks_test', obj=ks_test)

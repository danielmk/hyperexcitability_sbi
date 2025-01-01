# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import brian2 as b2
import torch
import sbi
# from sbi.inference import SNPE, simulate_for_sbi
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
import matplotlib.pyplot as plt
import platform
from scipy.stats import spearmanr
from sbi.analysis import conditional_corrcoeff, conditional_pairplot, pairplot, conditional_potential
from sbi.utils.user_input_checks import prepare_for_sbi
from copy import deepcopy
from scipy.stats import ks_2samp
from metadata import EpilepsyMetadata
import conditionals

"""HYPERPARAMTERS"""
n_samples = 100000
# results_dir = '/flash/FukaiU/danielmk/sbiemd/truncated_sequential_npe_restricted_network_baseline_net_one_simulator_constant_dynamics_short_fewer_pc_cython_one.pickle'
# results_dir = r'truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_interictal_spiking_01.pickle'

np.random.seed(321)

torch.manual_seed(45234567)

if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
    f = os.path.join(results_dir, "truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_healthy_v2_01.pickle")
elif platform.system() == 'Linux':
    results_dir = r'/flash/FukaiU/danielmk/sbiemd/'
    f = os.path.join(results_dir, "truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_healthy_v2_01.pickle")

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

data = list(loadall(f))

x_o = outcomes.x_healthy_v2['x_healthy_v2']

inference_round = 3

inference = data[inference_round][list(data[inference_round].keys())[0]]['inference']

# posterior = inference.build_posterior().set_default_x(x_o)

prior_dict = priors.baseline_epilepsy

prior = priors.prior_dict_to_tensor(prior_dict)

simulator = Simulator(0.1 * b2.ms,
                1.0 * b2.second,
                prior_dict['constants'],
                prior_dict)

simulator, prior = prepare_for_sbi(simulator.run, prior)

# posterior = inference.build_posterior().set_default_x(x_o)

# marginal_samples = posterior.sample((n_samples,), x=x_o)

posterior_estimator = inference._neural_net

conditional_normal = synapse_only_normal_conditional

conditional_patho = synapse_only_sprouting_conditional 

samples = {"Normal": None, "Patho": None}

potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
    posterior_estimator, prior=prior, x_o=x_o
)

conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
    potential_fn=potential_fn,
    theta_transform=theta_transform,
    prior=prior,
    condition=torch.as_tensor(
        conditional_normal.condition,dtype=torch.float),
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
    condition=torch.as_tensor(
        conditional_patho.condition,dtype=torch.float),
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
    curr_ks = ks_2samp(samples['Normal'][:,idx], samples['Patho'][:,idx])
    ks_test.append(curr_ks)

ks_test = np.array(ks_test)

prior = deepcopy(priors.baseline_epilepsy)

n_simulations = 100

theta_baseline = conditional_normal.make_theta(samples['Normal'])[:n_simulations]
x_baseline = simulator(theta_baseline)

theta_hyperexcitable = conditional_patho.make_theta(samples['Patho'])[:n_simulations]
x_hyperexcitable = simulator(theta_hyperexcitable)

output_path = os.path.join(results_dir, 'conditionals_output_data.h5')

condition='sprouting_only_v4'
with tables.open_file(output_path, mode='a') as output:
    output.create_group('/', f'{condition}')
    output.create_array(f'/{condition}', 'x_healthy', obj=x_baseline.numpy())
    output.create_array(f'/{condition}', 'x_sprouted', obj=x_hyperexcitable.numpy())
    output.create_array(f'/{condition}', 'theta_healthy', obj=samples['Normal'].numpy())
    output.create_array(f'/{condition}', 'theta_sprouted', obj=samples['Patho'].numpy())
    output.create_array(f'/{condition}', 'ks_test', obj=ks_test)

"""PLOT SCATTERPLOTS OF ALL SIGNIFICANT INTERACTIONS"""





"""
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, correlations.correlation[i, j],
                       ha="center", va="center", color="w")
"""

"""
mass_functions = []
for x in range(len(labels)):
    samples = posterior_samples[:, x]
    plt.figure()
    hist_out = plt.hist(samples, bins=100, density=False, weights=np.ones(len(samples)) / len(samples))
    plt.xlim((priors.baseline_epilepsy['low'][labels[x]], priors.baseline_epilepsy['high'][labels[x]]))
    plt.xlabel(labels[x])
    mass_functions.append(hist_out[0])
    """

# simulator = SimulatorConstantDynamicsShort()

# posterior_sampled_output = simulator.run_evaluation(posterior_samples[0])

"""Get theta, x for restriction estimator"""
"""
files = []
for f in filenames:
    results_path = os.path.join(results_dir, f)
    file = tables.open_file(results_path, mode='r')
    files.append(file)

restriction_theta = []
restriction_x = []
for f in files:
    runs = list(f.root._v_children)
    for k in runs:
        x = f.root[k].x.read()
        theta = f.root[k].theta.read()
        restriction_x.append(x)
        restriction_theta.append(theta)

restriction_x_flat = torch.Tensor(np.array(list(chain.from_iterable(restriction_x))))
restriction_theta_flat = torch.Tensor(np.array(list(chain.from_iterable(restriction_theta))))

prior = priors.dynamics_constant_prior

sim = SimulatorConstantDynamicsShort()
        
simulator, prior = sbi.inference.prepare_for_sbi(sim.run, prior)

restriction_estimator = sbi.utils.RestrictionEstimator(prior=prior)

restriction_estimator.append_simulations(restriction_theta_flat, restriction_x_flat)

classifier = restriction_estimator.train()

restricted_prior = restriction_estimator.restrict_prior()

# Create the output file if it does not exist yet
inference = SNPE(prior=prior)



num_rounds=32
num_simulations=10000

posteriors = []
proposal = restricted_prior

output_filename = f'truncated_sequential_npe_restricted_network_{sim.network_type.version}_simulator_{sim.version}.pickle'

output_path = os.path.join(results_dir, output_filename)

if not os.path.isfile(output_path):
    with open(output_path, 'wb') as f:
        pass

for idx in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, num_workers=64)
    _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
    posterior = inference.build_posterior().set_default_x(x_o)
    accept_reject_fn = utils.get_density_thresholder(posterior, quantile=1e-4)
    proposal = utils.RestrictedPrior(restricted_prior, accept_reject_fn, sample_with="rejection")
    
    c = datetime.now()

    data = {str(c):{
        'theta': theta,
        'x': x,
        'inference': inference}
        }

    with open(output_path, 'ab') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

"""

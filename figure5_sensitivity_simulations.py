# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import brian2 as b2
import torch
import sbi
# from sbi.inference import SNPE, simulate_for_sbi
import priors
from simulators_epilepsy import Simulator
import outcomes
import os
import pickle
import tables
import numpy as np
import sys
import matplotlib.pyplot as plt
import platform
from sbi.utils.user_input_checks import prepare_for_sbi
from copy import deepcopy
from scipy.stats import ks_2samp
from sbi.analysis import pairplot, conditional_potential
from metadata import EpilepsyMetadata
import conditionals
import pdb

np.random.seed(321)

torch.manual_seed(45234567)


output_path = os.path.join(EpilepsyMetadata.results_dir, 'sensitivity_analysis.h5')

data_file = tables.open_file(output_path, mode='r')

prior_dict = priors.baseline_epilepsy

prior = priors.prior_dict_to_tensor(prior_dict)

simulator = Simulator(
    EpilepsyMetadata.sim_dt,
    EpilepsyMetadata.sim_duration,
    prior_dict['constants'],
    prior_dict)

simulator, prior = prepare_for_sbi(simulator.run, prior)

# sys.exit()

"""SET THE CONDITION HERE"""
# CONDITION IS SPECIFIED HERE!
conditional = conditionals.in_loss_conditional

all_thetas = data_file.root.in_loss.thetas.read()

output_collection = []
for curr_thetas in all_thetas:
    #theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations, num_workers=4)
    curr_input = conditional.make_theta(curr_thetas)
    curr_x = simulator(curr_input, num_workers=4)
    output_collection.append(curr_x)

output_path = os.path.join(EpilepsyMetadata.results_dir, 'conditionals_output_data_healthy_vs_hyperexcitable.h5')

condition='in_loss'
with tables.open_file(output_path, mode='a') as output:
    output.create_array(f'/{condition}', 'output_x', obj=np.array(output_collection))




"""PLOT SCATTERPLOTS OF ALL SIGNIFICANT INTERACTIONS"""


"""
fig, ax = plt.subplots(1, 5)
ax = ax.flatten()
for idx in np.arange(largest_parameters_map.shape[1]):
    # a.hist(marginal_samples_array[:,idx], bins=bins,color=colors[0], histtype=u'step')
    ax[idx].hist(samples['Baseline'][:,idx], bins=bins,color=colors[1], histtype=u'step')
    ax[idx].hist(samples['Hyperexcitable'][:,idx], bins=bins,color=colors[2], histtype=u'step')
    ax[idx].set_xlabel(largest_labels[idx])
    # ax[idx].set_xlim(limits[idx])

ax[0].legend(("Healthy", "IN Loss"))
"""


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

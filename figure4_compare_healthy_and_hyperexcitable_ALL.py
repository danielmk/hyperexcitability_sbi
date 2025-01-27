# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import torch
import sbi
# from sbi.inference import SNPE, simulate_for_sbi
import priors
from simulators import Simulator
import outcomes
import os
import pickle
import tables
import numpy as np
import matplotlib.pyplot as plt
from sbi.utils.user_input_checks import prepare_for_sbi
from scipy.stats import ks_2samp
from sbi.analysis import conditional_potential
from metadata import Metadata
import conditionals

np.random.seed(321)

torch.manual_seed(45234567)

f_bl = os.path.join(Metadata.results_dir,
                    "truncated_sequential_npe_network_baseline_conductance_based_01_baseline.pickle")
f_he = os.path.join(Metadata.results_dir,
                    "truncated_sequential_npe_network_baseline_conductance_based_01_hyperexcitable.pickle")


def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


data_bl = list(loadall(f_bl))
data_he = list(loadall(f_he))

inference_round = 3
inference_bl = data_bl[inference_round][list(data_bl[inference_round].keys())[0]]['inference']
inference_he = data_he[inference_round][list(data_he[inference_round].keys())[0]]['inference']

prior_dict = priors.baseline

prior = priors.prior_dict_to_tensor(prior_dict)

simulator = Simulator(
    Metadata.sim_dt,
    Metadata.sim_duration,
    prior_dict['constants'],
    prior_dict)

simulator, prior = prepare_for_sbi(simulator.run, prior)

"""SET THE CONDITION HERE"""
conditional = conditionals.synapse_only_sprouting_conditional  # CONDITION IS SPECIFIED HERE!

"""SAMPLE FROM THE TWO DIFFERENT NEURAL NETS WITH THE SAME CONDITION"""
posterior_estimators = {'Baseline': inference_bl._neural_net, 'Hyperexcitable': inference_he._neural_net}
outcomes = {'Baseline': outcomes.x_baseline['baseline'], 'Hyperexcitable': outcomes.x_hyperexcitable['hyperexcitable']}

samples = {"Baseline": None, "Hyperexcitable": None}

for key, value in posterior_estimators.items():
    posterior_estimator = value
    x_o = outcomes[key]

    potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
        posterior_estimator, prior=prior, x_o=x_o
    )

    conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        prior=prior,
        condition=torch.as_tensor(conditional.condition, dtype=torch.float),
        dims_to_sample=conditional.dims_to_sample,
    )

    mcmc_posterior = sbi.inference.MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=restricted_tf,
        proposal=restricted_prior,
        method="slice_np_vectorized",
        num_chains=20,
    ).set_default_x(x_o)

    samples[key] = mcmc_posterior.sample((conditional.n_samples,))

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 12})

"""PLOTTING"""
colors = ["#a6cee3", "#1f78b4", "#b2df8a"]


fig, ax = plt.subplots(6, 6)
bins = 100
ks_test = []
significant = []
ax = ax.flatten()
for idx in range(len(conditional.unconditioned_labels)):
    ax[idx].hist(samples['Baseline'][:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[idx].hist(samples['Hyperexcitable'][:, idx], bins=bins, color=colors[2], histtype=u'step')
    ax[idx].set_xlabel(conditional.unconditioned_labels[idx])
    curr_ks = ks_2samp(samples['Baseline'][:, idx], samples['Hyperexcitable'][:, idx])
    ks_test.append(curr_ks)

ax[0, 0].legend(("IN Loss Baseline", "IN Loss Hyperexcitable"))
# SAMPLING FROM MAP CONDITIONED

alpha = 0.001 / len(ks_test)

ks_test = np.array(ks_test)

fig, ax = plt.subplots(6, 5)

ax = ax.flatten()

significant_labels = np.array(conditional.unconditioned_labels)[(ks_test[:, 1] < alpha)]

significant_parameters_baseline = samples['Baseline'][:, (ks_test[:, 1] < alpha)]

significant_parameters_hyperexcitable = samples['Hyperexcitable'][:, (ks_test[:, 1] < alpha)]

for idx in np.arange(significant_parameters_baseline.shape[1]):
    # a.hist(marginal_samples_array[:,idx], bins=bins,color=colors[0], histtype=u'step')
    ax[idx].hist(significant_parameters_baseline[:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[idx].hist(significant_parameters_hyperexcitable[:, idx], bins=bins, color=colors[2], histtype=u'step')
    ax[idx].set_xlabel(significant_labels[idx])
    # ax[idx].set_xlim(limits[idx])

ax[0].legend(("IN Loss Baseline", "IN Loss Hyperexcitable"))

"""LARGEST STATISTIC"""
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})
largest_statistic = np.argsort(ks_test[:, 0])[::-1]
largest_statistic = largest_statistic[:5]

largest_labels = np.array(conditional.unconditioned_labels)[largest_statistic]

largest_parameters_map = np.array(samples['Baseline'])[:, largest_statistic]

largest_parameters_loss = np.array(samples['Hyperexcitable'])[:, largest_statistic]

# sys.exit()

n_simulations = 100
theta_baseline = conditional.make_theta(samples['Baseline'])[:n_simulations]
x_baseline = simulator(theta_baseline)

theta_hyperexcitable = conditional.make_theta(samples['Hyperexcitable'])[:n_simulations]
x_hyperexcitable = simulator(theta_hyperexcitable)

output_path = os.path.join(Metadata.results_dir, 'conditionals_output_data_healthy_vs_hyperexcitable.h5')

condition = 'sprouting_only_v3'
with tables.open_file(output_path, mode='a') as output:
    output.create_group('/', condition)
    output.create_array(f'/{condition}', 'x_healthy', obj=x_baseline.numpy())
    output.create_array(f'/{condition}', 'x_hyperexcitable', obj=x_hyperexcitable.numpy())
    output.create_array(f'/{condition}', 'theta_baseline', obj=samples['Baseline'].numpy())
    output.create_array(f'/{condition}', 'theta_hyperexcitable', obj=samples['Hyperexcitable'].numpy())
    output.create_array(f'/{condition}', 'ks_test', obj=ks_test)

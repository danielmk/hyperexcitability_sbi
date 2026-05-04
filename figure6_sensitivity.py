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

f_healthy = os.path.join(EpilepsyMetadata.results_dir, "truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_healthy_v2_01.pickle")


def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

data_healthy = list(loadall(f_healthy))

inference_round = 3
inference_baseline = data_healthy[inference_round][list(data_healthy[inference_round].keys())[0]]['inference']

prior_dict = priors.baseline_epilepsy

prior = priors.prior_dict_to_tensor(prior_dict)

simulator = Simulator(
    EpilepsyMetadata.sim_dt,
    EpilepsyMetadata.sim_duration,
    prior_dict['constants'],
    prior_dict)

simulator, prior = prepare_for_sbi(simulator.run, prior)

"""SET THE CONDITION HERE"""
conditional = conditionals.in_loss_conditional  # CONDITION IS SPECIFIED HERE!
conditional.n_samples=100

"""SAMPLE FROM THE TWO DIFFERENT NEURAL NETS WITH THE SAME CONDITION"""
# posterior_estimators = {'Baseline': inference_baseline._neural_net, 'Hyperexcitable': inference_hyperexcitable._neural_net}

posterior_estimator = inference_baseline._neural_net
x_o = outcomes.x_healthy_v2['x_healthy_v2']

potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
    posterior_estimator, prior=prior, x_o=x_o
)

conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
    potential_fn=potential_fn,
    theta_transform=theta_transform,
    prior=prior,
    condition=torch.as_tensor(
        conditional.condition,dtype=torch.float),
    dims_to_sample=conditional.dims_to_sample,
)

mcmc_posterior = sbi.inference.MCMCPosterior(
    potential_fn=conditioned_potential_fn,
    theta_transform=restricted_tf,
    proposal=restricted_prior,
    method="slice_np_vectorized",
    num_chains=20,
).set_default_x(x_o)

initial_samples = mcmc_posterior.sample((conditional.n_samples,))

thetas = conditional.make_theta(initial_samples)

parameter_samples = []
for free_parameter in conditional.unconditioned_labels:
    index_free = conditional.parameter_labels.index(free_parameter)
    for sample in thetas:
        curr_theta = dict(zip(range(sample.shape[0]), np.array(sample)))
        del curr_theta[index_free]

        curr_conditional = conditionals.Conditional("", curr_theta, EpilepsyMetadata.parameter_labels)
        curr_conditional.n_samples = 100
        
        potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
            posterior_estimator, prior=prior, x_o=x_o
        )

        conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            prior=prior,
            condition=torch.as_tensor(
                curr_conditional.condition,dtype=torch.float),
            dims_to_sample=conditional.dims_to_sample,
        )

        mcmc_posterior = sbi.inference.MCMCPosterior(
            potential_fn=conditioned_potential_fn,
            theta_transform=restricted_tf,
            proposal=restricted_prior,
            method="slice_np_vectorized",
            num_chains=20,
        ).set_default_x(x_o)
        
        curr_samples = mcmc_posterior.sample((conditional.n_samples,))
    parameter_samples.append(curr_samples)

sys.exit()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 12})

"""PLOTTING"""
colors = ["#a6cee3", "#1f78b4", "#b2df8a"]

fig, ax = plt.subplots(5, 6)
bins=100
ks_test = []
significant = []
for idx, a in enumerate(ax.flatten()):
    a.hist(samples['Baseline'][:,idx], bins=bins,color=colors[1], histtype=u'step')
    a.hist(samples['Hyperexcitable'][:,idx], bins=bins,color=colors[2], histtype=u'step')
    a.set_xlabel(conditional.parameter_labels[idx])
    curr_ks = ks_2samp(samples['Baseline'][:,idx], samples['Hyperexcitable'][:,idx])
    ks_test.append(curr_ks)

ax[0,0].legend(("IN Loss Baseline", "IN Loss Hyperexcitable"))
# SAMPLING FROM MAP CONDITIONED

alpha = 0.001 / len(ks_test)

ks_test = np.array(ks_test)

fig, ax = plt.subplots(6, 5)

ax = ax.flatten()

significant_labels = np.array(conditional.unconditioned_labels)[(ks_test[:, 1] < alpha)]

significant_parameters_baseline = samples['Baseline'][:,(ks_test[:, 1] < alpha)]

significant_parameters_hyperexcitable = samples['Hyperexcitable'][:,(ks_test[:, 1] < alpha)]

for idx in np.arange(significant_parameters_baseline.shape[1]):
    # a.hist(marginal_samples_array[:,idx], bins=bins,color=colors[0], histtype=u'step')
    ax[idx].hist(significant_parameters_baseline[:,idx], bins=bins,color=colors[1], histtype=u'step')
    ax[idx].hist(significant_parameters_hyperexcitable[:,idx], bins=bins,color=colors[2], histtype=u'step')
    ax[idx].set_xlabel(significant_labels[idx])
    # ax[idx].set_xlim(limits[idx])

ax[0].legend(("IN Loss Baseline", "IN Loss Hyperexcitable"))

"""LARGEST STATISTIC"""
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})
largest_statistic = np.argsort(ks_test[:, 0])[::-1]
largest_statistic = largest_statistic[:5]

largest_labels = np.array(conditional.unconditioned_labels)[largest_statistic]

largest_parameters_map = np.array(samples['Baseline'])[:,largest_statistic]

largest_parameters_loss = np.array(samples['Hyperexcitable'])[:,largest_statistic]

sys.exit()

n_simulations = 100
theta_baseline = conditional.make_theta(samples['Baseline'])[:n_simulations]
x_baseline = simulator(theta_baseline)

theta_hyperexcitable = conditional.make_theta(samples['Hyperexcitable'])[:n_simulations]
x_hyperexcitable = simulator(theta_baseline)

output_path = os.path.join(EpilepsyMetadata.results_dir, 'conditionals_output_data_healthy_vs_hyperexcitable.h5')

with tables.open_file(output_path, mode='a') as output:
    output.create_group('/', 'intrinsics')
    output.create_array(f'/intrinsics', 'x_healthy', obj=x_baseline.numpy())
    output.create_array(f'/intrinsics', 'x_hyperexcitable', obj=x_hyperexcitable.numpy())
    output.create_array(f'/intrinsics', 'theta_baseline', obj=samples['Baseline'].numpy())
    output.create_array(f'/intrinsics', 'theta_hyperexcitable', obj=samples['Hyperexcitable'].numpy())
    output.create_array(f'/intrinsics', 'ks_test', obj=ks_test)


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

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
import pandas as pd
import seaborn as sns
from metadata import EpilepsyMetadata
import conditionals

# results_dir = '/flash/FukaiU/danielmk/sbiemd/truncated_sequential_npe_restricted_network_baseline_net_one_simulator_constant_dynamics_short_fewer_pc_cython_one.pickle'
# results_dir = r'truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_interictal_spiking_01.pickle'

n_samples = 100000

np.random.seed(321)

torch.manual_seed(45234567)

labels = EpilepsyMetadata.parameter_labels

col_labels = EpilepsyMetadata.outcome_labels

if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
elif platform.system() == 'Linux':
    results_dir = r'/flash/FukaiU/danielmk/sbiemd/'

output_path = os.path.join(results_dir, 'conditionals_output_data_healthy_vs_hyperexcitable.h5')

data_file = tables.open_file(output_path, mode='r')

x_in_loss_healthy = data_file.root.in_loss.x_healthy.read()

theta_in_loss_healthy = data_file.root.in_loss.theta_baseline.read()

x_in_loss_in_loss = data_file.root.in_loss.x_hyperexcitable.read()

x_intrinsics_healthy = data_file.root.intrinsics.x_healthy.read()

x_intrinsics_hyperexcitable = data_file.root.intrinsics.x_hyperexcitable.read()

x_sprouting_healthy = data_file.root.sprouting_only_v3.x_healthy.read()

x_sprouting_hyperexcitable = data_file.root.sprouting_only_v3.x_hyperexcitable.read()

df_x_in_loss_healthy = pd.DataFrame(x_in_loss_healthy, columns=col_labels)
df_x_in_loss_healthy['conditional'] = 'IN Loss'
df_x_in_loss_healthy['condition'] = 'Baseline'

df_x_in_loss_in_loss = pd.DataFrame(x_in_loss_in_loss, columns=col_labels)
df_x_in_loss_in_loss['conditional'] = 'IN Loss'
df_x_in_loss_in_loss['condition'] = 'Hyperexcitable'

df_x_intrinsics_healthy = pd.DataFrame(x_intrinsics_healthy, columns=col_labels)
df_x_intrinsics_healthy['conditional'] = 'Intrinsics'
df_x_intrinsics_healthy['condition'] = 'Baseline'

df_x_intrinsics_hyperexcitable = pd.DataFrame(x_intrinsics_hyperexcitable, columns=col_labels)
df_x_intrinsics_hyperexcitable['conditional'] = 'Intrinsics'
df_x_intrinsics_hyperexcitable['condition'] = 'Hyperexcitable'

df_x_sprouting_healthy = pd.DataFrame(x_sprouting_healthy, columns=col_labels)
df_x_sprouting_healthy['conditional'] = 'Sprouting'
df_x_sprouting_healthy['condition'] = 'Baseline'

df_x_sprouting_hyperexcitable = pd.DataFrame(x_sprouting_hyperexcitable, columns=col_labels)
df_x_sprouting_hyperexcitable['conditional'] = 'Sprouting'
df_x_sprouting_hyperexcitable['condition'] = 'Hyperexcitable'

df = pd.concat([df_x_in_loss_healthy, df_x_in_loss_in_loss, df_x_intrinsics_healthy, df_x_intrinsics_hyperexcitable, df_x_sprouting_healthy, df_x_sprouting_hyperexcitable])

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(1, len(col_labels))

colors = ["#1f78b4", "#b2df8a"]

for idx, l in enumerate(col_labels):
    order = ['IN Loss', 'Intrinsics', 'Sprouting']
    sns.boxplot(x='conditional', y=l, hue='condition', order=order, data=df, palette=colors, ax=ax[idx])
    # ax[idx].plot(0, x_o_healthy[idx], marker='x', color='k', markersize='20')
    # ax[idx].plot(1, x_o_synchrony[idx], marker='x', color='k', markersize='20')
    # ax[idx].plot(2, x_o_spiking[idx], marker='x', color='k', markersize='20')
    ax[idx].tick_params(axis='x', rotation=90)

ks_test_in_loss = data_file.root.in_loss.ks_test.read()

ks_test_in_loss = np.insert(ks_test_in_loss, 4, [np.nan, np.nan], axis=0)
ks_test_in_loss = np.insert(ks_test_in_loss, 9, [np.nan, np.nan], axis=0)

ks_test_hyperexcitable = data_file.root.intrinsics.ks_test.read()

ks_test_hyperexcitable = np.insert(ks_test_hyperexcitable, 1, [np.nan, np.nan], axis=0)
ks_test_hyperexcitable = np.insert(ks_test_hyperexcitable, 2, [np.nan, np.nan], axis=0)

ks_test_sprouting = data_file.root.sprouting_only_v3.ks_test.read()

ks_test_sprouting = np.insert(ks_test_sprouting, 14, [np.nan, np.nan], axis=0)

colors = ['#66c2a5', '#fc8d62', '#8da0cb']
y = np.arange(ks_test_in_loss.shape[0])

fig, ax = plt.subplots(1)
y = np.arange(ks_test_in_loss.shape[0])
ls = 'solid'
alpha = 1
ax.hlines(np.arange(len(labels)), 0, np.nanmax([ks_test_sprouting[:,0], ks_test_hyperexcitable[:,0], ks_test_hyperexcitable[:,0]], axis=0), color='grey', alpha=alpha, linestyle=ls)
ax.plot(ks_test_in_loss[:,0], y, marker='o', linestyle='None', markersize=20, color=colors[0], alpha=0.8)
ax.plot(ks_test_sprouting[:,0],y, marker='d', linestyle='None', markersize=20, color=colors[1], alpha=0.8)
ax.plot(ks_test_hyperexcitable[:,0], y, marker='s', linestyle='None', markersize=20, color=colors[2], alpha=0.8)
ax.set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax.set_yticklabels(labels)
# plt.setp(ax[0].get_yticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor")
ax.set_xlabel("KS Test Statistic")
ax.legend(("", "IN Loss", "Sprouting", "Intrinsic"))
ax.set_xlim((0,1.1))


"""MAKE THE HISTOGRAM COMPARISON PLOT"""
colors = ["#1f78b4", "#b2df8a"]
ks_test_in_loss = data_file.root.in_loss.ks_test.read()
ks_test_hyperexcitable = data_file.root.intrinsics.ks_test.read()
ks_test_sprouting = np.array(data_file.root.sprouting_only_v3.ks_test.read())

theta_in_loss_healthy = data_file.root.in_loss.theta_baseline.read()

theta_in_loss_in_loss = data_file.root.in_loss.theta_hyperexcitable.read()

theta_hyperexcitable_healthy = data_file.root.intrinsics.theta_baseline.read()

theta_hyperexcitable_hyperexcitable = data_file.root.intrinsics.theta_hyperexcitable.read()

theta_sprouting_healthy = data_file.root.sprouting_only_v3.theta_baseline.read()

theta_sprouting_sprouting = data_file.root.sprouting_only_v3.theta_hyperexcitable.read()

in_conditional = conditionals.in_loss_conditional
in_loss_labels = in_conditional.unconditioned_labels

sprouting_conditional = conditionals.synapse_only_sprouting_conditional
sprouting_labels = sprouting_conditional.unconditioned_labels

intrinsics_conditional = conditionals.intrinsics_depolarized_conditional
excitable_labels = intrinsics_conditional.unconditioned_labels

largest_statistic_in_loss = np.argsort(ks_test_in_loss[:, 0])[::-1]
largest_statistic_in_loss = largest_statistic_in_loss[:5]

largest_labels_in_loss = np.array(in_loss_labels)[largest_statistic_in_loss]
largest_parameters_in_loss_healthy = np.array(theta_in_loss_healthy)[:,largest_statistic_in_loss]
largest_parameters_in_loss_unhealthy = np.array(theta_in_loss_in_loss)[:,largest_statistic_in_loss]

largest_statistic_sprouting = np.argsort(ks_test_sprouting[:, 0])[::-1]
largest_statistic_sprouting = largest_statistic_sprouting[:5]

largest_labels_sprouting = np.array(sprouting_labels)[largest_statistic_sprouting]
largest_parameters_sprouting_healthy = np.array(theta_sprouting_healthy)[:,largest_statistic_sprouting]
largest_parameters_sprouting_unhealthy = np.array(theta_sprouting_sprouting)[:,largest_statistic_sprouting]

largest_statistic_hyperexcitable = np.argsort(ks_test_hyperexcitable[:, 0])[::-1]
largest_statistic_hyperexcitable = largest_statistic_hyperexcitable[:5]

largest_labels_hyperexcitable = np.array(excitable_labels)[largest_statistic_hyperexcitable]
largest_parameters_hyperexcitable_healthy = np.array(theta_hyperexcitable_healthy)[:,largest_statistic_hyperexcitable]
largest_parameters_hyperexcitable_unhealthy = np.array(theta_hyperexcitable_hyperexcitable)[:,largest_statistic_hyperexcitable]

fig, ax = plt.subplots(3, 5)
bins = 100
for idx, param in enumerate(largest_statistic_in_loss):
    ax[0,idx].hist(largest_parameters_in_loss_healthy[:,idx], bins=bins, color=colors[0], histtype=u'step')
    ax[0,idx].hist(largest_parameters_in_loss_unhealthy[:,idx], bins=bins, color=colors[1], histtype=u'step')
    ax[0,idx].set_xlabel(in_loss_labels[param])
ax[0,0].legend(("Baseline", "Hyperexcitable"))

for idx, param in enumerate(largest_statistic_sprouting):
    ax[1,idx].hist(largest_parameters_sprouting_healthy[:,idx], bins=bins, color=colors[0], histtype=u'step')
    ax[1,idx].hist(largest_parameters_sprouting_unhealthy[:,idx], bins=bins, color=colors[1], histtype=u'step')
    ax[1,idx].set_xlabel(sprouting_labels[param])
ax[1,0].legend(("Baseline", "Hyperexcitable"))

for idx, param in enumerate(largest_statistic_hyperexcitable):
    ax[2,idx].hist(largest_parameters_hyperexcitable_healthy[:,idx], bins=bins, color=colors[0], histtype=u'step')
    ax[2,idx].hist(largest_parameters_hyperexcitable_unhealthy[:,idx], bins=bins, color=colors[1], histtype=u'step')
    ax[2,idx].set_xlabel(excitable_labels[param])
ax[2,0].legend(("Baseline", "Hyperexcitable"))

"""SCATTERPLOTS OF STATISTICS"""
baseline_vs_hyperexcitable_data_file = data_file

correlations_path = os.path.join(EpilepsyMetadata.results_dir, 'marginal_and_conditional_correlation_matrices.h5')

correlations_data_file = tables.open_file(correlations_path, mode='r')

healthy_vs_healthy_path = os.path.join(results_dir, 'conditionals_output_data.h5')

baseline_data_file = tables.open_file(healthy_vs_healthy_path , mode='r')

ks_test_in_loss_bl = data_file.root.in_loss.ks_test.read()

df_in_loss_bl = pd.DataFrame(ks_test_in_loss_bl, columns=['Statistic IN Loss BL', "p-value IN Loss"], index=conditionals.in_loss_conditional.unconditioned_labels)

ks_test_hyperexcitable_bl = data_file.root.intrinsics.ks_test.read()

df_hyperexctiabl_bl = pd.DataFrame(ks_test_hyperexcitable_bl, columns=['Statistic Hyperexctiable BL', "p-value Hyperexctiable"], index=conditionals.intrinsics_depolarized_conditional.unconditioned_labels)

ks_test_sprouting_bl = data_file.root.sprouting_only_v3.ks_test.read()

df_sprouting_bl = pd.DataFrame(ks_test_sprouting_bl, columns=['Statistic Sprouting BL', "p-value Sprouting"], index=conditionals.synapse_only_sprouting_conditional.unconditioned_labels)

df_baseline = pd.concat([df_in_loss_bl, df_hyperexctiabl_bl, df_sprouting_bl], axis=1, join='outer')

# ks_test_sprouting_bl = np.insert(ks_test_sprouting_bl, 14, [np.nan, np.nan], axis=0)

# CONSTRUCt THE IN LOSS DATA FRAME HERE
statistic_bl = baseline_data_file.root.in_loss.ks_test.read()[:,0]

statistic_bl_vs_hyp = baseline_vs_hyperexcitable_data_file.root.in_loss.ks_test.read()[:,0]

marginal_correlations = correlations_data_file.root.marginal_correlations.read()
abs_marginal_correlations = (np.abs(marginal_correlations[4]) + np.abs(marginal_correlations[9]))/2
abs_marginal_correlations = np.delete(abs_marginal_correlations, [4,9])

conditional_correlations = correlations_data_file.root.conditional_correlations.read()
abs_conditional_correlations = (np.abs(conditional_correlations[4]) + np.abs(conditional_correlations[9]))/2
abs_conditional_correlations = np.delete(abs_conditional_correlations, [4,9])

df_in_loss = df2 = pd.DataFrame(np.array([statistic_bl, statistic_bl_vs_hyp, abs_marginal_correlations, abs_conditional_correlations]).T,
                   columns=['KS Statistic BL', 'KS Statistic BL vs Hyp', 'Absolute Marginal Correlation', 'Absolute Conditional Correlation'])

df_in_loss['condition'] = "IN Loss"

#CONSTRUCT SPROUTING DF HERE
statistic_bl = baseline_data_file.root.sprouting_only_v4.ks_test.read()[:,0]

statistic_bl_vs_hyp = np.array(baseline_vs_hyperexcitable_data_file.root.sprouting_only_v3.ks_test.read())[:,0]

marginal_correlations = correlations_data_file.root.marginal_correlations.read()
abs_marginal_correlations = np.abs(marginal_correlations[14])
abs_marginal_correlations = np.delete(abs_marginal_correlations, [14])

conditional_correlations = correlations_data_file.root.conditional_correlations.read()
abs_conditional_correlations = np.abs(conditional_correlations[14])
abs_conditional_correlations = np.delete(abs_conditional_correlations, [14])

df_sprouting = df2 = pd.DataFrame(np.array([statistic_bl, statistic_bl_vs_hyp, abs_marginal_correlations, abs_conditional_correlations]).T,
                   columns=['KS Statistic BL', 'KS Statistic BL vs Hyp', 'Absolute Marginal Correlation', 'Absolute Conditional Correlation'])
df_sprouting['condition'] = "Sprouting"


#CONSTRUCT SPROUTING DF HERE
statistic_bl = baseline_data_file.root.hyperexcitable_v4.ks_test.read()[:,0]

statistic_bl_vs_hyp = np.array(baseline_vs_hyperexcitable_data_file.root.intrinsics.ks_test.read())[:,0]

marginal_correlations = correlations_data_file.root.marginal_correlations.read()
abs_marginal_correlations = (np.abs(marginal_correlations[1]) + np.abs(marginal_correlations[2]))/2
abs_marginal_correlations = np.delete(abs_marginal_correlations, [1, 2])

conditional_correlations = correlations_data_file.root.conditional_correlations.read()
abs_conditional_correlations = (np.abs(conditional_correlations[1]) + np.abs(conditional_correlations[2]))/2
abs_conditional_correlations = np.delete(abs_conditional_correlations, [1, 2])

df_intrinsics = df2 = pd.DataFrame(np.array([statistic_bl, statistic_bl_vs_hyp, abs_marginal_correlations, abs_conditional_correlations]).T,
                   columns=['KS Statistic BL', 'KS Statistic BL vs Hyp', 'Absolute Marginal Correlation', 'Absolute Conditional Correlation'])
df_intrinsics['condition'] = "Intrinsic"

df_full = pd.concat([df_in_loss, df_sprouting, df_intrinsics])

colors = ['#66c2a5', '#fc8d62', '#8da0cb']
sns.scatterplot(data=df_full, x='KS Statistic BL', y='Absolute Marginal Correlation', hue='condition',palette=colors)

sns.scatterplot(data=df_full, x='KS Statistic BL', y='Absolute Conditional Correlation', hue='condition',palette=colors)

sns.scatterplot(data=df_full, x='KS Statistic BL', y='KS Statistic BL vs Hyp', hue='condition',palette=colors)

sns.scatterplot(data=df_full, x='Absolute Marginal Correlation', y='Absolute Conditional Correlation', hue='condition',palette=colors)

"""
plt.figure()
plt.scatter(ks_test_in_loss[:,0],  ks_test_sprouting[:,0], c=ks_test_hyperexcitable[:,0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(ks_test_in_loss[:,0],  ks_test_hyperexcitable[:,0], ks_test_sprouting[:,0], c=ks_test_sprouting[:,0], s=200)

ax.set_xlabel('KS Stat IN Loss')
ax.set_ylabel('KS Stat Depolarized')
ax.set_zlabel('KS Stat Sprouting')
"""

"""
fig, ax = plt.subplots(1, 3)
y = np.arange(ks_test_in_loss.shape[0])
ax[0].hlines(np.arange(len(labels)), ks_test_in_loss[:,0], ks_test_sprouting[:,0], color='k')

ax[0].plot(ks_test_in_loss[:,0], y, marker='o', rasterized=True, linestyle='None', markersize=20, color=colors[0], alpha=0.8)
ax[0].plot(ks_test_sprouting[:,0],y, marker='d', rasterized=True, linestyle='None', markersize=20, color=colors[1], alpha=0.8)
ax[0].set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax[0].set_yticklabels(labels)
# plt.setp(ax[0].get_yticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor")
ax[0].set_xlabel("KS Test Statistic")
ax[0].legend(("", "IN Loss", "Sprouting"))
ax[0].set_xlim((0,1))

# for idx, l in enumerate(labels):

# fig, ax = plt.subplots(1)
ax[1].hlines(np.arange(len(labels)), ks_test_in_loss[:,0], ks_test_hyperexcitable[:,0], color='k')
ax[1].plot(ks_test_in_loss[:,0], y, marker='o', rasterized=True, linestyle='None', markersize=20, color=colors[0], alpha=0.8)
ax[1].plot(ks_test_hyperexcitable[:,0], y, marker='s', rasterized=True, linestyle='None', markersize=20, color=colors[2], alpha=0.8)
ax[1].set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax[1].set_yticklabels([])

ax[1].set_xlabel("KS Test Statistic")
ax[1].legend(("","IN Loss", "Hyperexcitable"))
ax[1].set_xlim((0,1))
"""


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

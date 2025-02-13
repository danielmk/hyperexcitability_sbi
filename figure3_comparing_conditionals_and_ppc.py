# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import torch
# from sbi.inference import SNPE, simulate_for_sbi
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
import matplotlib.pyplot as plt
import platform
from scipy.stats import spearmanr
from sbi.analysis import conditional_corrcoeff, conditional_pairplot, pairplot, conditional_potential
from sbi.utils.user_input_checks import prepare_for_sbi
from copy import deepcopy
from scipy.stats import ks_2samp
import pandas as pd
import seaborn as sns
from metadata import Metadata
import conditionals

n_samples = 100000

np.random.seed(321)

torch.manual_seed(45234567)

labels = Metadata.parameter_labels

col_labels = Metadata.outcome_labels

f = os.path.join(Metadata.results_dir,
                 "truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_healthy_v2_01.pickle")

output_path = os.path.join(Metadata.results_dir, 'conditionals_output_data.h5')

data_file = tables.open_file(output_path, mode='r')

x_in_loss_healthy = data_file.root.in_loss.x_healthy.read()

x_in_loss_in_loss = data_file.root.in_loss.x_in_loss.read()

x_hyperexcitable_healthy = data_file.root.hyperexcitable_v4.x_healthy.read()

x_hyperexcitable_hyperexcitable = data_file.root.hyperexcitable_v4.x_hyperexcitable.read()

x_sprouting_healthy = data_file.root.sprouting_only_v4.x_healthy.read()

x_sprouting_sprouting = data_file.root.sprouting_only_v4.x_sprouted.read()

df_x_in_loss_healthy = pd.DataFrame(x_in_loss_healthy, columns=col_labels)
df_x_in_loss_healthy['conditional'] = 'IN Loss'
df_x_in_loss_healthy['condition'] = 'Healthy'

df_x_in_loss_in_loss = pd.DataFrame(x_in_loss_in_loss, columns=col_labels)
df_x_in_loss_in_loss['conditional'] = 'IN Loss'
df_x_in_loss_in_loss['condition'] = 'Unhealthy'

df_x_hyperexcitable_healthy = pd.DataFrame(x_hyperexcitable_healthy, columns=col_labels)
df_x_hyperexcitable_healthy['conditional'] = 'Hyperexcitable'
df_x_hyperexcitable_healthy['condition'] = 'Healthy'

df_x_hyperexcitable_hyperexcitable = pd.DataFrame(x_hyperexcitable_hyperexcitable, columns=col_labels)
df_x_hyperexcitable_hyperexcitable['conditional'] = 'Hyperexcitable'
df_x_hyperexcitable_hyperexcitable['condition'] = 'Unhealthy'

df_x_sprouting_healthy = pd.DataFrame(x_sprouting_healthy, columns=col_labels)
df_x_sprouting_healthy['conditional'] = 'Sprouting'
df_x_sprouting_healthy['condition'] = 'Healthy'

df_x_sprouting_sprouting = pd.DataFrame(x_sprouting_sprouting, columns=col_labels)
df_x_sprouting_sprouting['conditional'] = 'Sprouting'
df_x_sprouting_sprouting['condition'] = 'Unhealthy'

df = pd.concat([df_x_in_loss_healthy,
                df_x_in_loss_in_loss,
                df_x_hyperexcitable_healthy,
                df_x_hyperexcitable_hyperexcitable,
                df_x_sprouting_healthy,
                df_x_sprouting_sprouting])

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(1, len(col_labels))

colors = ["#1f78b4", "#b2df8a"]

for idx, l in enumerate(col_labels):
    order = ['IN Loss', 'Hyperexcitable', 'Sprouting']
    sns.boxplot(x='conditional', y=l, hue='condition', order=order, data=df, palette=colors, ax=ax[idx])
    # ax[idx].plot(0, x_o_healthy[idx], marker='x', color='k', markersize='20')
    # ax[idx].plot(1, x_o_synchrony[idx], marker='x', color='k', markersize='20')
    # ax[idx].plot(2, x_o_spiking[idx], marker='x', color='k', markersize='20')
    ax[idx].tick_params(axis='x', rotation=90)

ks_test_in_loss = data_file.root.in_loss.ks_test.read()

ks_test_in_loss = np.insert(ks_test_in_loss, 4, [np.nan, np.nan], axis=0)
ks_test_in_loss = np.insert(ks_test_in_loss, 9, [np.nan, np.nan], axis=0)

ks_test_hyperexcitable = data_file.root.hyperexcitable_v4.ks_test.read()

ks_test_hyperexcitable = np.insert(ks_test_hyperexcitable, 1, [np.nan, np.nan], axis=0)
ks_test_hyperexcitable = np.insert(ks_test_hyperexcitable, 2, [np.nan, np.nan], axis=0)

ks_test_sprouting = data_file.root.sprouting_only_v4.ks_test.read()

ks_test_sprouting = np.insert(ks_test_sprouting, 14, [np.nan, np.nan], axis=0)


ks_test_all = data_file.root.all.ks_test.read()
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

correlations_path = os.path.join(Metadata.results_dir, 'marginal_and_conditional_correlation_matrices.h5')

correlation_file = tables.open_file(correlations_path, mode='r')

marginal_correlations = correlation_file.root.marginal_correlations.read()

conditional_correlations = correlation_file.root.conditional_correlations.read()

np.fill_diagonal(marginal_correlations, np.nan)

np.fill_diagonal(conditional_correlations, np.nan)

in_loss_corrs = (np.abs(conditional_correlations[4]) + np.abs(conditional_correlations[9]))/2

intrinsics_corrs = (np.abs(conditional_correlations[1]) + np.abs(conditional_correlations[2]))/2

sprouting_corrs = np.abs(conditional_correlations[14])

fig, ax = plt.subplots(1, 3)
y = np.arange(in_loss_corrs.shape[0])
ls = 'solid'
alpha = 1

ax[1].hlines(np.arange(len(labels)),
             0,
             np.nanmax([sprouting_corrs, intrinsics_corrs, in_loss_corrs], axis=0),
             color='grey',
             alpha=alpha,
             linestyle=ls)
ax[1].plot(in_loss_corrs, y, marker='o', linestyle='None', markersize=20, color=colors[0], alpha=0.8)
ax[1].plot(sprouting_corrs, y, marker='d', linestyle='None', markersize=20, color=colors[1], alpha=0.8)
ax[1].plot(intrinsics_corrs, y, marker='s', linestyle='None', markersize=20, color=colors[2], alpha=0.8)
ax[1].set_yticks(np.arange(len(labels)))
# [0]... and label them with the respective list entries
ax[1].set_yticklabels(labels)
# plt.setp(ax[0].get_yticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor")
ax[1].set_xlabel("Absolute Conditional Correlation Coefficient")
ax[1].legend(("", "IN Loss", "Sprouting", "Intrinsic"))
ax[1].set_xlim((0, 1.0))

y = np.arange(ks_test_in_loss.shape[0])
ax[0].hlines(np.arange(len(labels)),
             0,
             np.nanmax([ks_test_sprouting[:, 0], ks_test_hyperexcitable[:, 0], ks_test_in_loss[:, 0]], axis=0),
             color='grey',
             alpha=alpha,
             linestyle=ls)
ax[0].plot(ks_test_in_loss[:, 0], y, marker='o', linestyle='None', markersize=20, color=colors[0], alpha=0.8)
ax[0].plot(ks_test_sprouting[:, 0], y, marker='d', linestyle='None', markersize=20, color=colors[1], alpha=0.8)
ax[0].plot(ks_test_hyperexcitable[:, 0], y, marker='s', linestyle='None', markersize=20, color=colors[2], alpha=0.8)
ax[0].set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax[0].set_yticklabels(labels)
# plt.setp(ax[0].get_yticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor")
ax[0].set_xlabel("KS Test Statistic")
ax[0].legend(("", "IN Loss", "Sprouting", "Intrinsic"))
ax[0].set_xlim((0, 1))

conditional_correlations = correlation_file.root.marginal_correlations.read()

np.fill_diagonal(marginal_correlations, np.nan)

np.fill_diagonal(conditional_correlations, np.nan)

in_loss_corrs = (np.abs(conditional_correlations[4]) + np.abs(conditional_correlations[9]))/2

intrinsics_corrs = (np.abs(conditional_correlations[1]) + np.abs(conditional_correlations[2]))/2

sprouting_corrs = np.abs(conditional_correlations[14])

# fig, ax = plt.subplots(1, 3)
y = np.arange(in_loss_corrs.shape[0])
ax[2].hlines(np.arange(len(labels)),
             0,
             np.nanmax([sprouting_corrs, intrinsics_corrs, in_loss_corrs], axis=0),
             color='grey',
             alpha=alpha,
             linestyle=ls)
ax[2].plot(in_loss_corrs, y, marker='o', linestyle='None', markersize=20, color=colors[0], alpha=0.8)
ax[2].plot(sprouting_corrs, y, marker='d', linestyle='None', markersize=20, color=colors[1], alpha=0.8)
ax[2].plot(intrinsics_corrs, y, marker='s', linestyle='None', markersize=20, color=colors[2], alpha=0.8)
ax[2].set_yticks(np.arange(len(labels)))
# [0]... and label them with the respective list entries
ax[2].set_yticklabels(labels)
# plt.setp(ax[0].get_yticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor")
ax[2].set_xlabel("Absolute Marginal Correlation Coefficient")
ax[2].legend(("", "IN Loss", "Sprouting", "Intrinsic"))
ax[2].set_xlim((0, 1.0))


"""MAKE THE HISTOGRAM COMPARISON PLOT"""
colors = ["#1f78b4", "#b2df8a"]
ks_test_in_loss = data_file.root.in_loss.ks_test.read()
ks_test_hyperexcitable = data_file.root.hyperexcitable_v4.ks_test.read()
ks_test_sprouting = data_file.root.sprouting_only_v4.ks_test.read()

theta_in_loss_healthy = data_file.root.in_loss.theta_healthy.read()

theta_in_loss_in_loss = data_file.root.in_loss.theta_in_loss.read()

theta_he_healthy = data_file.root.hyperexcitable_v4.theta_healthy.read()

theta_he_unhealthy = data_file.root.hyperexcitable_v4.theta_hyperexcitable.read()

theta_sprouting_healthy = data_file.root.sprouting_only_v4.theta_healthy.read()

theta_sprouting_sprouting = data_file.root.sprouting_only_v4.theta_sprouted.read()

# in_loss_labels = deepcopy(labels)
in_loss_labels = conditionals.in_loss_conditional._get_unconditioned_labels()

sprouting_labels = conditionals.synapse_only_normal_conditional._get_unconditioned_labels()

excitable_labels = conditionals.intrinsics_normal_conditional._get_unconditioned_labels()

largest_statistic_in_loss = np.argsort(ks_test_in_loss[:, 0])[::-1]
largest_statistic_in_loss = largest_statistic_in_loss[:5]

largest_labels_in_loss = np.array(in_loss_labels)[largest_statistic_in_loss]
largest_parameters_in_loss_healthy = np.array(theta_in_loss_healthy)[:, largest_statistic_in_loss]
largest_parameters_in_loss_unhealthy = np.array(theta_in_loss_in_loss)[:, largest_statistic_in_loss]

largest_statistic_sprouting = np.argsort(ks_test_sprouting[:, 0])[::-1]
largest_statistic_sprouting = largest_statistic_sprouting[:5]

largest_labels_sprouting = np.array(sprouting_labels)[largest_statistic_sprouting]
largest_parameters_sprouting_healthy = np.array(theta_sprouting_healthy)[:, largest_statistic_sprouting]
largest_parameters_sprouting_unhealthy = np.array(theta_sprouting_sprouting)[:, largest_statistic_sprouting]

largest_statistic_he = np.argsort(ks_test_hyperexcitable[:, 0])[::-1]
largest_statistic_he = largest_statistic_he[:5]

largest_labels_he = np.array(excitable_labels)[largest_statistic_he]
largest_parameters_he_healthy = np.array(theta_he_healthy)[:, largest_statistic_he]
largest_parameters_he_unhealthy = np.array(theta_he_unhealthy)[:, largest_statistic_he]

fig, ax = plt.subplots(3, 5)
bins = 100
for idx, param in enumerate(largest_statistic_in_loss):
    ax[0, idx].hist(largest_parameters_in_loss_healthy[:, idx], bins=bins, color=colors[0], histtype=u'step')
    ax[0, idx].hist(largest_parameters_in_loss_unhealthy[:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[0, idx].set_xlabel(largest_labels_in_loss[idx])
ax[0, 0].legend(("Healthy", "IN Loss"))

for idx, param in enumerate(largest_statistic_sprouting):
    ax[1, idx].hist(largest_parameters_sprouting_healthy[:, idx], bins=bins, color=colors[0], histtype=u'step')
    ax[1, idx].hist(largest_parameters_sprouting_unhealthy[:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[1, idx].set_xlabel(largest_labels_sprouting[idx])
ax[1, 0].legend(("Healthy", "Sprouting"))

for idx, param in enumerate(largest_labels_he):
    ax[2, idx].hist(largest_parameters_he_healthy[:, idx], bins=bins, color=colors[0], histtype=u'step')
    ax[2, idx].hist(largest_parameters_he_unhealthy[:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[2, idx].set_xlabel(largest_labels_he[idx])
ax[2, 0].legend(("Healthy", "Hyperexcitable"))


















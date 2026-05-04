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
from scipy.stats import pearsonr

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
plt.rcParams.update({'font.size': 16})

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

"""SEQUENTIAL"""

output_path_replicate_01 = os.path.join(
    Metadata.results_dir,
    'conditionals_output_data_replicate_01.h5'
)

data_file_replicate_01 = tables.open_file(output_path_replicate_01, mode='r')

ks_test_in_loss_replicate_01 = data_file_replicate_01.root.in_loss.ks_test.read()
ks_test_in_loss_replicate_01 = np.insert(ks_test_in_loss_replicate_01, 4, [np.nan, np.nan], axis=0)
ks_test_in_loss_replicate_01 = np.insert(ks_test_in_loss_replicate_01, 9, [np.nan, np.nan], axis=0)

ks_test_hyperexcitable_replicate_01 = data_file_replicate_01.root.hyperexcitable.ks_test.read()
ks_test_hyperexcitable_replicate_01 = np.insert(ks_test_hyperexcitable_replicate_01, 1, [np.nan, np.nan], axis=0)
ks_test_hyperexcitable_replicate_01 = np.insert(ks_test_hyperexcitable_replicate_01, 2, [np.nan, np.nan], axis=0)

ks_test_sprouting_replicate_01 = data_file_replicate_01.root.sprouting.ks_test.read()
ks_test_sprouting_replicate_01 = np.insert(ks_test_sprouting_replicate_01, 14, [np.nan, np.nan], axis=0)


# --- replicate_02 ---
output_path_replicate_02 = os.path.join(
    Metadata.results_dir,
    'conditionals_output_data_replicate_02.h5'
)

data_file_replicate_02 = tables.open_file(output_path_replicate_02, mode='r')

ks_test_in_loss_replicate_02 = data_file_replicate_02.root.in_loss.ks_test.read()
ks_test_in_loss_replicate_02 = np.insert(ks_test_in_loss_replicate_02, 4, [np.nan, np.nan], axis=0)
ks_test_in_loss_replicate_02 = np.insert(ks_test_in_loss_replicate_02, 9, [np.nan, np.nan], axis=0)

ks_test_hyperexcitable_replicate_02 = data_file_replicate_02.root.hyperexcitable.ks_test.read()
ks_test_hyperexcitable_replicate_02 = np.insert(ks_test_hyperexcitable_replicate_02, 1, [np.nan, np.nan], axis=0)
ks_test_hyperexcitable_replicate_02 = np.insert(ks_test_hyperexcitable_replicate_02, 2, [np.nan, np.nan], axis=0)

ks_test_sprouting_replicate_02 = data_file_replicate_02.root.sprouting.ks_test.read()
ks_test_sprouting_replicate_02 = np.insert(ks_test_sprouting_replicate_02, 14, [np.nan, np.nan], axis=0)



colors = ['#66c2a5', '#fc8d62', '#8da0cb']

plt.figure()
plt.scatter(ks_test_in_loss[:, 0], ks_test_in_loss_replicate_01[:, 0], color=colors[0])
plt.scatter(ks_test_sprouting[:, 0], ks_test_sprouting_replicate_01[:, 0], color=colors[1])
plt.scatter(ks_test_hyperexcitable[:, 0], ks_test_hyperexcitable_replicate_01[:, 0], color=colors[2])
plt.scatter(ks_test_in_loss[:, 0], ks_test_in_loss_replicate_02[:, 0], color=colors[0], marker='x')
plt.scatter(ks_test_sprouting[:, 0], ks_test_sprouting_replicate_02[:, 0], color=colors[1], marker='x')
plt.scatter(ks_test_hyperexcitable[:, 0], ks_test_hyperexcitable_replicate_02[:, 0], color=colors[2], marker='x')
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.legend(["IN Loss", "Sprouting", "Intrinsic"])


plt.xlabel("KS Statistic Original")
plt.ylabel("KS Statistic Replicate 01")
# plt.legend(["IN Loss", "Hyperexcitable", "Sprouting"])

# ks_test_all = data_file.root.all.ks_test.read()
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

"""CORRELATION COMPARISON"""
# ks_test_all = data_file.root.all.ks_test.read()


# -------------------------
# replicate_01
# -------------------------
correlations_path_replicate_01 = os.path.join(
    Metadata.results_dir,
    'marginal_and_conditional_correlation_matrices_replicate_01.h5'
)

correlation_file_replicate_01 = tables.open_file(correlations_path_replicate_01, mode='r')

marginal_correlations_replicate_01 = correlation_file_replicate_01.root.marginal_correlations.read()
conditional_correlations_replicate_01 = correlation_file_replicate_01.root.conditional_correlations.read()

np.fill_diagonal(marginal_correlations_replicate_01, np.nan)
np.fill_diagonal(conditional_correlations_replicate_01, np.nan)

in_loss_corrs_replicate_01 = (
    np.abs(conditional_correlations_replicate_01[4]) +
    np.abs(conditional_correlations_replicate_01[9])
) / 2

intrinsics_corrs_replicate_01 = (
    np.abs(conditional_correlations_replicate_01[1]) +
    np.abs(conditional_correlations_replicate_01[2])
) / 2

sprouting_corrs_replicate_01 = np.abs(conditional_correlations_replicate_01[14])


# -------------------------
# replicate_02
# -------------------------
correlations_path_replicate_02 = os.path.join(
    Metadata.results_dir,
    'marginal_and_conditional_correlation_matrices_replicate_02.h5'
)

correlation_file_replicate_02 = tables.open_file(correlations_path_replicate_02, mode='r')

marginal_correlations_replicate_02 = correlation_file_replicate_02.root.marginal_correlations.read()
conditional_correlations_replicate_02 = correlation_file_replicate_02.root.conditional_correlations.read()

np.fill_diagonal(marginal_correlations_replicate_02, np.nan)
np.fill_diagonal(conditional_correlations_replicate_02, np.nan)

in_loss_corrs_replicate_02 = (
    np.abs(conditional_correlations_replicate_02[4]) +
    np.abs(conditional_correlations_replicate_02[9])
) / 2

intrinsics_corrs_replicate_02 = (
    np.abs(conditional_correlations_replicate_02[1]) +
    np.abs(conditional_correlations_replicate_02[2])
) / 2

sprouting_corrs_replicate_02 = np.abs(conditional_correlations_replicate_02[14])


plt.figure()
plt.scatter(marginal_correlations.flatten(), marginal_correlations_replicate_01.flatten(), color='k', alpha=0.2)
plt.scatter(marginal_correlations.flatten(), marginal_correlations_replicate_02.flatten(), color='k', marker='x', alpha=0.2)
plt.plot([-1, 1], [-1, 1], linestyle='--', color='k')
plt.xlabel("Marginal Correlations Original")
plt.ylabel("Marginal Correlations Replicates")

plt.figure()
plt.scatter(conditional_correlations.flatten(), conditional_correlations_replicate_01.flatten(), color='k', alpha=0.2)
plt.scatter(conditional_correlations.flatten(), conditional_correlations_replicate_02.flatten(), color='k', marker='x', alpha=0.2)
plt.plot([-1, 1], [-1, 1], linestyle='--', color='k')
plt.xlabel("Conditional Correlations Original")
plt.ylabel("Conditional Correlations Replicates")

"""CORRELATION COEFFICIENTS"""
orig = marginal_correlations.flatten()
rep1 = marginal_correlations_replicate_01.flatten()
rep2 = marginal_correlations_replicate_02.flatten()

# Concatenate replicates
rep_all = np.concatenate([rep1, rep2])

# Repeat original to match shape
orig_all = np.concatenate([orig, orig])

# Remove NaNs jointly
values = ~np.isnan(orig_all) & ~np.isnan(rep_all)

corr_marginal, p_marginal = pearsonr(orig_all[values], rep_all[values])

print("Marginal correlation (original vs both replicates):", corr_marginal)


orig = conditional_correlations.flatten()
rep1 = conditional_correlations_replicate_01.flatten()
rep2 = conditional_correlations_replicate_02.flatten()

rep_all = np.concatenate([rep1, rep2])
orig_all = np.concatenate([orig, orig])

values = ~np.isnan(orig_all) & ~np.isnan(rep_all)

corr_conditional, p_conditional = pearsonr(orig_all[values], rep_all[values])

print("Conditional correlation (original vs both replicates):", corr_conditional)


# ---------- originals ----------
orig_all = np.concatenate([
    ks_test_in_loss[:, 0],
    ks_test_sprouting[:, 0],
    ks_test_hyperexcitable[:, 0]
])

# ---------- replicates ----------
rep_all = np.concatenate([
    ks_test_in_loss_replicate_01[:, 0],
    ks_test_sprouting_replicate_01[:, 0],
    ks_test_hyperexcitable_replicate_01[:, 0],
    ks_test_in_loss_replicate_02[:, 0],
    ks_test_sprouting_replicate_02[:, 0],
    ks_test_hyperexcitable_replicate_02[:, 0]
])

# Repeat originals to align with two replicates
orig_all_repeated = np.concatenate([orig_all, orig_all])

# Remove NaNs jointly
values = ~np.isnan(orig_all_repeated) & ~np.isnan(rep_all)

corr, p_value = pearsonr(orig_all_repeated[values], rep_all[values])

print("Pooled KS correlation (original vs both replicates):", corr)



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
from sbi.analysis import conditional_corrcoeff, conditional_pairplot, pairplot
from metadata import EpilepsyMetadata

# results_dir = '/flash/FukaiU/danielmk/sbiemd/truncated_sequential_npe_restricted_network_baseline_net_one_simulator_constant_dynamics_short_fewer_pc_cython_one.pickle'
# results_dir = r'truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_interictal_spiking_01.pickle'

np.random.seed(321)

torch.manual_seed(104)

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

posterior = inference.build_posterior().set_default_x(x_o)

posterior_samples = posterior.sample((200000,), x=x_o)

correlations = spearmanr(posterior_samples)

labels = list(priors.baseline_epilepsy['low'].keys())

labels = [r'$PC_C$',
          r'$PC_{g_L}$',
          r'$PC_{E_L}$',
          r'$PC_{V_T}$',
          r'$AIN_N$',
          r'$AIN_C$',
          r'$AIN_{g_L}$',
          r'$AIN_{E_L}$',
          r'$AIN_{V_T}$',
          r'$NIN_N$',
          r'$NIN_C$',
          r'$NIN_{g_L}$',
          r'$NIN_{E_L}$',
          r'$NIN_{V_T}$',
          r'$PC-PC_P$',
          r'$PC-PC_{A_{SE}}$',
          r'$PC-AIN_P$',
          r'$PC-AIN_{A_{SE}}$',
          r'$PC-NIN_P$',
          r'$PC-NIN_{A_{SE}}$',
          r'$NIN-PC_P$',
          r'$NIN-PC_{A_{SE}}$',
          r'$NIN-NIN_P$',
          r'$NIN-NIN_{A_{SE}}$',
          r'$AIN-PC_P$',
          r'$AIN-PC_{A_{SE}}$',
          r'$AIN-NIN_P$',
          r'$AIN-NIN_{A_{SE}}$',
          r'$AIN-AIN_P$',
          r'$AIN-AIN_{A_{SE}}$',
          r'$NIN-NIN_{GAP_P}$',
          r'$NIN-NIN_{GAP_W}$',
          ]


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0., vmax=1)
# pcolormesh(your_data, cmap="coolwarm", norm=divnorm)

# fig, ax = plt.subplots(1, 2)

# pos_statistic = ax[0].imshow(correlations.correlation, cmap="coolwarm", origin='lower', interpolation=None, norm=divnorm)

# fig.colorbar(pos_statistic, ax=ax[0])

# pos_significance = ax[1].imshow(correlations[1], cmap="Greys_r", origin='lower', interpolation=None)

# fig.colorbar(pos_significance, ax=ax[1])

"""LABELED HEATMAP"""
fig, ax = plt.subplots()

marginal_corr_matrix = correlations.correlation

np.fill_diagonal(marginal_corr_matrix, 0.0)

im = ax.imshow(marginal_corr_matrix, cmap="coolwarm", origin='lower', interpolation=None, norm=divnorm)

fig.colorbar(im, ax=ax)

# We want to show all ticks...
ax.set_xticks(np.arange(len(EpilepsyMetadata.parameter_labels)))
ax.set_yticks(np.arange(len(EpilepsyMetadata.parameter_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(EpilepsyMetadata.parameter_labels)
ax.set_yticklabels(EpilepsyMetadata.parameter_labels)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

fig.tight_layout()

"""PLOT PAIRS WITH MEANINGFUL (>0.1) MARGINAL INTERACTIONS"""
meaningful_correlations = np.absolute(correlations.correlation) > 0.1
meaningful_upper = np.triu(meaningful_correlations, k=1)

meaningful_indices = np.argwhere(meaningful_upper)

ind_sqrt = np.sqrt(meaningful_indices.shape[0])
rows = 3
cols = 7

fig, ax = plt.subplots(rows, cols, constrained_layout=True)
ax = ax.flatten()
hist_list = []
for idx, idc in enumerate(meaningful_indices):
    curr_hist = ax[idx].hist2d(posterior_samples[:, idc[0]], posterior_samples[:, idc[1]], bins=50, cmap="Greys_r", rasterized=True)
    hist_list.append(curr_hist)
    ax[idx].set_xlabel(EpilepsyMetadata.parameter_labels[idc[0]])
    ax[idx].set_ylabel(EpilepsyMetadata.parameter_labels[idc[1]])

for a in ax:
    a.set_box_aspect(1)

max_count = np.max([x[0].max() for x in hist_list])

fig.tight_layout()


#CALCULATE CONDITIONAL CORRELATIONS
prior = priors.baseline_epilepsy

prior_tensor = priors.prior_dict_to_pure_tensor(prior)

prior_tensor = torch.vstack(prior_tensor)

healthy_map = posterior.map()

cond_coeff_mat = conditional_corrcoeff(
    density=posterior,
    condition=healthy_map,
    limits=prior_tensor.T,
)

"""CONDITIONAL HEATMAP"""
fig, ax = plt.subplots()
conditional_corr_matrix = np.array(cond_coeff_mat)

np.fill_diagonal(conditional_corr_matrix, 0.0)

im = ax.imshow(conditional_corr_matrix, cmap="coolwarm", origin='lower', interpolation=None, norm=divnorm)

fig.colorbar(im, ax=ax)

# We want to show all ticks...
ax.set_xticks(np.arange(len(EpilepsyMetadata.parameter_labels)))
ax.set_yticks(np.arange(len(EpilepsyMetadata.parameter_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(EpilepsyMetadata.parameter_labels)
ax.set_yticklabels(EpilepsyMetadata.parameter_labels)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")


plt.rcParams.update({'font.size': 22})
conditional_pairplot(
    density=posterior,
    condition=healthy_map,
    limits=prior_tensor.T,
    labels=EpilepsyMetadata.parameter_labels)

"""SCATTERPLOT MARGINAL VS CONDITIONAL"""
u_idc = np.triu_indices(32, k=1)

marginal_corr_array = np.array(marginal_corr_matrix[u_idc])

conditional_corr_array = np.array(conditional_corr_matrix[u_idc])

fig, ax = plt.subplots()
ax.scatter(marginal_corr_array, conditional_corr_array, color='k')
ax.set_xlabel("Marginal Correlation Coefficient")
ax.set_ylabel("MAP Conditioned Correlation Coefficient")
ax.hlines([-0.1, 0.1], -1, 1, color='k', linestyle='dashed')
ax.vlines([-0.1, 0.1], -1, 1, color='k', linestyle='dashed')
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
ax.set_aspect('equal', 'box')

sys.exit()

"""SAVE THE CORRELATION MATRICES"""
output_path = os.path.join(EpilepsyMetadata.results_dir, 'marginal_and_conditional_correlation_matrices.h5')

with tables.open_file(output_path, mode='a') as output:
    output.create_array(f'/', 'conditional_correlations', obj=conditional_corr_matrix)
    output.create_array(f'/', 'marginal_correlations', obj=marginal_corr_matrix)




"""
x = np.arange(0, cond_coeff_mat_flat[cond_coeff_mat_flat_sorted[:-71:-1]].shape[0], 1)
tick_labels = np.array(cross_corr_labels)[cond_coeff_mat_flat_sorted[:-71:-1]]
fig, ax = plt.subplots()
plt.xticks(x, tick_labels)
plt.plot(cond_coeff_mat_flat[cond_coeff_mat_flat_sorted[:-71:-1]], 'o', markersize=20, color='k')
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
plt.ylabel("Spearman Correlation Coefficient")
plt.hlines(0, x[0]-1, x[-1]+1, color='k')
plt.xlim((x[0]-1, x[-1]+1))

plt.rcParams.update({'font.size': 10})
fig, ax = pairplot(
    samples=posterior_samples,
    limits=prior_tensor.T,
    labels=EpilepsyMetadata.parameter_labels,
    aspect='equal',
)

fig, ax = plt.subplots()
plt.xticks(x, tick_labels)
plt.plot(np.abs(cond_coeff_mat_flat[cond_coeff_mat_flat_sorted[:-71:-1]]), 'o', markersize=20, color='k')
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
plt.ylabel("Spearman Correlation Coefficient")
plt.hlines(0, x[0]-1, x[-1]+1, color='k')
plt.xlim((x[0]-1, x[-1]+1))
"""


"""

"""

"""
x = np.arange(0, correlations_flat[correlations_flat_sorted[:-20:-1]].shape[0], 1)
tick_labels = np.array(cross_corr_labels)[correlations_flat_sorted[:-20:-1]]
fig, ax = plt.subplots()
plt.xticks(x, tick_labels)
plt.plot(correlations_flat[correlations_flat_sorted[:-20:-1]], 'o', markersize=20, color='k')
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
plt.ylabel("Spearman Correlation Coefficient")
plt.hlines(0, x[0]-1, x[-1]+1, color='k')
plt.xlim((x[0]-1, x[-1]+1))
"""

"""CREATE SCATTERPLOTS OR HEATMAPS OF THE SIGNIFICANT INTERACTIONS"""
"""
significance_upper = np.triu(significance, k=1)

significance_indices = np.argwhere(significance_upper)

meaningful_upper = np.triu(c, k=1)

meaningful_indices = np.argwhere(meaningful_upper)

ind_sqrt = np.sqrt(significance_indices.shape[0])

rows = int(np.ceil(ind_sqrt))

cols = int(np.ceil(ind_sqrt))

fig, ax = plt.subplots(rows, cols)
ax = ax.flatten()
for idx, idc in enumerate(significance_indices):
    ax[idx].hist2d(posterior_samples[:, idc[0]], posterior_samples[:, idc[1]], bins=50, cmap="Greys_r")
    ax[idx].set_xlabel(labels[idc[0]])
    ax[idx].set_ylabel(labels[idc[1]])


for a in ax:
    a.set_box_aspect(1)
"""


"""LABELED HEATMAP SIGNIFICANCE"""
"""
fig, ax = plt.subplots()

ut_idc = np.triu_indices(32, k=1)

labels_array = np.array(labels)

cross_corr_labels = list(map(' x '.join, zip(labels_array[ut_idc[0]], labels_array[ut_idc[1]])))

pvalues_flat = correlations.pvalue[ut_idc]

pvalues_flat_sorted = np.argsort(pvalues_flat)

correlations_flat = correlations.correlation[ut_idc]

correlations_flat_sorted = np.argsort(np.abs(correlations_flat))

correlations_meaningful = np.argwhere(np.abs(correlations_flat) > 0.1)

alpha = 0.0001

corrected_alpha = (alpha / len(labels))

significance = correlations.pvalue < corrected_alpha




im = ax.imshow(significance, cmap="Greys_r", origin='lower', interpolation=None)

fig.colorbar(im, ax=ax)

# We want to show all ticks...
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
"""
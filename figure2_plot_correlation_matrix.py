# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import torch
import priors
import outcomes
import os
import pickle
import tables
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from metadata import Metadata
from matplotlib import colors
from sbi.analysis.conditional_density import conditional_corrcoeff
from sbi.analysis.plot import conditional_pairplot


np.random.seed(321)

torch.manual_seed(104)

f = os.path.join(Metadata.results_dir, "truncated_sequential_npe_network_baseline_conductance_based_01_baseline.pickle")


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

posterior = inference.build_posterior().set_default_x(x_o)

posterior_samples = posterior.sample((200000,), x=x_o)

correlations = spearmanr(posterior_samples)

labels = list(priors.baseline['low'].keys())

labels = Metadata.parameter_labels


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

divnorm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0., vmax=1)

"""LABELED HEATMAP"""
fig, ax = plt.subplots()

marginal_corr_matrix = correlations.correlation

np.fill_diagonal(marginal_corr_matrix, 0.0)

im = ax.imshow(marginal_corr_matrix, cmap="coolwarm", origin='lower', interpolation=None, norm=divnorm)

fig.colorbar(im, ax=ax)

# We want to show all ticks...
ax.set_xticks(np.arange(len(Metadata.parameter_labels)))
ax.set_yticks(np.arange(len(Metadata.parameter_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(Metadata.parameter_labels)
ax.set_yticklabels(Metadata.parameter_labels)
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
    curr_hist = ax[idx].hist2d(posterior_samples[:, idc[0]],
                               posterior_samples[:, idc[1]],
                               bins=50,
                               cmap="Greys_r",
                               rasterized=True)
    hist_list.append(curr_hist)
    ax[idx].set_xlabel(Metadata.parameter_labels[idc[0]])
    ax[idx].set_ylabel(Metadata.parameter_labels[idc[1]])

for a in ax:
    a.set_box_aspect(1)

max_count = np.max([x[0].max() for x in hist_list])

fig.tight_layout()


# CALCULATE CONDITIONAL CORRELATIONS
prior = priors.baseline

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
ax.set_xticks(np.arange(len(Metadata.parameter_labels)))
ax.set_yticks(np.arange(len(Metadata.parameter_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(Metadata.parameter_labels)
ax.set_yticklabels(Metadata.parameter_labels)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")


plt.rcParams.update({'font.size': 22})
conditional_pairplot(
    density=posterior,
    condition=healthy_map,
    limits=prior_tensor.T,
    labels=Metadata.parameter_labels)

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

"""SAVE THE CORRELATION MATRICES"""
output_path = os.path.join(Metadata.results_dir, 'marginal_and_conditional_correlation_matrices.h5')

with tables.open_file(output_path, mode='a') as output:
    output.create_array('/', 'conditional_correlations', obj=conditional_corr_matrix)
    output.create_array('/', 'marginal_correlations', obj=marginal_corr_matrix)

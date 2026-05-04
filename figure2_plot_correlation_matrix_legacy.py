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
import sbi


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

"""MAP CONDITIONAL SAMPLES"""
prior = priors.baseline
prior_tensor = priors.prior_dict_to_tensor(prior)
n_samples = 100000
n_parameters = len(Metadata.parameter_labels)
conditional_samples_array = np.empty((n_parameters, n_parameters, n_samples, 2))

potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
    inference._neural_net, prior=prior_tensor, x_o=x_o
)

for i in range(0, n_parameters):
    for j in range(i+1, n_parameters):
        curr_sample_list = []
        dims_to_sample = np.array([i, j])
        

        conditioned_potential_fn, restricted_tf, restricted_prior = sbi.analysis.conditional_potential(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            prior=prior_tensor,
            condition=healthy_map,
            dims_to_sample=dims_to_sample,
        )
        
        mcmc_posterior = sbi.inference.MCMCPosterior(
            potential_fn=conditioned_potential_fn,
            theta_transform=restricted_tf,
            proposal=restricted_prior,
            method="slice_np_vectorized",
            num_chains=20,
        ).set_default_x(x_o)
        
        curr_samples = mcmc_posterior.sample((n_samples,))
    
        conditional_samples_array[i,j] = np.array(curr_samples)
        conditional_samples_array[j,i] = np.array(curr_samples)
"""
conditional_samples_file = tables.open_file(os.path.join(Metadata.results_dir, 'conditional_samples_replicate_02.h5'))

conditional_samples = conditional_samples_file.root.conditional_samples.read()

indices_to_plot = [[0, 1],
                   [0, 2],
                   [0, 3],
                   [0, 15],
                   [0, 24],
                   [1, 3],
                   [1, 14],
                   ]

fig, ax = plt.subplots(rows, cols, constrained_layout=True)

ax = ax.flatten()
hist_list = []
for idx, idc in enumerate(indices_to_plot):
    curr_hist = ax[idx].hist2d(conditional_samples[idc[0], idc[1], :, 0],
                               conditional_samples[idc[0], idc[1], :, 1],
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
"""
"""SAVE THE CORRELATION MATRICES"""
output_path = os.path.join(Metadata.results_dir, 'conditional_samples_baseline.h5')

with tables.open_file(output_path, mode='a') as output:
    output.create_array('/', 'conditional_samples', obj=conditional_samples_array)

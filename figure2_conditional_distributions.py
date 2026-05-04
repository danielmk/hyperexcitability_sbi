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

f = os.path.join(Metadata.results_dir, "truncated_sequential_npe_network_baseline_conductance_based_01_baseline_replicate_01.pickle")


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


"""POSTERIOR CONDITIONAL SAMPLES"""
prior = priors.baseline
prior_tensor = priors.prior_dict_to_tensor(prior)
n_marginal_samples = 100
n_conditional_samples = 200
n_parameters = len(Metadata.parameter_labels)
marginal_samples = posterior.sample((n_marginal_samples,), x=x_o)
conditional_samples_array = np.empty((n_parameters, n_parameters, n_marginal_samples * n_conditional_samples, 2))

potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
    inference._neural_net, prior=prior_tensor, x_o=x_o
)

for i in range(0, n_parameters):
    for j in range(i+1, n_parameters):
        curr_sample_list = []
        dims_to_sample = np.array([i, j])
        
        for s in marginal_samples:
            conditioned_potential_fn, restricted_tf, restricted_prior = sbi.analysis.conditional_potential(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior_tensor,
                condition=torch.as_tensor(s, dtype=torch.float),
                dims_to_sample=dims_to_sample,
            )
            
            mcmc_posterior = sbi.inference.MCMCPosterior(
                potential_fn=conditioned_potential_fn,
                theta_transform=restricted_tf,
                proposal=restricted_prior,
                method="slice_np_vectorized",
                num_chains=1,
            ).set_default_x(x_o)
            
            curr_samples = mcmc_posterior.sample((n_conditional_samples,))
            curr_sample_list.append(curr_samples)
    
        conditional_samples_array[i,j] = np.array(curr_sample_list).reshape(20000, 2)
            
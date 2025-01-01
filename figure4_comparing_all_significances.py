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
from scipy.stats import t
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

# results_dir = '/flash/FukaiU/danielmk/sbiemd/truncated_sequential_npe_restricted_network_baseline_net_one_simulator_constant_dynamics_short_fewer_pc_cython_one.pickle'
# results_dir = r'truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_interictal_spiking_01.pickle'

n_samples = 100000

np.random.seed(321)

torch.manual_seed(45234567)

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

labels = EpilepsyMetadata.parameter_labels

col_labels = ['Mean Rate', 'Mean Entropy', 'Theta Power', ' Gamma Power', 'Fast Power', 'Correlation' , 'CV']

if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
elif platform.system() == 'Linux':
    results_dir = r'/flash/FukaiU/danielmk/sbiemd/'

correlations_path = os.path.join(EpilepsyMetadata.results_dir, 'marginal_and_conditional_correlation_matrices.h5')

healthy_vs_healthy_path = os.path.join(results_dir, 'conditionals_output_data.h5')

healthy_vs_hyperexcitable_path = os.path.join(results_dir, 'conditionals_output_data_healthy_vs_hyperexcitable.h5')

correlations_file = tables.open_file(correlations_path, mode='r')

healthy_vs_healthy_file = tables.open_file(healthy_vs_healthy_path, mode='r')

healthy_vs_hyperexcitable_file = tables.open_file(healthy_vs_hyperexcitable_path, mode='r')

"""CORRELATIONS SIGNIFICANCE"""
# We need to calculate the p-values from the correlation coefficients
n = 200000
r = correlations_file.root.marginal_correlations.read()
t_stat = (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2)

dof = n - 2

p_values = 2*(1 - t.cdf(abs(t_stat), dof))

condition_idc = [4, 9, 1, 2, 14]

conditions = [labels[x] for x in condition_idc]

correlations_p_values = {x: None for x in conditions}
correlations_ks_stat = {x: None for x in conditions}

for idx, key in enumerate(correlations_p_values):
    correlations_p_values[key] = np.abs(p_values[condition_idc[idx]])
    correlations_ks_stat[key] = np.abs(r[condition_idc[idx]])

"""HEALTHY-VS-HEALTHY SIGNIFICANCE"""
conditions = ['in_loss', 'hyperexcitable_v4', 'sprouting_only_v4']
healthy_vs_healthy_p_values = {x: None for x in conditions}
healthy_vs_healthy_ks_stat = {x: None for x in conditions}

for k in healthy_vs_healthy_p_values.keys():
    curr_p = np.array(healthy_vs_healthy_file.root[k].ks_test.read())[:,1]
    curr_stat = np.array(healthy_vs_healthy_file.root[k].ks_test.read())[:,0]
    healthy_vs_healthy_p_values[k] = curr_p
    healthy_vs_healthy_ks_stat[k] = curr_stat

healthy_vs_healthy_p_values['in_loss'] = np.insert(healthy_vs_healthy_p_values['in_loss'], [4], np.nan)
healthy_vs_healthy_p_values['in_loss'] = np.insert(healthy_vs_healthy_p_values['in_loss'], [9], np.nan)

healthy_vs_healthy_p_values['hyperexcitable_v4'] = np.insert(healthy_vs_healthy_p_values['hyperexcitable_v4'], [1], np.nan)
healthy_vs_healthy_p_values['hyperexcitable_v4'] = np.insert(healthy_vs_healthy_p_values['hyperexcitable_v4'], [2], np.nan)

healthy_vs_healthy_p_values['sprouting_only_v4'] = np.insert(healthy_vs_healthy_p_values['sprouting_only_v4'], [14], np.nan)

healthy_vs_healthy_ks_stat['in_loss'] = np.insert(healthy_vs_healthy_ks_stat['in_loss'], [4], np.nan)
healthy_vs_healthy_ks_stat['in_loss'] = np.insert(healthy_vs_healthy_ks_stat['in_loss'], [9], np.nan)

healthy_vs_healthy_ks_stat['hyperexcitable_v4'] = np.insert(healthy_vs_healthy_ks_stat['hyperexcitable_v4'], [1], np.nan)
healthy_vs_healthy_ks_stat['hyperexcitable_v4'] = np.insert(healthy_vs_healthy_ks_stat['hyperexcitable_v4'], [2], np.nan)

healthy_vs_healthy_ks_stat['sprouting_only_v4'] = np.insert(healthy_vs_healthy_ks_stat['sprouting_only_v4'], [14], np.nan)
    
"""HEALTHY-VS-HYPEREXCITABLE SIGNIFICANCE"""
conditions = ['in_loss', 'intrinsics', 'sprouting_only_v3']
healthy_vs_hyperexcitable_p_values = {x: None for x in conditions}
healthy_vs_hyperexcitable_ks_stat = {x: None for x in conditions}

for k in healthy_vs_hyperexcitable_p_values.keys():
    curr_p = np.array(healthy_vs_hyperexcitable_file.root[k].ks_test.read())[:,1]
    curr_stat = np.array(healthy_vs_hyperexcitable_file.root[k].ks_test.read())[:,0]
    healthy_vs_hyperexcitable_p_values[k] = curr_p
    healthy_vs_hyperexcitable_ks_stat[k] = curr_stat

healthy_vs_hyperexcitable_p_values['in_loss'] = np.insert(healthy_vs_hyperexcitable_p_values['in_loss'], [4], np.nan)
healthy_vs_hyperexcitable_p_values['in_loss'] = np.insert(healthy_vs_hyperexcitable_p_values['in_loss'], [9], np.nan)

healthy_vs_hyperexcitable_p_values['intrinsics'] = np.insert(healthy_vs_hyperexcitable_p_values['intrinsics'], [1], np.nan)
healthy_vs_hyperexcitable_p_values['intrinsics'] = np.insert(healthy_vs_hyperexcitable_p_values['intrinsics'], [2], np.nan)

healthy_vs_hyperexcitable_p_values['sprouting_only_v3'] = np.insert(healthy_vs_hyperexcitable_p_values['sprouting_only_v3'], [14], np.nan)

healthy_vs_hyperexcitable_ks_stat['in_loss'] = np.insert(healthy_vs_hyperexcitable_ks_stat['in_loss'], [4], np.nan)
healthy_vs_hyperexcitable_ks_stat['in_loss'] = np.insert(healthy_vs_hyperexcitable_ks_stat['in_loss'], [9], np.nan)

healthy_vs_hyperexcitable_ks_stat['intrinsics'] = np.insert(healthy_vs_hyperexcitable_ks_stat['intrinsics'], [1], np.nan)
healthy_vs_hyperexcitable_ks_stat['intrinsics'] = np.insert(healthy_vs_hyperexcitable_ks_stat['intrinsics'], [2], np.nan)

healthy_vs_hyperexcitable_ks_stat['sprouting_only_v3'] = np.insert(healthy_vs_hyperexcitable_ks_stat['sprouting_only_v3'], [14], np.nan)


"""CONSTRUCT THE IMAGES"""
cmap = matplotlib.colors.ListedColormap(['#ffff99', '#386cb0'])

p = 0.0001 / 32

in_loss_corr = ()

in_loss_image_p = np.vstack([correlations_p_values['$AIN_N$'], correlations_p_values['$NIN_N$'], healthy_vs_healthy_p_values['in_loss'],  healthy_vs_hyperexcitable_p_values['in_loss']])
intrinsics_image_p = np.vstack([correlations_p_values['$PC_{g_L}$'], correlations_p_values['$PC_{E_L}$'], healthy_vs_healthy_p_values['hyperexcitable_v4'],  healthy_vs_hyperexcitable_p_values['intrinsics']])
sprouting_image_p = np.vstack([correlations_p_values['$PC-PC_P$'], healthy_vs_healthy_p_values['sprouting_only_v4'],  healthy_vs_hyperexcitable_p_values['sprouting_only_v3']])

in_loss_image_s = np.vstack([correlations_ks_stat['$AIN_N$'], correlations_ks_stat['$NIN_N$'], healthy_vs_healthy_ks_stat['in_loss'],  healthy_vs_hyperexcitable_ks_stat['in_loss']])
intrinsics_image_s = np.vstack([correlations_ks_stat['$PC_{g_L}$'], correlations_ks_stat['$PC_{E_L}$'], healthy_vs_healthy_ks_stat['hyperexcitable_v4'],  healthy_vs_hyperexcitable_ks_stat['intrinsics']])
sprouting_image_s = np.vstack([correlations_ks_stat['$PC-PC_P$'], healthy_vs_healthy_ks_stat['sprouting_only_v4'],  healthy_vs_hyperexcitable_ks_stat['sprouting_only_v3']])


fig, ax = plt.subplots(1, 3)
ax[0].imshow(in_loss_image_p.T < p, aspect='auto', cmap=cmap,  origin='lower', interpolation=None)

ax[1].imshow(intrinsics_image_p.T < p, aspect='auto', cmap=cmap, origin='lower', interpolation=None)

im = ax[2].imshow(sprouting_image_p.T < p, aspect='auto', cmap=cmap, origin='lower', interpolation=None)

divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='5%', pad=0.05)

fig.colorbar(im, cax=cax, orientation='vertical')

ax[0].set_title("IN Loss")
ax[1].set_title("Intrinsics")
ax[2].set_title("Sprouting")

ax[0].set_yticks(np.arange(len(EpilepsyMetadata.parameter_labels)))
ax[0].set_yticklabels(EpilepsyMetadata.parameter_labels)


x_ticks_zero = ['$AIN_N$', '$NIN_N$', 'IN Loss x Normal', 'Baseline x Hyperexcitable']
x_ticks_one = ['$PC_{g_L}$', '$PC_{E_L}$', 'Hyperexcitable x Normal', 'Baseline x Hyperexcitable']
x_ticks_two = ['$PC-PC_P$', 'Sprouting x Normal', 'Baseline x Hyperexcitable']

ax[0].set_xticks(np.arange(len(x_ticks_zero)))
ax[0].set_xticklabels(x_ticks_zero)

ax[1].set_xticks(np.arange(len(x_ticks_one)))
ax[1].set_xticklabels(x_ticks_one)

ax[2].set_xticks(np.arange(len(x_ticks_two)))
ax[2].set_xticklabels(x_ticks_two)

sys.exit()



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

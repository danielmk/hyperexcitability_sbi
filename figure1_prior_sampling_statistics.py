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
import outcomes
import os
import pickle
from datetime import datetime
import tables
import numpy as np
from itertools import chain
import sys
from sbi import analysis as analysis
import platform
import matplotlib.pyplot as plt

# results_dir = '/flash/FukaiU/danielmk/sbiemd/truncated_sequential_npe_restricted_network_baseline_net_one_simulator_constant_dynamics_short_fewer_pc_cython_one.pickle'

if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
    f = os.path.join(results_dir, "amortized_inference_with_nan_sbi-main.pickle")
elif platform.system() == 'Linux':
    results_dir = r'/flash/FukaiU/danielmk/sbiemd/'
    f = os.path.join(results_dir, "amortized_inference_with_nan_sbi-main.pickle")

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

data = list(loadall(f))

final = data[0]

final = data[-1]

data = final.get_simulations()

active = data[1][:, 0] > 0

cv_defined = data[1][:, 6] > 0

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})
# plt.rcParams["font.family"] = "Arial"

colors = ['#1b9e77', '#d95f02', '#7570b3']

fig, ax = plt.subplots(1, 4)

labels = ['Mean Rate', 'Mean Entropy', 'Theta Power', ' Gamma Power', 'Fast Power', 'Correlation' , 'ISI CV']

alpha = 0.01

x, y = 0, 1
ax[0].plot(data[1][active & cv_defined, x], data[1][active & cv_defined, y], 'o', rasterized=True, color='k', alpha=alpha)
ax[0].set_xlabel(labels[x])
ax[0].set_ylabel(labels[y])
ax[0].set_box_aspect(1)

x, y = 2, 3
ax[1].plot(data[1][active & cv_defined, x], data[1][active & cv_defined, y], 'o', rasterized=True, color='k', alpha=alpha)
ax[1].set_xlabel(labels[x])
ax[1].set_ylabel(labels[y])
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_box_aspect(1)

x, y = 2, 4
ax[2].plot(data[1][active & cv_defined, x], data[1][active & cv_defined, y], 'o', rasterized=True, color='k', alpha=alpha)
ax[2].set_xlabel(labels[x])
ax[2].set_ylabel(labels[y])
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_box_aspect(1)

x, y = 5, 6
ax[3].plot(data[1][active & cv_defined, x], data[1][active & cv_defined, y], 'o', rasterized=True, color='k', alpha=alpha)
ax[3].set_xlabel(labels[x])
ax[3].set_ylabel(labels[y])
ax[3].set_box_aspect(1)

"""OUTPUT STATISTICS FROM figure1_map_estimates.py"""
"""
healthy_map_outcome = np.array([1.7616e+01, 2.6857e+00, 1.0933e-07, 1.0157e-08, 1.0624e-09, 3.7278e-02,
                              4.2526e-01])

interictal_map_outcome = np.array([6.6824e+01, 9.4704e-01, 8.3128e-06, 8.4021e-09, 8.5110e-10, 9.8949e-01,
        4.1919e+00])
"""
healthy_map_outcome = outcomes.x_healthy_v2['x_healthy_v2']

interictal_map_outcome = outcomes.x_interictal_spiking['x_interictal_spiking']

ms = 40
mw = 5

ax[0].plot(healthy_map_outcome[0], healthy_map_outcome[1], 'x', color=colors[0], markersize=ms, markeredgewidth=mw)
ax[0].plot(interictal_map_outcome[0], interictal_map_outcome[1], 'x', color=colors[1], markersize=ms, markeredgewidth=mw)
# ax[0].plot(synchrony_map_outcome[0], synchrony_map_outcome[1], 'x', color=colors[2], markersize=ms, markeredgewidth=mw)

ax[1].plot(healthy_map_outcome[2], healthy_map_outcome[3], 'x', color=colors[0], markersize=ms, markeredgewidth=mw)
ax[1].plot(interictal_map_outcome[2], interictal_map_outcome[3], 'x', color=colors[1], markersize=ms, markeredgewidth=mw)
# ax[1].plot(synchrony_map_outcome[2], synchrony_map_outcome[3], 'x', color=colors[2], markersize=ms, markeredgewidth=mw)

ax[2].plot(healthy_map_outcome[2], healthy_map_outcome[4], 'x', color=colors[0], markersize=ms, markeredgewidth=mw)
ax[2].plot(interictal_map_outcome[2], interictal_map_outcome[4], 'x', color=colors[1], markersize=ms, markeredgewidth=mw)
# ax[2].plot(synchrony_map_outcome[2], synchrony_map_outcome[4], 'x', color=colors[2], markersize=ms, markeredgewidth=mw)

ax[3].plot(healthy_map_outcome[5], healthy_map_outcome[6], 'x', color=colors[0], markersize=ms, markeredgewidth=mw)
ax[3].plot(interictal_map_outcome[5], interictal_map_outcome[6], 'x', color=colors[1], markersize=ms, markeredgewidth=mw)
# ax[3].plot(synchrony_map_outcome[5], synchrony_map_outcome[6], 'x', color=colors[2], markersize=ms, markeredgewidth=mw)



"""
for idx, x in enumerate(labels):
    plt.figure()
    plt.hist(data[1][active, idx], bins=100)
    plt.title(x)
"""



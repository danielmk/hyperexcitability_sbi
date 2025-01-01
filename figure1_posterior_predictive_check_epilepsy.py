# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import brian2 as b2
import torch
# import sbi
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import prepare_for_sbi
from sbi import utils as utils
import priors
from simulators_epilepsy import Simulator
import outcomes
import os
import pickle
from datetime import datetime
# import tables
import numpy as np
from itertools import chain
import sys
import thetas_vetted
from sbi.diagnostics.sbc import check_sbc, run_sbc, get_nltp
from sbi.analysis.plot import sbc_rank_plot
import sbi.analysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb

# results_dir = '/flash/FukaiU/danielmk/sbiemd/truncated_sequential_npe_restricted_network_baseline_net_one_simulator_constant_dynamics_short_fewer_pc_cython_one.pickle'
results_dir_healthy = r'C:\Users\Daniel\repos\sbiemd\data\truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_healthy_v2_01.pickle'

results_dir_interictal = r'C:\Users\Daniel\repos\sbiemd\data\truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_interictal_spiking_01.pickle'

results_dir_synchrony = r'C:\Users\Daniel\repos\sbiemd\data\truncated_sequential_npe_network_baseline_conductance_based_01_outcome_x_theta_synchrony_01.pickle'

# sys.exit()

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

labels = ['Mean Rate', 'Mean Entropy', 'Theta Power', ' Gamma Power', 'Fast Power', 'Correlation' , 'CV']

results_loaded = {}
conditions = []
x_o_list = []

for results_dir in [results_dir_healthy, results_dir_interictal]:

    condition = '_'.join(results_dir.split(os.path.sep)[-1].split('_')[-4:-1])

    x_o = eval(f"outcomes.{condition}['{condition}']")
    
    data = list(loadall(results_dir))

    inference_round = 3

    inference = data[inference_round][list(data[inference_round].keys())[0]]['inference']

    posterior = inference.build_posterior().set_default_x(x_o)

    dirname = os.path.dirname(__file__)

    prior_dict = priors.baseline_epilepsy

    simulator = Simulator(0.1 * b2.ms,
                    1.0 * b2.second,
                    prior_dict['constants'],
                    prior_dict)

    prior = priors.prior_dict_to_tensor(prior_dict)

    simulator, prior = prepare_for_sbi(simulator.run, prior)


    theta, x = simulate_for_sbi(simulator, posterior, num_simulations=1000, num_workers=4)

    results_loaded[condition] = x
    
    conditions.append(condition)
    
    x_o_list.append(x_o)

# DATAFRAME CREATION TODO
df_healthy = pd.DataFrame(results_loaded['x_healthy_v2'], columns=labels)
df_healthy['condition'] = 'Baseline'

df_spiking = pd.DataFrame(results_loaded['x_interictal_spiking'], columns=labels)
df_spiking['condition'] = 'Hyperexcitable'


df = pd.concat([df_healthy, df_spiking])

x_o_healthy = outcomes.x_healthy_v2['x_healthy_v2']

x_o_spiking = outcomes.x_interictal_spiking['x_interictal_spiking']

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

colors = ['#1b9e77', '#d95f02'] # , '#7570b3'

fig, ax = plt.subplots(1, len(labels))

for idx, l in enumerate(labels):
    order = ['Baseline', 'Hyperexcitable']
    sns.boxplot(x='condition', y=l, order=order, data=df, palette=colors, ax=ax[idx])
    ax[idx].plot(0, x_o_healthy[idx], marker='x', color='k', markersize='20')
    #ax[idx].plot(1, x_o_synchrony[idx], marker='x', color='k', markersize='20')
    ax[idx].plot(1, x_o_spiking[idx], marker='x', color='k', markersize='20')
    ax[idx].tick_params(axis='x', rotation=90)

"""
df = pd.DataFrame(x, columns=labels)

fig, ax = plt.subplots(1, len(x_o))
# Create boxplots
for i in range(x.shape[1]):
    ax[i].boxplot(x[:, i])
    ax[i].scatter([1], [x_o[i]], 400, marker='x')
    ax[i].set_xlabel(labels[i])
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import prepare_for_sbi
import priors
from simulators import Simulator
import outcomes
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from metadata import Metadata


results_dir_healthy = os.path.join(
    Metadata.results_dir,
    "truncated_sequential_npe_network_baseline_conductance_based_01_baseline.pickle")

results_dir_interictal = os.path.join(
    Metadata.results_dir,
    "truncated_sequential_npe_network_baseline_conductance_based_01_hyperexcitable.pickle")


def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


labels = Metadata.outcome_labels

simulation_results = {}
for results_dir in [results_dir_healthy, results_dir_interictal]:

    condition = results_dir.split(os.path.sep)[-1].split('_')[-1].split('.')[0]

    x_o = eval(f"outcomes.x_{condition}['{condition}']")

    data = list(loadall(results_dir))

    inference_round = 3

    inference = data[inference_round][list(data[inference_round].keys())[0]]['inference']

    posterior = inference.build_posterior().set_default_x(x_o)

    dirname = os.path.dirname(__file__)

    prior_dict = priors.baseline

    simulator = Simulator(Metadata.sim_dt,
                          Metadata.sim_duration,
                          prior_dict['constants'],
                          prior_dict)

    prior = priors.prior_dict_to_tensor(prior_dict)

    simulator, prior = prepare_for_sbi(simulator.run, prior)

    theta, x = simulate_for_sbi(simulator, posterior, num_simulations=1000, num_workers=4)

    simulation_results[condition] = x


# Create dataframe from simulation results
df_healthy = pd.DataFrame(simulation_results['baseline'], columns=labels)
df_healthy['condition'] = 'Baseline'

df_spiking = pd.DataFrame(simulation_results['hyperexcitable'], columns=labels)
df_spiking['condition'] = 'Hyperexcitable'

df = pd.concat([df_healthy, df_spiking])

x_o_healthy = outcomes.baseline['baseline']

x_o_spiking = outcomes.hyperexcitable['hyperexcitable']

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

colors = ['#1b9e77', '#d95f02']

fig, ax = plt.subplots(1, len(labels))

for idx, l in enumerate(labels):
    order = ['Baseline', 'Hyperexcitable']
    sns.boxplot(x='condition', y=l, order=order, data=df, palette=colors, ax=ax[idx])
    ax[idx].plot(0, x_o_healthy[idx], marker='x', color='k', markersize='20')
    ax[idx].plot(1, x_o_spiking[idx], marker='x', color='k', markersize='20')
    ax[idx].tick_params(axis='x', rotation=90)

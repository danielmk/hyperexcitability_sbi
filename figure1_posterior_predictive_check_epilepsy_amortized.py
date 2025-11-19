# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

from joblib import Parallel, delayed
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
import torch
import io
import sys
import numpy as np


results_dir_amortized = os.path.join(
    Metadata.results_dir,
    "amortized_inference_one.pickle")

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


labels = Metadata.outcome_labels

simulation_results = {}
for results_dir in ["truncated_sequential_npe_network_baseline_conductance_based_01_baseline.pickle",
                    "truncated_sequential_npe_network_baseline_conductance_based_01_hyperexcitable.pickle"]:

    condition = results_dir.split('_')[-1].split('.')[0]

    x_o = eval(f"outcomes.x_{condition}['{condition}']")

    #data = list(loadall(results_dir_amortized))
    with open(results_dir_amortized, "rb") as f:
        inference = CPU_Unpickler(f).load()

    posterior = inference.build_posterior(inference._neural_net.to("cpu")).set_default_x(x_o)

    dirname = os.path.dirname(__file__)

    prior_dict = priors.baseline

    simulator = Simulator(Metadata.sim_dt,
                          Metadata.sim_duration,
                          prior_dict['constants'],
                          prior_dict)

    prior = priors.prior_dict_to_tensor(prior_dict)

    # simulator, prior = prepare_for_sbi(simulator.run, prior)

    posterior_samples = posterior.sample((1000,))

    output = Parallel(n_jobs=8)(map(delayed(simulator.run), posterior_samples))

    # theta, x = simulate_for_sbi(simulator.run, posterior, num_simulations=1000, num_workers=4)

    simulation_results[condition] = output

simulation_results['baseline'] = np.array(simulation_results['baseline'])

simulation_results['hyperexcitable'] = np.array(simulation_results['hyperexcitable'])

# Create dataframe from simulation results
df_healthy = pd.DataFrame(simulation_results['baseline'], columns=labels)
df_healthy['condition'] = 'Baseline'

df_spiking = pd.DataFrame(simulation_results['hyperexcitable'], columns=labels)
df_spiking['condition'] = 'Hyperexcitable'

df = pd.concat([df_healthy, df_spiking])

x_o_healthy = outcomes.x_baseline['baseline']

x_o_spiking = outcomes.x_hyperexcitable['hyperexcitable']

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
    if 'Power' in l:
        ax[idx].set_yscale('log')

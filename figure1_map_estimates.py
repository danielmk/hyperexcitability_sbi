# -*- coding: utf-8 -*-
"""
Run simulations with the MAP estimates of the baseline and hyperexcitable posteriors and plots the output.
"""

import brian2 as b2
import matplotlib.pyplot as plt
import thetas_vetted
import simulators
import priors
from matplotlib.gridspec import GridSpec
from metadata import Metadata

b2.seed(106)

b2.prefs.codegen.target = 'cython'

prior = priors.baseline

simulator = simulators.Simulator(dt=Metadata.sim_dt,
                                 duration=Metadata.sim_duration,
                                 network_constants=prior['constants'],
                                 prior=prior)

theta_baseline = thetas_vetted.baseline_map

theta_tensor_baseline_map = thetas_vetted.theta_dict_to_tensor(theta_baseline)

theta_interical_spiking_map = thetas_vetted.hyperexcitable_map

theta_tensor_interical_spiking_map = thetas_vetted.theta_dict_to_tensor(theta_interical_spiking_map)

statistics_baseline, nw_baseline = simulator.run_evaluate(theta_tensor_baseline_map)

statistics_hyperexcitable, nw_hyperexcitable = simulator.run_evaluate(theta_tensor_interical_spiking_map)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})
colors = ['#1b9e77', '#d95f02', '#7570b3']

n_cells = 50

baseline_pc_spiketrains = list((nw_baseline['pc_spikes'].spike_trains().values()))
hyperexcitable_pc_spiketrains = list((nw_hyperexcitable['pc_spikes'].spike_trains().values()))

fig = plt.figure(layout="constrained")
gs = GridSpec(6, 2, figure=fig)
ax_volt_baseline = fig.add_subplot(gs[0, 0])
ax_volt_hyperexcitable = fig.add_subplot(gs[0, 1])


ax_spikes_baseline = fig.add_subplot(gs[1:, 0])
ax_spikes_hyperexcitable = fig.add_subplot(gs[1:, 1])


ax_volt_baseline.plot(nw_baseline['pc_vs'].t, nw_baseline['pc_vs'].vs[0], color=colors[0])
ax_volt_hyperexcitable.plot(nw_hyperexcitable['pc_vs'].t, nw_hyperexcitable['pc_vs'].vs[0], color=colors[1])

ax_spikes_baseline.eventplot(baseline_pc_spiketrains[:n_cells], color=colors[0])
ax_spikes_hyperexcitable.eventplot(hyperexcitable_pc_spiketrains[:n_cells], color=colors[1])

ax_volt_baseline.set_ylabel("Voltage (V)")
ax_spikes_baseline.set_ylabel("# Neuron")

for a in [ax_volt_baseline, ax_volt_hyperexcitable, ax_spikes_baseline, ax_spikes_hyperexcitable]:
    a.set_xlim((0, 1.0))

for a in [ax_spikes_baseline, ax_spikes_hyperexcitable]:
    a.set_xlabel("Time (s)")
    a.set_ylim((0, n_cells))

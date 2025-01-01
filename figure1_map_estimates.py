# -*- coding: utf-8 -*-
"""
This script illustrates the network properties by simulating it with handpicked
parameters defined in parameters.py

It uses network.py to create the network and summary_statistics.py to calculate
informative statistics of the baseline network.


"""

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import thetas_vetted
import summary_statistics
import simulators_epilepsy
import sys
import priors
from matplotlib.gridspec import GridSpec

b2.seed(106)

b2.prefs.codegen.target = 'cython'

prior = priors.baseline_epilepsy

duration = 1.0*b2.second
_dt = 0.1*b2.ms  # Needs underscore not to conflict with b2 internal dt

simulator = simulators_epilepsy.Simulator(dt=_dt,
                                duration=duration,
                                network_constants=prior['constants'],
                                prior=prior)

theta_healthy = thetas_vetted.healthy_v2_baseline_map

theta_tensor_healthy_map = thetas_vetted.theta_dict_to_tensor(theta_healthy)

theta_interical_spiking_map = thetas_vetted.interical_spiking_map

theta_tensor_interical_spiking_map = thetas_vetted.theta_dict_to_tensor(theta_interical_spiking_map)

theta_theta_synchrony_map = thetas_vetted.theta_synchrony_map

theta_tensor_theta_synchrony_map = thetas_vetted.theta_dict_to_tensor(theta_theta_synchrony_map)

statistics_healthy, nw_healthy = simulator.run_evaluate(theta_tensor_healthy_map)

statistics_interictal, nw_interictal = simulator.run_evaluate(theta_tensor_interical_spiking_map)

statistics_synchrony, nw_synchrony = simulator.run_evaluate(theta_tensor_theta_synchrony_map)


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})
colors = ['#1b9e77', '#d95f02', '#7570b3']

fig, ax = plt.subplots(6, 3)

healthy_pc_spiketrains = list((nw_healthy['pc_spikes'].spike_trains().values()))
healthy_nin_spiketrains = list((nw_healthy['nin_spikes'].spike_trains().values()))
healthy_ain_spiketrains = list((nw_healthy['ain_spikes'].spike_trains().values()))

interictal_pc_spiketrains = list((nw_interictal['pc_spikes'].spike_trains().values()))
interictal_nin_spiketrains = list((nw_interictal['nin_spikes'].spike_trains().values()))
interictal_ain_spiketrains = list((nw_interictal['ain_spikes'].spike_trains().values()))

synchrony_pc_spiketrains = list((nw_synchrony['pc_spikes'].spike_trains().values()))
synchrony_nin_spiketrains = list((nw_synchrony['nin_spikes'].spike_trains().values()))
synchrony_ain_spiketrains = list((nw_synchrony['ain_spikes'].spike_trains().values()))

ax[0,0].plot(nw_healthy['pc_vs'].t, nw_healthy['pc_vs'].vs[0])
ax[0,1].plot(nw_healthy['nin_vs'].t, nw_healthy['nin_vs'].vs[0])
ax[0,2].plot(nw_healthy['ain_vs'].t, nw_healthy['ain_vs'].vs[0])

ax[1,0].eventplot(healthy_pc_spiketrains[:10])
ax[1,1].eventplot(healthy_nin_spiketrains[:10])
ax[1,2].eventplot(healthy_ain_spiketrains[:10])

ax[2,0].plot(nw_interictal['pc_vs'].t, nw_interictal['pc_vs'].vs[0])
ax[2,1].plot(nw_interictal['nin_vs'].t, nw_interictal['nin_vs'].vs[0])
ax[2,2].plot(nw_interictal['ain_vs'].t, nw_interictal['ain_vs'].vs[0])

ax[3,0].eventplot(interictal_pc_spiketrains[:10])
ax[3,1].eventplot(interictal_nin_spiketrains[:10])
ax[3,2].eventplot(interictal_ain_spiketrains[:10])

ax[4,0].plot(nw_synchrony['pc_vs'].t, nw_synchrony['pc_vs'].vs[0])
ax[4,1].plot(nw_synchrony['nin_vs'].t, nw_synchrony['nin_vs'].vs[0])
ax[4,2].plot(nw_synchrony['ain_vs'].t, nw_synchrony['ain_vs'].vs[0])

ax[5,0].eventplot(synchrony_pc_spiketrains[:10])
ax[5,1].eventplot(synchrony_nin_spiketrains[:10])
ax[5,2].eventplot(synchrony_ain_spiketrains[:10])


fig, ax = plt.subplots(2, 3)

healthy_pc_spiketrains = list((nw_healthy['pc_spikes'].spike_trains().values()))
healthy_nin_spiketrains = list((nw_healthy['nin_spikes'].spike_trains().values()))
healthy_ain_spiketrains = list((nw_healthy['ain_spikes'].spike_trains().values()))

interictal_pc_spiketrains = list((nw_interictal['pc_spikes'].spike_trains().values()))
interictal_nin_spiketrains = list((nw_interictal['nin_spikes'].spike_trains().values()))
interictal_ain_spiketrains = list((nw_interictal['ain_spikes'].spike_trains().values()))

synchrony_pc_spiketrains = list((nw_synchrony['pc_spikes'].spike_trains().values()))
synchrony_nin_spiketrains = list((nw_synchrony['nin_spikes'].spike_trains().values()))
synchrony_ain_spiketrains = list((nw_synchrony['ain_spikes'].spike_trains().values()))

ax[0,0].plot(nw_healthy['pc_vs'].t, nw_healthy['pc_vs'].vs[0], color=colors[0])
ax[0,2].plot(nw_interictal['pc_vs'].t, nw_interictal['pc_vs'].vs[0], color=colors[1])
ax[0,1].plot(nw_synchrony['pc_vs'].t, nw_synchrony['pc_vs'].vs[0], color=colors[2])

ax[1,0].eventplot(healthy_pc_spiketrains[:500], color=colors[0], rasterized=True)
ax[1,2].eventplot(interictal_pc_spiketrains[:500], color=colors[1], rasterized=True)
ax[1,1].eventplot(synchrony_pc_spiketrains[:500], color=colors[2], rasterized=True)

ax[0,0].set_ylabel("Voltage (V)")

ax[1,0].set_ylabel("# Neuron")

for a in ax.flatten():
    a.set_xlim(0, 1.0)

for a in ax[1,:]:
    a.set_xlabel("Time (s)")

ax[0,0].set_title("Healthy")
ax[0,2].set_title("Interictal Spiking")
ax[0,1].set_title("Theta Synchrony")

n_cells = 50

fig = plt.figure(layout="constrained")
gs = GridSpec(6, 2, figure=fig)
ax_volt_healthy = fig.add_subplot(gs[0, 0])
ax_volt_interictal = fig.add_subplot(gs[0, 1])
# ax_volt_synchrony = fig.add_subplot(gs[0, 1])

ax_spikes_healthy = fig.add_subplot(gs[1:, 0])
ax_spikes_interictal = fig.add_subplot(gs[1:, 1])
#ax_spikes_synchrony = fig.add_subplot(gs[1:, 1])

ax_volt_healthy.plot(nw_healthy['pc_vs'].t, nw_healthy['pc_vs'].vs[0], color=colors[0])
ax_volt_interictal.plot(nw_interictal['pc_vs'].t, nw_interictal['pc_vs'].vs[0], color=colors[1])
#ax_volt_synchrony.plot(nw_synchrony['pc_vs'].t, nw_synchrony['pc_vs'].vs[0], color=colors[2])

ax_spikes_healthy.eventplot(healthy_pc_spiketrains[:n_cells], color=colors[0])
ax_spikes_interictal.eventplot(interictal_pc_spiketrains[:n_cells], color=colors[1])
#ax_spikes_synchrony.eventplot(synchrony_pc_spiketrains[:500], color=colors[2], rasterized=True)

ax_volt_healthy.set_ylabel("Voltage (V)")
ax_spikes_healthy.set_ylabel("# Neuron")

for a in [ax_volt_healthy, ax_volt_interictal, ax_spikes_healthy, ax_spikes_interictal]:
    a.set_xlim((0, 1.0))
    
for a in [ax_spikes_healthy, ax_spikes_interictal]:
    a.set_xlabel("Time (s)")
    a.set_ylim((0, n_cells))

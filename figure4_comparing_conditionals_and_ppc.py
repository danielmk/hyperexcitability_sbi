# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import torch
import os
import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from metadata import Metadata
import conditionals

n_samples = 100000

np.random.seed(321)

torch.manual_seed(45234567)

labels = Metadata.parameter_labels

col_labels = Metadata.outcome_labels

output_path = os.path.join(Metadata.results_dir, 'conditionals_output_data_healthy_vs_hyperexcitable.h5')

data_file = tables.open_file(output_path, mode='r')

x_in_loss_healthy = data_file.root.in_loss.x_healthy.read()

theta_in_loss_healthy = data_file.root.in_loss.theta_baseline.read()

x_in_loss_in_loss = data_file.root.in_loss.x_hyperexcitable.read()

x_intrinsics_healthy = data_file.root.intrinsics.x_healthy.read()

x_intrinsics_hyperexcitable = data_file.root.intrinsics.x_hyperexcitable.read()

x_sprouting_healthy = data_file.root.sprouting_only_v3.x_healthy.read()

x_sprouting_hyperexcitable = data_file.root.sprouting_only_v3.x_hyperexcitable.read()

df_x_in_loss_healthy = pd.DataFrame(x_in_loss_healthy, columns=col_labels)
df_x_in_loss_healthy['conditional'] = 'IN Loss'
df_x_in_loss_healthy['condition'] = 'Baseline'

df_x_in_loss_in_loss = pd.DataFrame(x_in_loss_in_loss, columns=col_labels)
df_x_in_loss_in_loss['conditional'] = 'IN Loss'
df_x_in_loss_in_loss['condition'] = 'Hyperexcitable'

df_x_intrinsics_healthy = pd.DataFrame(x_intrinsics_healthy, columns=col_labels)
df_x_intrinsics_healthy['conditional'] = 'Intrinsics'
df_x_intrinsics_healthy['condition'] = 'Baseline'

df_x_intrinsics_hyperexcitable = pd.DataFrame(x_intrinsics_hyperexcitable, columns=col_labels)
df_x_intrinsics_hyperexcitable['conditional'] = 'Intrinsics'
df_x_intrinsics_hyperexcitable['condition'] = 'Hyperexcitable'

df_x_sprouting_healthy = pd.DataFrame(x_sprouting_healthy, columns=col_labels)
df_x_sprouting_healthy['conditional'] = 'Sprouting'
df_x_sprouting_healthy['condition'] = 'Baseline'

df_x_sprouting_hyperexcitable = pd.DataFrame(x_sprouting_hyperexcitable, columns=col_labels)
df_x_sprouting_hyperexcitable['conditional'] = 'Sprouting'
df_x_sprouting_hyperexcitable['condition'] = 'Hyperexcitable'

df = pd.concat([df_x_in_loss_healthy,
                df_x_in_loss_in_loss,
                df_x_intrinsics_healthy,
                df_x_intrinsics_hyperexcitable,
                df_x_sprouting_healthy,
                df_x_sprouting_hyperexcitable])

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(1, len(col_labels))

colors = ["#1f78b4", "#b2df8a"]

for idx, l in enumerate(col_labels):
    order = ['IN Loss', 'Intrinsics', 'Sprouting']
    sns.boxplot(x='conditional', y=l, hue='condition', order=order, data=df, palette=colors, ax=ax[idx])
    # ax[idx].plot(0, x_o_healthy[idx], marker='x', color='k', markersize='20')
    # ax[idx].plot(1, x_o_synchrony[idx], marker='x', color='k', markersize='20')
    # ax[idx].plot(2, x_o_spiking[idx], marker='x', color='k', markersize='20')
    ax[idx].tick_params(axis='x', rotation=90)

ks_test_in_loss = data_file.root.in_loss.ks_test.read()

ks_test_in_loss = np.insert(ks_test_in_loss, 4, [np.nan, np.nan], axis=0)
ks_test_in_loss = np.insert(ks_test_in_loss, 9, [np.nan, np.nan], axis=0)

ks_test_hyperexcitable = data_file.root.intrinsics.ks_test.read()

ks_test_hyperexcitable = np.insert(ks_test_hyperexcitable, 1, [np.nan, np.nan], axis=0)
ks_test_hyperexcitable = np.insert(ks_test_hyperexcitable, 2, [np.nan, np.nan], axis=0)

ks_test_sprouting = data_file.root.sprouting_only_v3.ks_test.read()

ks_test_sprouting = np.insert(ks_test_sprouting, 14, [np.nan, np.nan], axis=0)

colors = ['#66c2a5', '#fc8d62', '#8da0cb']
y = np.arange(ks_test_in_loss.shape[0])

fig, ax = plt.subplots(1)
y = np.arange(ks_test_in_loss.shape[0])
ls = 'solid'
alpha = 1
ax.hlines(np.arange(len(labels)),
          0,
          np.nanmax([ks_test_sprouting[:, 0], ks_test_hyperexcitable[:, 0], ks_test_hyperexcitable[:, 0]], axis=0),
          color='grey',
          alpha=alpha,
          linestyle=ls)
ax.plot(ks_test_in_loss[:, 0], y, marker='o', linestyle='None', markersize=20, color=colors[0], alpha=0.8)
ax.plot(ks_test_sprouting[:, 0], y, marker='d', linestyle='None', markersize=20, color=colors[1], alpha=0.8)
ax.plot(ks_test_hyperexcitable[:, 0], y, marker='s', linestyle='None', markersize=20, color=colors[2], alpha=0.8)
ax.set_yticks(np.arange(len(labels)))
# ... and label them with the respective list entries
ax.set_yticklabels(labels)
# plt.setp(ax[0].get_yticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor")
ax.set_xlabel("KS Test Statistic")
ax.legend(("", "IN Loss", "Sprouting", "Intrinsic"))
ax.set_xlim((0, 1.1))


"""MAKE THE HISTOGRAM COMPARISON PLOT"""
colors = ["#1f78b4", "#b2df8a"]
ks_test_in_loss = data_file.root.in_loss.ks_test.read()
ks_test_hyperexcitable = data_file.root.intrinsics.ks_test.read()
ks_test_sprouting = np.array(data_file.root.sprouting_only_v3.ks_test.read())

theta_in_loss_healthy = data_file.root.in_loss.theta_baseline.read()

theta_in_loss_in_loss = data_file.root.in_loss.theta_hyperexcitable.read()

theta_hyperexcitable_healthy = data_file.root.intrinsics.theta_baseline.read()

theta_hyperexcitable_hyperexcitable = data_file.root.intrinsics.theta_hyperexcitable.read()

theta_sprouting_healthy = data_file.root.sprouting_only_v3.theta_baseline.read()

theta_sprouting_sprouting = data_file.root.sprouting_only_v3.theta_hyperexcitable.read()

in_conditional = conditionals.in_loss_conditional
in_loss_labels = in_conditional.unconditioned_labels

sprouting_conditional = conditionals.synapse_only_sprouting_conditional
sprouting_labels = sprouting_conditional.unconditioned_labels

intrinsics_conditional = conditionals.intrinsics_depolarized_conditional
excitable_labels = intrinsics_conditional.unconditioned_labels

largest_statistic_in_loss = np.argsort(ks_test_in_loss[:, 0])[::-1]
largest_statistic_in_loss = largest_statistic_in_loss[:5]

largest_labels_in_loss = np.array(in_loss_labels)[largest_statistic_in_loss]
largest_parameters_in_loss_healthy = np.array(theta_in_loss_healthy)[:, largest_statistic_in_loss]
largest_parameters_in_loss_unhealthy = np.array(theta_in_loss_in_loss)[:, largest_statistic_in_loss]

largest_statistic_sprouting = np.argsort(ks_test_sprouting[:, 0])[::-1]
largest_statistic_sprouting = largest_statistic_sprouting[:5]

largest_labels_sprouting = np.array(sprouting_labels)[largest_statistic_sprouting]
largest_parameters_sprouting_healthy = np.array(theta_sprouting_healthy)[:, largest_statistic_sprouting]
largest_parameters_sprouting_unhealthy = np.array(theta_sprouting_sprouting)[:, largest_statistic_sprouting]

largest_statistic_hyperexcitable = np.argsort(ks_test_hyperexcitable[:, 0])[::-1]
largest_statistic_hyperexcitable = largest_statistic_hyperexcitable[:5]

largest_labels_hyperexcitable = np.array(excitable_labels)[largest_statistic_hyperexcitable]
largest_parameters_hyperexcitable_healthy = np.array(theta_hyperexcitable_healthy)[:, largest_statistic_hyperexcitable]
largest_parameters_hyperexcitable_unhealthy = np.array(theta_hyperexcitable_hyperexcitable)[:, largest_statistic_hyperexcitable]

fig, ax = plt.subplots(3, 5)
bins = 100
for idx, param in enumerate(largest_statistic_in_loss):
    ax[0, idx].hist(largest_parameters_in_loss_healthy[:, idx], bins=bins, color=colors[0], histtype=u'step')
    ax[0, idx].hist(largest_parameters_in_loss_unhealthy[:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[0, idx].set_xlabel(in_loss_labels[param])
ax[0, 0].legend(("Baseline", "Hyperexcitable"))

for idx, param in enumerate(largest_statistic_sprouting):
    ax[1, idx].hist(largest_parameters_sprouting_healthy[:, idx], bins=bins, color=colors[0], histtype=u'step')
    ax[1, idx].hist(largest_parameters_sprouting_unhealthy[:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[1, idx].set_xlabel(sprouting_labels[param])
ax[1, 0].legend(("Baseline", "Hyperexcitable"))

for idx, param in enumerate(largest_statistic_hyperexcitable):
    ax[2, idx].hist(largest_parameters_hyperexcitable_healthy[:, idx], bins=bins, color=colors[0], histtype=u'step')
    ax[2, idx].hist(largest_parameters_hyperexcitable_unhealthy[:, idx], bins=bins, color=colors[1], histtype=u'step')
    ax[2, idx].set_xlabel(excitable_labels[param])
ax[2, 0].legend(("Baseline", "Hyperexcitable"))

"""SCATTERPLOTS OF STATISTICS"""
baseline_vs_hyperexcitable_data_file = data_file

correlations_path = os.path.join(Metadata.results_dir, 'marginal_and_conditional_correlation_matrices.h5')

correlations_data_file = tables.open_file(correlations_path, mode='r')

healthy_vs_healthy_path = os.path.join(Metadata.results_dir, 'conditionals_output_data.h5')

baseline_data_file = tables.open_file(healthy_vs_healthy_path, mode='r')

ks_test_in_loss_bl = data_file.root.in_loss.ks_test.read()

df_in_loss_bl = pd.DataFrame(ks_test_in_loss_bl, columns=['Statistic IN Loss BL', "p-value IN Loss"], index=conditionals.in_loss_conditional.unconditioned_labels)

ks_test_hyperexcitable_bl = data_file.root.intrinsics.ks_test.read()

df_hyperexctiabl_bl = pd.DataFrame(ks_test_hyperexcitable_bl, columns=['Statistic Hyperexctiable BL', "p-value Hyperexctiable"], index=conditionals.intrinsics_depolarized_conditional.unconditioned_labels)

ks_test_sprouting_bl = data_file.root.sprouting_only_v3.ks_test.read()

df_sprouting_bl = pd.DataFrame(ks_test_sprouting_bl, columns=['Statistic Sprouting BL', "p-value Sprouting"], index=conditionals.synapse_only_sprouting_conditional.unconditioned_labels)

df_baseline = pd.concat([df_in_loss_bl, df_hyperexctiabl_bl, df_sprouting_bl], axis=1, join='outer')

# ks_test_sprouting_bl = np.insert(ks_test_sprouting_bl, 14, [np.nan, np.nan], axis=0)

# CONSTRUCt THE IN LOSS DATA FRAME HERE
statistic_bl = baseline_data_file.root.in_loss.ks_test.read()[:, 0]

statistic_bl_vs_hyp = baseline_vs_hyperexcitable_data_file.root.in_loss.ks_test.read()[:, 0]

marginal_correlations = correlations_data_file.root.marginal_correlations.read()
abs_marginal_correlations = (np.abs(marginal_correlations[4]) + np.abs(marginal_correlations[9]))/2
abs_marginal_correlations = np.delete(abs_marginal_correlations, [4,9])

conditional_correlations = correlations_data_file.root.conditional_correlations.read()
abs_conditional_correlations = (np.abs(conditional_correlations[4]) + np.abs(conditional_correlations[9]))/2
abs_conditional_correlations = np.delete(abs_conditional_correlations, [4,9])

df_in_loss = df2 = pd.DataFrame(np.array([statistic_bl, statistic_bl_vs_hyp, abs_marginal_correlations, abs_conditional_correlations]).T,
                   columns=['KS Statistic BL', 'KS Statistic BL vs Hyp', 'Absolute Marginal Correlation', 'Absolute Conditional Correlation'])

df_in_loss['condition'] = "IN Loss"

#CONSTRUCT SPROUTING DF HERE
statistic_bl = baseline_data_file.root.sprouting_only_v4.ks_test.read()[:, 0]

statistic_bl_vs_hyp = np.array(baseline_vs_hyperexcitable_data_file.root.sprouting_only_v3.ks_test.read())[:, 0]

marginal_correlations = correlations_data_file.root.marginal_correlations.read()
abs_marginal_correlations = np.abs(marginal_correlations[14])
abs_marginal_correlations = np.delete(abs_marginal_correlations, [14])

conditional_correlations = correlations_data_file.root.conditional_correlations.read()
abs_conditional_correlations = np.abs(conditional_correlations[14])
abs_conditional_correlations = np.delete(abs_conditional_correlations, [14])

df_sprouting = df2 = pd.DataFrame(np.array([statistic_bl, statistic_bl_vs_hyp, abs_marginal_correlations, abs_conditional_correlations]).T,
                   columns=['KS Statistic BL', 'KS Statistic BL vs Hyp', 'Absolute Marginal Correlation', 'Absolute Conditional Correlation'])
df_sprouting['condition'] = "Sprouting"


#CONSTRUCT SPROUTING DF HERE
statistic_bl = baseline_data_file.root.hyperexcitable_v4.ks_test.read()[:, 0]

statistic_bl_vs_hyp = np.array(baseline_vs_hyperexcitable_data_file.root.intrinsics.ks_test.read())[:, 0]

marginal_correlations = correlations_data_file.root.marginal_correlations.read()
abs_marginal_correlations = (np.abs(marginal_correlations[1]) + np.abs(marginal_correlations[2]))/2
abs_marginal_correlations = np.delete(abs_marginal_correlations, [1, 2])

conditional_correlations = correlations_data_file.root.conditional_correlations.read()
abs_conditional_correlations = (np.abs(conditional_correlations[1]) + np.abs(conditional_correlations[2]))/2
abs_conditional_correlations = np.delete(abs_conditional_correlations, [1, 2])

df_intrinsics = df2 = pd.DataFrame(np.array([statistic_bl, statistic_bl_vs_hyp, abs_marginal_correlations, abs_conditional_correlations]).T,
                   columns=['KS Statistic BL', 'KS Statistic BL vs Hyp', 'Absolute Marginal Correlation', 'Absolute Conditional Correlation'])
df_intrinsics['condition'] = "Intrinsic"

df_full = pd.concat([df_in_loss, df_sprouting, df_intrinsics])

colors = ['#66c2a5', '#fc8d62', '#8da0cb']
sns.scatterplot(data=df_full, x='KS Statistic BL', y='Absolute Marginal Correlation', hue='condition', palette=colors)

sns.scatterplot(data=df_full, x='KS Statistic BL', y='Absolute Conditional Correlation', hue='condition', palette=colors)

sns.scatterplot(data=df_full, x='KS Statistic BL', y='KS Statistic BL vs Hyp', hue='condition', palette=colors)

sns.scatterplot(data=df_full, x='Absolute Marginal Correlation', y='Absolute Conditional Correlation', hue='condition', palette=colors)

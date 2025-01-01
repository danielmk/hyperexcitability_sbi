# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:13:20 2024

@author: Daniel
"""

import brian2 as b2
import torch
#import sbi
# from sbi.inference import SNPE, simulate_for_sbi
#import priors
#from simulators_epilepsy import Simulator
import os
import pickle
import tables
import numpy as np
import sys
import matplotlib.pyplot as plt
import platform
#from sbi.utils.user_input_checks import prepare_for_sbi
from copy import deepcopy
from scipy.stats import ks_2samp
#from sbi.analysis import pairplot, conditional_potential
from metadata import EpilepsyMetadata
import conditionals
import pdb
import pandas as pd
pd.options.mode.copy_on_write = True
import seaborn as sns
from sklearn import preprocessing
from sklearn.manifold import TSNE
import statsmodels.api as sm
from statsmodels.formula.api import ols


np.random.seed(321)

torch.manual_seed(45234567)

output_path = os.path.join(EpilepsyMetadata.results_dir, 'sensitivity_analysis_hyperexcitable.h5')

data_file = tables.open_file(output_path, mode='r')

sensitivity_parameters_labels = (
    r'$PC_C$',
    r'$PC_{g_L}$',
    r'$PC_{E_L}$',
    r'$PC_{V_T}$',
    r'$AIN_N$',
    r'$NIN_N$',
    r'$PC-PC_P$',
    r'$PC-PC_{A_{SE}}$',
    r'$NIN-PC_P$',
    r'$NIN-PC_{A_{SE}}$',
    r'$AIN-PC_P$',
    )

data_file.root.in_loss.conditional = conditionals.in_loss_conditional

data_file.root.sprouting.conditional = conditionals.synapse_only_sprouting_conditional

data_file.root.intrinsics.conditional = conditionals.intrinsics_depolarized_conditional

"""DATAFRAME CREATION"""
for cond in data_file.root:
    cond_sensitivity_labels = cond.sensitivity_labels.read()
    
    thetas_sh = cond.thetas.shape
    output_sh = cond.output_x.shape
    
    thetas_reshaped = cond.thetas.read().reshape((thetas_sh[0]*thetas_sh[1], thetas_sh[2]))
    output_reshaped = cond.output_x.read().reshape((output_sh[0]*output_sh[1], output_sh[2]))
    np.nan_to_num(output_reshaped, copy=False, nan=0.0)


    theta_idc_unique = np.arange(thetas_sh[0])
    out_idc_unique = np.arange(output_sh[0])
    
    thetas_sensitivity_idc = np.repeat(theta_idc_unique, thetas_sh[1])
    output_sensitivity_idc = np.repeat(out_idc_unique, output_sh[1])
    
    thetas_mapping = {x: cond_sensitivity_labels[x] for x in theta_idc_unique}
    output_mapping = {x: cond_sensitivity_labels[x] for x in out_idc_unique}
    
    df_thetas = pd.DataFrame(thetas_reshaped, columns=EpilepsyMetadata.parameter_labels)
    df_thetas['sensitivity_label'] = thetas_sensitivity_idc
    df_thetas.replace({'sensitivity_label': thetas_mapping}, inplace=True)
    
    df_output = pd.DataFrame(output_reshaped, columns=EpilepsyMetadata.outcome_labels)
    
    df = pd.concat([df_thetas, df_output], axis=1)
    
    df['condition'] = cond.conditional.name
    
    cond.df_final = df

df = pd.concat([data_file.root.in_loss.df_final, data_file.root.sprouting.df_final, data_file.root.intrinsics.df_final], axis=0, ignore_index=True)

df['sensitivity_label'] = df.sensitivity_label.astype(str)

"""PLOTTING"""
colors = ['#66c2a5', '#fc8d62', '#8da0cb']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 16})

parameters = (
    '$PC_C$',
    '$PC_{g_L}$',
    '$PC_{E_L}$',
    '$PC_{V_T}$',
    '$PC-PC_P$',
    '$PC-PC_{A_{SE}}$',
    '$NIN-PC_P$',
    '$AIN-PC_P$')

fig, ax = plt.subplots(ncols=len(parameters), nrows=len(EpilepsyMetadata.outcome_labels), figsize=(8.27, 11.69), dpi=100)

for ip, parameter in enumerate(parameters):
    for io, outcome in enumerate(EpilepsyMetadata.outcome_labels):
        # parameter = ['$PC_{V_T}$']
        sns.lineplot(x=parameter,
                     y=outcome,
                     hue='condition',
                     hue_order=df['condition'].unique(),
                     data=df.loc[df.sensitivity_label.isin([parameter])],
                     ax=ax[io, ip], palette=['#66c2a5', '#fc8d62', '#8da0cb'])
        
for a in ax.flatten()[1:]:
    a.legend("", frameon=False)
    # a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
for a in ax[:-1,:].flatten():
    a.set_xticks(ticks=[], labels=[])
    a.set_xlabel(None)
    
for a in ax[2,:]:
    a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
for a in ax[2:5,:].flatten():
    a_lim = a.get_ylim()
    distance = np.abs(a_lim[1] / a_lim[0])
    if distance >20:
        a.set_yscale('log')
    else:
        a.set_yscale('linear')

"""CALCULATE STATISTICS"""
parameters_stats = []
for p in parameters:
    new_p = p.replace('$', '').replace('{', '').replace('}', '').replace('-', '')
    df[new_p] = df[p]
    parameters_stats.append(new_p)
    
outcome_stats = []
for o in EpilepsyMetadata.outcome_labels:
    new_o = o.replace(' ', '')
    df[new_o] = df[o]
    outcome_stats.append(new_o)
    
stats = []
for ip, parameter in enumerate(parameters_stats):
    for io, outcome in enumerate(outcome_stats):
        moore_lm = ols(f'{outcome} ~ C(condition) * C({parameter})',
                       data=df.loc[df.sensitivity_label.isin([parameters[ip]])]).fit()
        table = sm.stats.anova_lm(moore_lm, typ=2)
        curr_series = table.loc[f"C(condition):C({parameter})"]
        curr_series.loc['parameter'] = parameter
        curr_series.loc['outcome'] = outcome
        stats.append(curr_series)

df_stats = pd.DataFrame(stats)
df_stats_pivoted = df_stats.pivot_table(values=['sum_sq', 'df', 'F', 'PR(>F)'], index=df_stats.index, columns='outcome')

sys.exit()

"""CALCULATE OUTCOME TSNE"""
X = df[list(EpilepsyMetadata.outcome_labels)]
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

tsne = TSNE(n_components=2)

tsne_result = tsne.fit_transform(X_scaled)

df['tsne1'] = tsne_result[:,0]

df['tsne2'] = tsne_result[:,1]

fig, ax = plt.subplots(ncols=len(sensitivity_parameters_labels), nrows=len(df['condition'].unique()))

for ip, parameter in enumerate(sensitivity_parameters_labels):
    for io, condition in enumerate(df['condition'].unique()):
        sns.scatterplot(x='tsne1',
                     y='tsne2',
                     hue=parameter,
                     data=df.loc[df.sensitivity_label.isin([parameter]) & df.condition.isin([condition])],
                     ax=ax[io, ip])

        
for a in ax.flatten()[1:]:
    a.legend("", frameon=False)
    # a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    

sys.exit()


"""SCATTERPLOT"""
label = b'$PC_{V_T}$'
for cond in data_file.root:
    string_dtype = cond.sensitivity_labels.read().dtype
    index_sensitivity = np.argwhere(cond.sensitivity_labels.read() == label)
    unconditioned_labels_array = np.array(EpilepsyMetadata.parameter_labels, dtype = string_dtype)
    index_unconditioned = np.argwhere(unconditioned_labels_array == label)
    plt.scatter(cond.thetas.read()[index_sensitivity,:,index_unconditioned], cond.output_x.read()[index_sensitivity,:,0])
    # print(np.unique(cond.thetas.read()[index_sensitivity,:,index_unconditioned]))
    print(cond, index_sensitivity, index_unconditioned)


"""MEAN PLOT"""
label = b'$PC_{V_T}$'
for cond in data_file.root:
    string_dtype = cond.sensitivity_labels.read().dtype
    index_sensitivity = np.argwhere(cond.sensitivity_labels.read() == label)
    unconditioned_labels_array = np.array(EpilepsyMetadata.parameter_labels, dtype = string_dtype)
    index_unconditioned = np.argwhere(unconditioned_labels_array == label)
    plt.scatter(cond.thetas.read().mean(axis=1)[index_sensitivity,index_unconditioned], cond.output_x.read().mean(axis=1)[index_sensitivity,0])
    # print(np.unique(cond.thetas.read()[index_sensitivity,:,index_unconditioned]))
    print(cond, index_sensitivity, index_unconditioned)


sys.exit()

prior_dict = priors.baseline_epilepsy

prior = priors.prior_dict_to_tensor(prior_dict)

simulator = Simulator(
    EpilepsyMetadata.sim_dt,
    EpilepsyMetadata.sim_duration,
    prior_dict['constants'],
    prior_dict)

simulator, prior = prepare_for_sbi(simulator.run, prior)

# sys.exit()

"""SET THE CONDITION HERE"""
# CONDITION IS SPECIFIED HERE!
conditional = conditionals.in_loss_conditional

all_thetas = data_file.root.in_loss.thetas.read()

output_collection = []
for curr_thetas in all_thetas:
    #theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations, num_workers=4)
    curr_input = conditional.make_theta(curr_thetas)
    curr_x = simulator(curr_input, num_workers=4)
    output_collection.append(curr_x)

output_path = os.path.join(EpilepsyMetadata.results_dir, 'conditionals_output_data_healthy_vs_hyperexcitable.h5')

condition='in_loss'
with tables.open_file(output_path, mode='a') as output:
    output.create_array(f'/{condition}', 'output_x', obj=np.array(output_collection))




"""PLOT SCATTERPLOTS OF ALL SIGNIFICANT INTERACTIONS"""


"""
fig, ax = plt.subplots(1, 5)
ax = ax.flatten()
for idx in np.arange(largest_parameters_map.shape[1]):
    # a.hist(marginal_samples_array[:,idx], bins=bins,color=colors[0], histtype=u'step')
    ax[idx].hist(samples['Baseline'][:,idx], bins=bins,color=colors[1], histtype=u'step')
    ax[idx].hist(samples['Hyperexcitable'][:,idx], bins=bins,color=colors[2], histtype=u'step')
    ax[idx].set_xlabel(largest_labels[idx])
    # ax[idx].set_xlim(limits[idx])

ax[0].legend(("Healthy", "IN Loss"))
"""


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

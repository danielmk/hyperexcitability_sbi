# -*- coding: utf-8 -*-
"""
This script loads the amortized inference object and plots the training data.
"""

import outcomes
import os
import pickle
import matplotlib.pyplot as plt
from metadata import Metadata


f = os.path.join(Metadata.results_dir, "amortized_inference.pickle")


def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


file_read = list(loadall(f))

inference = file_read[0]

data = inference.get_simulations()

active = data[1][:, 0] > 0

cv_defined = data[1][:, 6] > 0

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})

colors = ['#1b9e77', '#d95f02', '#7570b3']

fig, ax = plt.subplots(1, 4)

parameter_labels = Metadata.outcome_labels

parameter_idc = [[0, 1], [2, 3], [2, 4], [5, 6]]

alpha = 0.01
ms = 40
mw = 5

bl_outcome = outcomes.x_baseline['baseline']

he_outcome = outcomes.x_hyperexcitable['hyperexcitable']

for idx, params in enumerate(parameter_idc):
    x, y = params
    ax[idx].plot(data[1][active & cv_defined, x], data[1][active & cv_defined, y],
                 'o',
                 rasterized=True,
                 color='k',
                 alpha=alpha)
    ax[idx].set_xlabel(parameter_labels[x])
    ax[idx].set_ylabel(parameter_labels[y])
    ax[idx].set_box_aspect(1)
    if x in [2, 3, 4]:
        ax[idx].set_xscale('log')
    if y in [2, 3, 4]:
        ax[idx].set_yscale('log')

    # Mark the target outcome of the two conditions
    ax[idx].plot(bl_outcome[x], bl_outcome[y], 'x', color=colors[0], markersize=ms, markeredgewidth=mw)
    ax[idx].plot(he_outcome[x], he_outcome[y], 'x', color=colors[1], markersize=ms, markeredgewidth=mw)

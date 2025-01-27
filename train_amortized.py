# -*- coding: utf-8 -*-
"""
Load amortized samples and train the density estimator. The density estimator
is saved by pickling.
"""

import torch
import priors
import tables
import pickle
import numpy as np
from itertools import chain
from sbi.inference import SNPE_C
import os
import platform
import sys

"""LOAD ALL AMORTIZE SAMPLES IN batch_directory"""
if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
elif platform.system() == 'Linux':
    results_dir = '/flash/FukaiU/danielmk/sbips_sparsity/'

batch_directory = 'amortized_samples'

filepath = os.path.join(results_dir, batch_directory)

filenames = os.listdir(filepath)

files = [os.path.join(filepath, f) for f in filenames]  # Full paths to all files

all_theta = []
all_x = []
for file in files:
    f = tables.open_file(file, mode='r')
    runs = list(f.root._v_children)
    for k in runs:
        x = f.root[k].x.read()
        theta = f.root[k].theta.read()
        all_x.append(x)
        all_theta.append(theta)

all_x_flat = torch.Tensor(np.array(list(chain.from_iterable(all_x))))
all_theta_flat = torch.Tensor(np.array(list(chain.from_iterable(all_theta))))

all_x_flat = torch.nan_to_num(all_x_flat, nan=0)  # Convert NaNs to zero

prior_dict = priors.baseline_epilepsy

prior = priors.prior_dict_to_tensor(prior_dict)

inference = SNPE_C(prior=prior)

inference = inference.append_simulations(all_theta_flat, all_x_flat)

print("Starting Training")

inference.train(training_batch_size=2000)

"""LOADING SAVING CODE!"""
with open(os.path.join(results_dir, 'amortized_inference_reproducibility_two'), 'wb') as f:
    pickle.dump(inference, f, pickle.HIGHEST_PROTOCOL)





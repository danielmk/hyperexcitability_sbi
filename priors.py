# -*- coding: utf-8 -*-
"""
priors.py contains dictionaries that define the parameters of the model that
are subject to simulation based inference and are therefore changed between
each simulation run by sampling from a distribution. Priors are always
BoxUniform so far.

A prior is defined by a torch.Tensor that defines the upper and lower bound.
Because torch.Tensor does not work with Brian2 quantities, quantities need
to be stripped, and then reattached to the sample within the simulator.

The helper function get_base_units defined here helps with getting the units
but it is sensitive to the order in the iterable and does not respect the
dictionary keys. Example code to generate a prior usable in simulation based
inference:
    >>> import brian2 as b2
    >>> import torch, priors
    >>> low = torch.tensor(b2.asarray(list(priors.baseline['low'].values())))
    >>> high = torch.tensor(b2.asarray(list(priors.baseline['high'].values())))
    >>> prior = sbi.utils.BoxUniform(low, high)


The parameters that are subject to simulation based inference are in priors.py
"""

from brian2 import *
import brian2 as b2
import sbi.utils
import torch

"""BASELINE LOW/HIGH FROM https://neuroelectro.org/
for pc dentate gyrus granule cell was referenced
for ain Hippocampus CA1 oriens lacunosum moleculare neuron
for nin Hippocampus CA1 basket cells
"""

baseline = {
    'low': {
        'pc_C': 10.0 * pF,
        'pc_gL': 2.0 * nS,
        'pc_EL': -80.0 * mV,
        'pc_VT': -49.0 * mV,
        
        'ain_n': 10,
        'ain_C': 96.0 * pF,
        'ain_gL': 2.0 * nS,
        'ain_EL': -65.0 * mV,
        'ain_VT': -63.0 * mV,
        
        'nin_n': 10,
        'nin_C': 10.0 * pF,
        'nin_gL': 3.0 * nS,
        'nin_EL': -70.0 * mV,
        'nin_VT': -52.0 * mV,
    
        'pc_pc_p': 0.0,
        'pc_pc_A_SE': 0 * nS,

        'pc_ain_p': 0.0,
        'pc_ain_A_SE': 0 * nS,
    
        'pc_nin_p': 0.0,
        'pc_nin_A_SE': 0 * nS,
    
        'nin_pc_p': 0.0,
        'nin_pc_A_SE': 0 * nS,
    
        'nin_nin_p': 0.0,
        'nin_nin_A_SE': 0 * nS,
    
        'ain_pc_p': 0.0,
        'ain_pc_A_SE': 0 * nS,
    
        'ain_nin_p': 0.0,
        'ain_nin_A_SE': 0 * nS,
    
        'ain_ain_p': 0.0,
        'ain_ain_A_SE': 0 * nS,
    
        'nin_nin_gap_p': 0.0,
        'nin_nin_gap_w': 0.0 * nS
        },
    'high': {
        'pc_C': 100.0 * pF,
        'pc_gL': 20.0 * nS,
        'pc_EL': -64.0 * mV,
        'pc_VT': -39.0 * mV,
        
        'ain_n': 100,
        'ain_C': 209.0 * pF,
        'ain_gL': 6.4 * nS,
        'ain_EL': -51.0 * mV,
        'ain_VT': -44.0 * mV,
        
        'nin_n': 100,
        'nin_C': 323.0 * pF,
        'nin_gL': 13.5 * nS,
        'nin_EL': -58.0 * mV,
        'nin_VT': -37.0 * mV,

        'pc_pc_p': 0.3,
        'pc_pc_A_SE': 50 * nS,

        'pc_ain_p': 0.3,
        'pc_ain_A_SE': 50 * nS,

        'pc_nin_p': 0.3,
        'pc_nin_A_SE': 50 * nS,

        'nin_pc_p': 0.3,
        'nin_pc_A_SE': 50 * nS,

        'nin_nin_p': 0.3,
        'nin_nin_A_SE': 50 * nS,

        'ain_pc_p': 0.3,
        'ain_pc_A_SE': 50 * nS,

        'ain_nin_p': 0.3,
        'ain_nin_A_SE': 50 * nS,

        'ain_ain_p': 0.3,
        'ain_ain_A_SE': 50 * nS,

        'nin_nin_gap_p': 0.3,
        'nin_nin_gap_w': 4 * nS
        },

    'constants': {
        'pc_n': 500,
        'pc_DeltaT': 0.8 * mV,
        'pc_tau_w': 88.0 * ms,
        'pc_a': -0.8 * nS,
        'pc_b': 65 * pA,
        'pc_Vr': -53.0 * mV,
        'ain_DeltaT': 5.5 * mV,
        'ain_tau_w': 41 * ms,
        'ain_a': 2.0 * nS,
        'ain_b': 55 * pA,
        'ain_Vr': -54.0 * mV,
        'nin_DeltaT': 3.0 * mV,
        'nin_tau_w': 16 * ms,
        'nin_a': 1.8 * nS,
        'nin_b': 61 * pA,
        'nin_Vr': -54.0 * mV,
        'inp_pc_p': 0.2,
        'inp_pc_A_SE': 5.5 * nS,
        'inp_pc_tau_inact': 5.5 * ms,
        'inp_pc_tau_rec': 800 * ms,
        'inp_pc_U_SE': 0.5,
        'inp_pc_tau_facil': 0 * ms,
        'inp_nin_p': 0.1,
        'inp_nin_A_SE': 7.0 * nS,
        'inp_nin_tau_inact': 6.3 * ms,
        'inp_nin_tau_rec': 800 * ms,
        'inp_nin_U_SE': 0.5,
        'inp_nin_tau_facil': 0 * ms,
        'pc_pc_tau_inact': 10.0 * ms,
        'pc_pc_tau_rec': 800 * ms,
        'pc_pc_U_SE': 0.5,
        'pc_pc_tau_facil': 0 * ms,
        'pc_ain_tau_inact': 4.0 * ms,
        'pc_ain_tau_rec': 130 * ms,
        'pc_ain_U_SE': 0.03,
        'pc_ain_tau_facil': 530 * ms,
        'pc_nin_tau_inact': 10.0 * ms,
        'pc_nin_tau_rec': 800 * ms,
        'pc_nin_U_SE': 0.5,
        'pc_nin_tau_facil': 0 * ms,
        'nin_pc_tau_inact': 5.5 * ms,
        'nin_pc_tau_rec': 800 * ms,
        'nin_pc_U_SE': 0.5,
        'nin_pc_tau_facil': 0 * ms,
        'nin_nin_tau_inact': 2.5 * ms,
        'nin_nin_tau_rec': 800 * ms,
        'nin_nin_U_SE': 0.5,
        'nin_nin_tau_facil': 0 * ms,
        'ain_pc_tau_inact': 6.0 * ms,
        'ain_pc_tau_rec': 800 * ms,
        'ain_pc_U_SE': 0.5,
        'ain_pc_tau_facil': 0 * ms,
        'ain_nin_tau_inact': 6.0 * ms,
        'ain_nin_tau_rec': 50 * ms,
        'ain_nin_U_SE': 0.03,
        'ain_nin_tau_facil': 0 * ms,
        'ain_ain_tau_inact': 5.5 * ms,
        'ain_ain_tau_rec': 130 * ms,
        'ain_ain_U_SE': 0.03,
        'ain_ain_tau_facil': 530 * ms,
        }
    }


baseline_low_names = list(baseline['low'].keys())
baseline_high_names = list(baseline['high'].keys())
assert baseline_low_names == baseline_high_names

"""HELPER"""
def get_base_units(iterable):
    return [units.get_unit(x.dimensions) if hasattr(x, 'dimensions') else 1 for x in iterable]

def test_prior(prior : dict):
    baseline_low_names = list(prior['low'].keys())
    baseline_high_names = list(prior['high'].keys())
    assert baseline_low_names == baseline_high_names

def prior_dict_to_tensor(prior: dict):
    return sbi.utils.BoxUniform(torch.tensor(b2.asarray(list(prior['low'].values()))),
                                                   torch.tensor(b2.asarray(list(prior['high'].values()))))

def prior_dict_to_pure_tensor(prior: dict):
    return torch.tensor(b2.asarray(list(prior['low'].values()))), torch.tensor(b2.asarray(list(prior['high'].values())))

def prior_dict_to_tensor_gpu(prior: dict):
    return sbi.utils.BoxUniform(torch.tensor(b2.asarray(list(prior['low'].values())), device='cuda'),
                                                   torch.tensor(b2.asarray(list(prior['high'].values())), device='cuda'))

"""TESTING"""
test_prior(baseline)


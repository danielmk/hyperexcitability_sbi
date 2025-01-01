# -*- coding: utf-8 -*-
"""

"""

from brian2 import *
import torch
import priors
from operator import mul

baseline = {
        'pc_C': 104.0 * pF,
        'pc_gL': 4.3 * nS,
        'pc_EL': -65.0 * mV,
        'pc_VT': -52.0 * mV,

        'ain_n': 50,
        'ain_C': 83.0 * pF,
        'ain_gL': 1.7 * nS,
        'ain_EL': -59.0 * mV,
        'ain_VT': -56.0 * mV,
        
        'nin_n': 50,
        'nin_C': 59.0 * pF,
        'nin_gL': 2.9 * nS,
        'nin_EL': -62.0 * mV,
        'nin_VT': -42.0 * mV,
    
        'pc_pc_p': 0.01,
        'pc_pc_A_SE': 8.5 * nS,
    
        'pc_ain_p': 0.1,
        'pc_ain_A_SE': 8.5 * nS,
    
        'pc_nin_p': 0.1,
        'pc_nin_A_SE': 8.5 * nS,
    
        'nin_pc_p': 0.2,
        'nin_pc_A_SE': 25.0 * nS,
    
        'nin_nin_p': 0.05,
        'nin_nin_A_SE': 8.5 * nS,
    
        'ain_pc_p': 0.2,
        'ain_pc_A_SE': 25.0 * nS,
    
        'ain_nin_p': 0.1,
        'ain_nin_A_SE': 8.5 * nS,
    
        'ain_ain_p': 0.05,
        'ain_ain_A_SE': 8.5 * nS,
    
        'nin_nin_gap_p': 0.02,
        'nin_nin_gap_w': 2.0 * nS
    }

hyperexcitable = {
        'pc_C': 104.0 * pF,
        'pc_gL': 4.3 * nS,
        'pc_EL': -65.0 * mV,
        'pc_VT': -52.0 * mV,

        'ain_n': 50,
        'ain_C': 83.0 * pF,
        'ain_gL': 1.7 * nS,
        'ain_EL': -59.0 * mV,
        'ain_VT': -56.0 * mV,
        
        'nin_n': 50,
        'nin_C': 59.0 * pF,
        'nin_gL': 2.9 * nS,
        'nin_EL': -62.0 * mV,
        'nin_VT': -42.0 * mV,
    
        'pc_pc_p': 0.1,
        'pc_pc_A_SE': 8.5 * nS,
    
        'pc_ain_p': 0.1,
        'pc_ain_A_SE': 8.5 * nS,
    
        'pc_nin_p': 0.1,
        'pc_nin_A_SE': 8.5 * nS,
    
        'nin_pc_p': 0.2,
        'nin_pc_A_SE': 25.0 * nS,
    
        'nin_nin_p': 0.05,
        'nin_nin_A_SE': 8.5 * nS,
    
        'ain_pc_p': 0.2,
        'ain_pc_A_SE': 25.0 * nS,
    
        'ain_nin_p': 0.1,
        'ain_nin_A_SE': 8.5 * nS,
    
        'ain_ain_p': 0.05,
        'ain_ain_A_SE': 8.5 * nS,
    
        'nin_nin_gap_p': 0.02,
        'nin_nin_gap_w': 2.0 * nS
    }


baseline_map = {
    'pc_C': 71.06055327 * pfarad,
    'pc_gL': 3.11561621 * nsiemens,
    'pc_EL': -67.86698848 * mvolt,
    'pc_VT': -47.98539728 * mvolt,
    'ain_n': 54.68229675292969,
    'ain_C': 152.44558393 * pfarad,
    'ain_gL': 4.13255563 * nsiemens,
    'ain_EL': -58.29555914 * mvolt,
    'ain_VT': -46.34563997 * mvolt,
    'nin_n': 54.014652252197266,
    'nin_C': 284.0975788 * pfarad,
    'nin_gL': 8.44374171 * nsiemens,
    'nin_EL': -64.3619895 * mvolt,
    'nin_VT': -50.49589649 * mvolt,
    'pc_pc_p': 0.01562972366809845,
    'pc_pc_A_SE': 4.35081171 * nsiemens,
    'pc_ain_p': 0.14659613370895386,
    'pc_ain_A_SE': 25.45225719 * nsiemens,
    'pc_nin_p': 0.13766950368881226,
    'pc_nin_A_SE': 23.79141684 * nsiemens,
    'nin_pc_p': 0.03093755431473255,
    'nin_pc_A_SE': 23.38078708 * nsiemens,
    'nin_nin_p': 0.15083956718444824,
    'nin_nin_A_SE': 25.64232915 * nsiemens,
    'ain_pc_p': 0.06018993631005287,
    'ain_pc_A_SE': 18.60139953 * nsiemens,
    'ain_nin_p': 0.14831459522247314,
    'ain_nin_A_SE': 25.15739261 * nsiemens,
    'ain_ain_p': 0.15746809542179108,
    'ain_ain_A_SE': 25.63612433 * nsiemens,
    'nin_nin_gap_p': 0.23130398988723755,
    'nin_nin_gap_w': 1.89489957 * nsiemens}

hyperexcitable_map = {
    'pc_C': 88.24110193 * pfarad,
    'pc_gL': 6.19234086 * nsiemens,
    'pc_EL': -64.32972848 * mvolt,
    'pc_VT': -48.51631075 * mvolt,
    'ain_n': 55.72412109375,
    'ain_C': 153.18854518 * pfarad,
    'ain_gL': 4.14563583 * nsiemens,
    'ain_EL': -58.36296827 * mvolt,
    'ain_VT': -52.8800115 * mvolt,
    'nin_n': 55.722434997558594,
    'nin_C': 203.07024118 * pfarad,
    'nin_gL': 8.16223444 * nsiemens,
    'nin_EL': -64.10915405 * mvolt,
    'nin_VT': -44.3085134 * mvolt,
    'pc_pc_p': 0.05446219816803932,
    'pc_pc_A_SE': 13.85207149 * nsiemens,
    'pc_ain_p': 0.1424388438463211,
    'pc_ain_A_SE': 22.28165741 * nsiemens,
    'pc_nin_p': 0.13640761375427246,
    'pc_nin_A_SE': 22.51029763 * nsiemens,
    'nin_pc_p': 0.032981738448143005,
    'nin_pc_A_SE': 18.39286234 * nsiemens,
    'nin_nin_p': 0.1505744755268097,
    'nin_nin_A_SE': 25.43468547 * nsiemens,
    'ain_pc_p': 0.052123356610536575,
    'ain_pc_A_SE': 18.85727485 * nsiemens,
    'ain_nin_p': 0.14363223314285278,
    'ain_nin_A_SE': 24.80638273 * nsiemens,
    'ain_ain_p': 0.15768589079380035,
    'ain_ain_A_SE': 25.76801528 * nsiemens,
    'nin_nin_gap_p': 0.15244834125041962,
    'nin_nin_gap_w': 1.95394145 * nsiemens}

"""HELPERS"""
def theta_dict_to_tensor(theta : dict):
    return torch.tensor(asarray(list(theta.values())))

def tensor_to_theta_dict(theta : torch.tensor, template : dict):
    keys = template.keys()
    units = priors.get_base_units(template.values())
    values = map(mul, theta.tolist(), units)
    dict_output = dict(zip(keys, values))
    return dict_output

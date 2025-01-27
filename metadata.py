# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:18:21 2024

@author: Daniel
"""

from dataclasses import dataclass
import os
import platform
import brian2 as b2

if platform.system() == 'Windows':
    results_dir = os.path.join(os.path.dirname(__file__), 'data')
elif platform.system() == 'Linux':
    results_dir = r'/data/'

@dataclass
class Metadata:
    parameter_labels: tuple = (
        r'$PC_C$',
        r'$PC_{g_L}$',
        r'$PC_{E_L}$',
        r'$PC_{V_T}$',
        r'$AIN_N$',
        r'$AIN_C$',
        r'$AIN_{g_L}$',
        r'$AIN_{E_L}$',
        r'$AIN_{V_T}$',
        r'$NIN_N$',
        r'$NIN_C$',
        r'$NIN_{g_L}$',
        r'$NIN_{E_L}$',
        r'$NIN_{V_T}$',
        r'$PC-PC_P$',
        r'$PC-PC_{A_{SE}}$',
        r'$PC-AIN_P$',
        r'$PC-AIN_{A_{SE}}$',
        r'$PC-NIN_P$',
        r'$PC-NIN_{A_{SE}}$',
        r'$NIN-PC_P$',
        r'$NIN-PC_{A_{SE}}$',
        r'$NIN-NIN_P$',
        r'$NIN-NIN_{A_{SE}}$',
        r'$AIN-PC_P$',
        r'$AIN-PC_{A_{SE}}$',
        r'$AIN-NIN_P$',
        r'$AIN-NIN_{A_{SE}}$',
        r'$AIN-AIN_P$',
        r'$AIN-AIN_{A_{SE}}$',
        r'$NIN-NIN_{GAP_P}$',
        r'$NIN-NIN_{GAP_W}$',
        )

    outcome_labels: tuple = (
        'Mean Rate',
        'Mean Entropy',
        'Theta Power',
        'Gamma Power',
        'Fast Power',
        'Correlation',
        'CV')
    
    results_dir: str = results_dir  # Where scripts save and load data
    
    sim_dt: b2.units.fundamentalunits.Quantity = 0.1 * b2.ms
    
    sim_duration: b2.units.fundamentalunits.Quantity = 1.0 * b2.second

    
    
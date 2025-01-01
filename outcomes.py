# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:00:55 2024

@author: Daniel
"""

import torch

x_healthy_baseline = {'x_healthy_baseline': 
                      torch.tensor([15, 3.0, 8.5e-9, 1.0e-8, 2.5e-10, 0.05, 0.45])
                      }

x_healthy_v2 = {'x_healthy_v2': 
                    torch.tensor([18, 2.7, 1.0e-7, 1.0e-8, 1.0e-9, 0.05, 0.45])
                    }

x_interictal_spiking = {'x_interictal_spiking':
                      torch.tensor([75, 1.0, 9e-6, 7e-9, 6e-10, 0.99, 4.0])
                      }
    
x_theta_synchrony = {'x_theta_synchrony':
                      torch.tensor([33, 2.6, 2e-6, 2e-9, 1e-10, 0.6, 1.4])
                      }

x_hyperexcitable = {'hyperexcitable':
                      torch.tensor([40, 1.0, 9e-6, 5e-8, 1.0e-8, 0.95, 4.0])
                      }
    
baseline_good_separation = torch.tensor([0.015])

baseline_intermediate_separation = torch.tensor([0.0008])

baseline_weak_separation = torch.tensor([7e-05])

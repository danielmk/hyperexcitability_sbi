# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:18:21 2024

@author: Daniel
"""

from dataclasses import dataclass
from copy import deepcopy
from metadata import EpilepsyMetadata
import os
import numpy as np


@dataclass
class Conditional:
    name: str
    fixed_parameters: dict
    parameter_labels: tuple
    n_samples: int = 100000
    
    def __post_init__(self):
        self.condition = self._get_condition()
        self.unconditioned_labels = self._get_unconditioned_labels()
        self.dims_to_sample = self._get_dims_to_sample()
    
    def _get_condition(self):
        condition = np.zeros(len(self.parameter_labels))
        for key, value in self.fixed_parameters.items():
            condition[key] = value
        return condition

    def _get_unconditioned_labels(self):
        curr_labels = list(self.parameter_labels)
          # DELETE THESE INDICES FROM THE LABELS LIST
        for index in sorted(self.fixed_parameters.keys(), reverse=True):
            del curr_labels[index]
        return curr_labels
    
    def _get_dims_to_sample(self):
        dims = np.arange(len(self.parameter_labels), dtype=int)
        dims_to_sample = np.delete(dims, list(self.fixed_parameters.keys()))
        return dims_to_sample
    
    def make_theta(self, thetas, axis=1):
        for key, value in self.fixed_parameters.items():
            thetas = np.insert(thetas, key, [value], axis=axis)
            
        return thetas

in_loss_conditional = Conditional("IN Loss", {4: 15, 9: 15}, EpilepsyMetadata.parameter_labels)

in_normal_conditional = Conditional("IN Normal", {4: 55, 9: 54}, EpilepsyMetadata.parameter_labels)

synapse_only_sprouting_conditional = Conditional("Synapse Sprouting Only", {14: 0.03}, EpilepsyMetadata.parameter_labels)

synapse_only_normal_conditional = Conditional("Synapse Sprouting Only", {14: 0.0015}, EpilepsyMetadata.parameter_labels)

#synapse_sprouting_conditional = Conditional("Synapse Sprouting", {14: 0.0066, 15: 4.35081171e-9}, EpilepsyMetadata.parameter_labels)

#synapse_normal_conditional = Conditional("Synapse Normal", {14: 0.0001, 15: 4.35081171e-9}, EpilepsyMetadata.parameter_labels)

intrinsics_depolarized_conditional = Conditional("Intrinsics Depolarized", {1: 2.5e-9, 2: -66e-3}, EpilepsyMetadata.parameter_labels)

intrinsics_normal_conditional= Conditional("Intrinsics Normal", {1: 3.0e-9, 2: -68e-3}, EpilepsyMetadata.parameter_labels)






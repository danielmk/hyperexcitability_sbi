# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:14:50 2023

@author: Daniel
"""

from brian2 import *
import matplotlib.pyplot as plt
import sys

start_scope()

# NEURON TYPE: Adaptive Exponential
# TODO Gotta define each neuron separtely with different synaptic currents
# pc_synaptic_terms = "I_inp + I_pcs + I_fb + I_in"
model_pc = '''dvs/dt = (-gL*(vs-EL)+gL*DeltaT*exp((vs-VT)/DeltaT)-w+I_step(t))/C : volt
           dw/dt = (a*(vs-EL)-w)/tau_w : amp
           tau_w : second
           b : amp
           DeltaT : volt
           a : siemens
           C : farad
           gL : siemens
           EL : volt
           VT : volt
           Vr : volt'''

model_reset='''vs=Vr
               w=w+b
            '''

# SYNAPSE TYPE: Tsodyks-Markram
# CELL Types: PC, FB-IN, IN
n_pc = 1
n_nin = 1
n_ain = 1

# Define PCs
# Parameters from Naud et al. 2008, of a "regular spiking" cell
pc = NeuronGroup(n_pc, model_pc, threshold='vs>0.0*mV', reset=model_reset)
pc.C = 71.06055327 * pfarad
pc.gL = 3.11561621 * nsiemens
pc.EL = -67.86698848 * mvolt
pc.VT = -47.98539728 * mvolt
pc.DeltaT = 0.8 * mV
pc.tau_w = 88.0 * ms
pc.a = -0.8 * nS
pc.b = 65 * pA
pc.Vr = -53.0 * mV

pc.vs = -67.866988485 * mV
pc.w = 0.0 * amp

# Parameters from Naud et al. 2008, of a "continuous accommodating" cell
# Reset threshold is 0mV
# ORIGINAL:

# MODIFIED
ain = NeuronGroup(n_ain, model_pc, threshold='vs>0.0*mV', reset=model_reset)
ain.C = 152.44558393 * pfarad
ain.gL = 4.13255563 * nsiemens
ain.EL = -58.29555914 * mvolt
ain.VT = -46.34563997 * mvolt
ain.DeltaT = 5.5 * mV
ain.tau_w = 41 * ms
ain.a = 2.0 * nS
ain.b = 55 * pA
ain.Vr = -54.0 * mV

ain.vs = -58.29555914 * mV
ain.w = 0.0 * amp

# Parameters from Naud et al. 2008, of a "continuous non-accommodating" cell
# Reset threshold is 0mV
nin = NeuronGroup(n_nin, model_pc, threshold='vs>0.0*mV', reset=model_reset)
nin.C = 284.0975788 * pfarad
nin.gL = 8.44374171 * nsiemens
nin.EL = -64.3619895 * mvolt
nin.VT = -50.49589649 * mvolt
nin.DeltaT = 3.0 * mV
nin.tau_w = 16 * ms
nin.a = 1.8 * nS
nin.b = 61 * pA
nin.Vr = -54.0 * mV

nin.vs = -64.3619895 * mV
nin.w = 0.0 * amp

magnitude = 200

I_step = TimedArray(
    [0, 0, magnitude, magnitude, magnitude, magnitude, magnitude, magnitude, magnitude, magnitude, magnitude, magnitude, 0, 0]*pA,
    dt=50*ms
)

"""RECORDINGS AND SIMULATION"""
pc_vs = StateMonitor(pc, 'vs', record=True)

ain_vs = StateMonitor(ain, 'vs', record=True)

nin_vs = StateMonitor(nin, 'vs', record=True)

pc_s = SpikeMonitor(pc, 'vs', record=True)

ain_s = SpikeMonitor(ain, 'vs', record=True)

nin_s = SpikeMonitor(nin, 'vs', record=True)

dt = 0.1*ms

defaultclock.dt = dt

run(1000*ms)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 16

# pc_v = pc_vs.vs[0]
# pc_v[(pc_s.t / dt).astype(int)-1] = 0

plt.figure()
plt.plot(pc_vs.t, pc_vs.vs[0])
plt.title("pc")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")

plt.figure()
plt.plot(ain_vs.t, ain_vs.vs[0])
plt.title("ain")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")

plt.figure()
plt.plot(nin_vs.t, nin_vs.vs[0])
plt.title("nin")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")


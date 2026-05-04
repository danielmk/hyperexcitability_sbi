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
model_pc = '''dvs/dt = (-gL*(vs-EL)+gL*DeltaT*exp((vs-VT)/DeltaT)-w+stim(t) + I_total)/C : volt
           dw/dt = (a*(vs-EL)-w)/tau_w : amp
           I_total = I_inp + I_pc - I_bc - I_inn : amp
           tau_w : second
           b : amp
           DeltaT : volt
           a : siemens
           C : farad
           gL : siemens
           EL : volt
           VT : volt
           Vr : volt
           I_inp : amp
           I_pc: amp
           I_bc : amp
           I_inn : amp'''

model_reset='''vs=Vr
               w=w+b
            '''

# SYNAPSE TYPE: Tsodyks-Markram
# CELL Types: PC, FB-IN, IN
n_pc = 2000
n_bc = 100
n_inn = 100

# Define PCs
# Parameters from Naud et al. 2008, of a "regular spiking" cell
pc = NeuronGroup(n_pc, model_pc, threshold='vs>0.0*mV', reset=model_reset)
pc.C = 104 *pF
pc.gL = 4.3 * nS
pc.EL = -65.0 * mV
pc.VT = -52.0 * mV
pc.DeltaT = 0.8 * mV
pc.tau_w = 88.0 * ms
pc.a = -0.8 * nS
pc.b = 65 * pA
pc.Vr = -53.0 * mV

pc.vs = np.random.normal(pc.EL[0], 3*mvolt, pc.vs.shape[0]) * volt

# Parameters from Naud et al. 2008, of a "continuous accommodating" cell
# Reset threshold is 0mV
# ORIGINAL:
"""
inn = NeuronGroup(n_inn, model_pc, threshold='vs>0.0*mV', reset=model_reset)
inn.C = 83 *pF
inn.gL = 1.7 * nS
inn.EL = -59.0 * mV
inn.VT = -56.0 * mV
inn.DeltaT = 5.5 * mV
inn.tau_w = 41 * ms
inn.a = 2.0 * nS
inn.b = 55 * pA
inn.Vr = -54.0 * mV
"""

# MODIFIED
inn = NeuronGroup(n_inn, model_pc, threshold='vs>0.0*mV', reset=model_reset)
inn.C = 59.0 *pF
inn.gL = 2.9 * nS
inn.EL = -62.0 * mV
inn.VT = -42.0 * mV
inn.DeltaT = 3.0 * mV
inn.tau_w = 16 * ms
inn.a = 1.8 * nS
inn.b = 61 * pA
inn.Vr = -54.0 * mV

inn.vs = np.random.normal(inn.EL[0], 3*mV, inn.vs.shape[0]) * volt

# Parameters from Naud et al. 2008, of a "continuous non-accommodating" cell
# Reset threshold is 0mV
bc = NeuronGroup(n_bc, model_pc, threshold='vs>0.0*mV', reset=model_reset)
bc.C = 59.0 *pF
bc.gL = 2.9 * nS
bc.EL = -62.0 * mV
bc.VT = -42.0 * mV
bc.DeltaT = 3.0 * mV
bc.tau_w = 16 * ms
bc.a = 1.8 * nS
bc.b = 61 * pA
bc.Vr = -54.0 * mV

bc.vs = np.random.normal(bc.EL[0], 3*mV, bc.vs.shape[0]) * volt

# Creating Synapses
# From: https://brian2.readthedocs.io/en/latest/examples/frompapers.Tsodyks_Pawelzik_Markram_1998.html
synapses_model =    """
                    dx/dt =  z/tau_rec   : 1 (clock-driven) # recovered
                    dy/dt = -y/tau_inact : 1 (clock-driven) # active
                    du/dt = -u/tau_facil : 1 (clock-driven)
                    tau_facil : second
                    A_SE : ampere
                    U_SE : 1
                    tau_inact : second
                    tau_rec : second
                    z = 1 - x - y : 1 # inactive
                    """
                    # I_inp_post = A_SE*y : ampere (summed)
synapses_input = synapses_model + f"I_inp_post = A_SE*y : ampere (summed)"
synapses_pc = synapses_model + f"I_pc_post = A_SE*y : ampere (summed)"
synapses_bc = synapses_model + f"I_bc_post = A_SE*y : ampere (summed)"
synapses_inn = synapses_model + f"I_inn_post = A_SE*y : ampere (summed)"              
                    
synapses_action =   """
                    u += U_SE*(1-u)
                    y += u*x # important: update y first
                    x += -u*x
                    """

def get_stimulus(start, stop, frequency):
    """
    start -- start time of stimulus
    stop -- stop time of stimulus
    frequency -- frequency of stimulus
    """

    times = np.arange(start / ms, stop / ms, 1 / (frequency / Hz) * 1e3) * ms
    stimulus = SpikeGeneratorGroup(1, [0] * len(times), times)

    return stimulus

# syn_stimulus = get_stimulus(500*ms, 1000*ms, 10*Hz)

# Connections:
    # I->pc
    # I->bc
    # pc->pc
    # pc->inn
    # pc->bc
    # bc->pc
    # bc->bc
    # bc->inn
    # inn->pc
    # inn->inn
    # inn->bc

P = PoissonGroup(200, 20*Hz)

# I->pc
inp_pc_synapses = Synapses(P,
                    pc,
                    model=synapses_input,
                    on_pre=synapses_action,
                    method="exponential_euler")

inp_pc_synapses.connect(p=0.5)

inp_pc_synapses.tau_inact = 1.5 * ms
inp_pc_synapses.A_SE = 400 * pA
inp_pc_synapses.tau_rec = 130 * ms
inp_pc_synapses.U_SE = 0.03
inp_pc_synapses.tau_facil = 530 * ms

# I->bc
inp_bc_synapses = Synapses(P,
                    bc,
                    model=synapses_input,
                    on_pre=synapses_action,
                    method="exponential_euler")

inp_bc_synapses.connect(p=0.5)

inp_bc_synapses.tau_inact = 1.5 * ms
inp_bc_synapses.A_SE = 540 * pA
inp_bc_synapses.tau_rec = 130 * ms
inp_bc_synapses.U_SE = 0.03
inp_bc_synapses.tau_facil = 530 * ms

# pc->pc
pc_pc_synapses = Synapses(pc,
                    pc,
                    model=synapses_pc,
                    on_pre=synapses_action,
                    method="exponential_euler")

pc_pc_synapses.connect(condition='i != j', p=0.01)

pc_pc_synapses.tau_inact = 1.5 * ms
pc_pc_synapses.A_SE = 200 * pA
pc_pc_synapses.tau_rec = 130 * ms
pc_pc_synapses.U_SE = 0.03
pc_pc_synapses.tau_facil = 530 * ms

# pc->inn
pc_inn_synapses = Synapses(pc,
                    inn,
                    model=synapses_pc,
                    on_pre=synapses_action,
                    method="exponential_euler")

pc_inn_synapses.connect(p=0.1)

pc_inn_synapses.tau_inact = 1.5 * ms
pc_inn_synapses.A_SE = 50 * pA
pc_inn_synapses.tau_rec = 130 * ms
pc_inn_synapses.U_SE = 0.03
pc_inn_synapses.tau_facil = 530 * ms

# pc->bc
pc_bc_synapses = Synapses(pc,
                    bc,
                    model=synapses_pc,
                    on_pre=synapses_action,
                    method="exponential_euler")

pc_bc_synapses.connect(p=0.1)

pc_bc_synapses.tau_inact = 1.5 * ms
pc_bc_synapses.A_SE = 100 * pA
pc_bc_synapses.tau_rec = 130 * ms
pc_bc_synapses.U_SE = 0.03
pc_bc_synapses.tau_facil = 530 * ms

# bc->pc
bc_pc_synapses = Synapses(bc,
                    pc,
                    model=synapses_bc,
                    on_pre=synapses_action,
                    method="exponential_euler")

bc_pc_synapses.connect(p=0.1)

bc_pc_synapses.tau_inact = 1.5 * ms
bc_pc_synapses.A_SE = 100 * pA
bc_pc_synapses.tau_rec = 130 * ms
bc_pc_synapses.U_SE = 0.03
bc_pc_synapses.tau_facil = 530 * ms

# bc->bc
bc_bc_synapses = Synapses(bc,
                    bc,
                    model=synapses_bc,
                    on_pre=synapses_action,
                    method="exponential_euler")

bc_bc_synapses.connect(condition='i != j', p=0.01)

bc_bc_synapses.tau_inact = 1.5 * ms
bc_bc_synapses.A_SE = 100 * pA
bc_bc_synapses.tau_rec = 130 * ms
bc_bc_synapses.U_SE = 0.03
bc_bc_synapses.tau_facil = 530 * ms

# bc->inn
bc_inn_synapses = Synapses(bc,
                    inn,
                    model=synapses_bc,
                    on_pre=synapses_action,
                    method="exponential_euler")

bc_inn_synapses.connect(p=0.01)

bc_inn_synapses.tau_inact = 1.5 * ms
bc_inn_synapses.A_SE = 100 * pA
bc_inn_synapses.tau_rec = 130 * ms
bc_inn_synapses.U_SE = 0.03
bc_inn_synapses.tau_facil = 530 * ms

# inn->pc
inn_pc_synapses = Synapses(inn,
                    pc,
                    model=synapses_inn,
                    on_pre=synapses_action,
                    method="exponential_euler")

inn_pc_synapses.connect(p=0.01)

inn_pc_synapses.tau_inact = 1.5 * ms
inn_pc_synapses.A_SE = 100 * pA
inn_pc_synapses.tau_rec = 130 * ms
inn_pc_synapses.U_SE = 0.03
inn_pc_synapses.tau_facil = 530 * ms

# inn->bc
inn_bc_synapses = Synapses(inn,
                    bc,
                    model=synapses_inn,
                    on_pre=synapses_action,
                    method="exponential_euler")

inn_bc_synapses.connect(p=0.1)

inn_bc_synapses.tau_inact = 1.5 * ms
inn_bc_synapses.A_SE = 100 * pA
inn_bc_synapses.tau_rec = 130 * ms
inn_bc_synapses.U_SE = 0.03
inn_bc_synapses.tau_facil = 530 * ms

# inn->inn
inn_inn_synapses = Synapses(inn,
                    inn,
                    model=synapses_inn,
                    on_pre=synapses_action,
                    method="exponential_euler")

inn_inn_synapses.connect(condition='i != j', p=0.01)

inn_inn_synapses.tau_inact = 1.5 * ms
inn_inn_synapses.A_SE = 100 * pA
inn_inn_synapses.tau_rec = 130 * ms
inn_inn_synapses.U_SE = 0.03
inn_inn_synapses.tau_facil = 530 * ms


"""RECORDINGS AND SIMULATION"""
pc_vs = StateMonitor(pc, 'vs', record=True)

inn_vs = StateMonitor(inn, 'vs', record=True)

bc_vs = StateMonitor(bc, 'vs', record=True)

pc_rate = PopulationRateMonitor(pc)

pc_spikes = SpikeMonitor(pc)

inn_spikes = SpikeMonitor(inn)

bc_spikes = SpikeMonitor(bc)

input_spikes = SpikeMonitor(P)

dt = 0.1*ms

defaultclock.dt = dt

time = np.arange(0*ms, 1500*ms, dt)

stim = np.zeros(time.shape[0]) * pA

# stim[5000:10000] = 100 * pA

stim = TimedArray(stim, dt=dt)

run(1500*ms)

plt.figure()
plt.plot(pc_vs.t, pc_vs.vs[0])
plt.title("pc")

plt.figure()
plt.plot(inn_vs.vs[0])
plt.title("inn")

plt.figure()
plt.plot(bc_vs.vs[0])
plt.title("bc")

plt.figure()
plot(pc_rate.t/ms, pc_rate.rate/Hz)

isi_list = []
for x in np.unique(pc_spikes.i):
    isi_list.append(asarray(np.diff(pc_spikes.t[pc_spikes.i == x])))
    
isi_array = np.concatenate(isi_list)

plt.figure()
plt.hist(isi_array, bins=50)

plt.figure()
plt.scatter(input_spikes.t, input_spikes.i, marker='|')



    





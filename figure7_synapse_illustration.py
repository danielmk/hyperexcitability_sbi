# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:14:50 2023

@author: Daniel
"""

from brian2 import *
import matplotlib.pyplot as plt
import sys
import pdb
import typing
import parameters

def synapse_tester(params : dict) -> dict:

    """DEFINE THE CONSTANTS OF THE NETWORK"""
    # NEURON TYPE: Adaptive Exponential
    # TODO Gotta define each neuron separtely with different synaptic currents
    # pc_synaptic_terms = "I_inp + I_pcs + I_fb + I_in"
    model_pc = '''dvs/dt = (-gL*(vs-EL)+gL*DeltaT*exp((vs-VT)/DeltaT)-w+I_total)/C : volt
               dw/dt = (a*(vs-EL)-w)/tau_w : amp
               I_total = I_inp + I_pc - I_bc - I_inn + I_gap : amp
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
               I_inn : amp
               I_gap : amp'''
    
    model_reset='''vs=Vr
                   w=w+b
                '''
    
    cell_types = ['pc', 'inn', 'bc']
    
    # SYNAPSE TYPE: Tsodyks-Markram
    # CELL Types: PC, FB-IN, IN
    cell_dict = {}
    for ct in cell_types:
        n = params[f'{ct}_n']
        cn = NeuronGroup(n, model_pc, threshold='vs>0.0*mV', reset=model_reset, method='euler')
        cell_dict[ct] = cn
        cell_dict[ct].C = params[f'{ct}_C']
        cell_dict[ct].gL = params[f'{ct}_gL']
        cell_dict[ct].EL = params[f'{ct}_EL']
        cell_dict[ct].VT = params[f'{ct}_VT']
        cell_dict[ct].DeltaT = params[f'{ct}_DeltaT']
        cell_dict[ct].tau_w = params[f'{ct}_tau_w']
        cell_dict[ct].a = params[f'{ct}_a']
        cell_dict[ct].b = params[f'{ct}_b']
        cell_dict[ct].Vr = params[f'{ct}_Vr']
        cell_dict[ct].vs = np.random.normal(cell_dict[ct].EL[0], 3*mvolt,cell_dict[ct].vs.shape[0]) * volt
    
    times = np.arange(100,600,50)*ms
    inp = SpikeGeneratorGroup(N=1, indices=[0] * len(times), times=times)
    cell_dict['inp'] = inp

    # Creating Synapses
    # From: https://brian2.readthedocs.io/en/latest/examples/frompapers.Tsodyks_Pawelzik_Markram_1998.html
    synapses_model =    """
                        dx/dt =  z/tau_rec   : 1 (clock-driven) # recovered
                        dy/dt = -y/tau_inact : 1 (clock-driven) # active
                        A_SE : ampere
                        U_SE : 1
                        tau_inact : second
                        tau_rec : second
                        z = 1 - x - y : 1 # inactive
                        """

    synapses_inp = synapses_model + f"I_inp_post = A_SE*y : ampere (summed)"
    synapses_pc = synapses_model + f"I_pc_post = A_SE*y : ampere (summed)"
    synapses_bc = synapses_model + f"I_bc_post = A_SE*y : ampere (summed)"
    synapses_inn = synapses_model + f"I_inn_post = A_SE*y : ampere (summed)"              
                        
    synapses_action =   """
                        y += u*x # important: update y first
                        x += -u*x
                        """
    
    synapse_types = ['inp_pc', 'inp_bc', 'pc_pc', 'pc_inn', 'pc_bc', 'bc_pc',
                     'bc_bc', 'inn_pc', 'inn_bc', 'inn_inn']
    
    synapses_dict = {}
    
    for s in synapse_types:
        pre, post = s.split('_')
        
        curr_synapse_eq = eval(f'synapses_{pre}')
        
        if params[f'{pre}_{post}_tau_facil'] != 0:
            curr_synapse_eq += '''du/dt = -u/tau_facil : 1 (clock-driven)
                                tau_facil : second'''
            curr_synapses_action = """
                                    u += U_SE*(1-u)
                                    y += u*x # important: update y first
                                    x += -u*x
                                    """
        else:
            curr_synapses_action = """
                                    y += U_SE*x # important: update y first
                                    x += -U_SE*x
                                    """
    
        cs = Synapses(inp,
                        cell_dict[post],
                        model=curr_synapse_eq,
                        on_pre=curr_synapses_action,
                        method="euler")
    
        if pre == post:
            cs.connect(condition='i != j', p=params[f'{pre}_{post}_p'])
        else:
            cs.connect(p=params[f'{pre}_{post}_p'])
    
        cs.tau_inact = params[f'{pre}_{post}_tau_inact']
        cs.A_SE = params[f'{pre}_{post}_A_SE']
        cs.tau_rec = params[f'{pre}_{post}_tau_rec']
        cs.U_SE = params[f'{pre}_{post}_U_SE']
        if params[f'{pre}_{post}_tau_facil'] != 0:
            cs.tau_facil = params[f'{pre}_{post}_tau_facil']
        
        synapses_dict[s] = cs
        
        # Start fully recovered
        cs.x = 1
        
        curr_monitor = StateMonitor(cs, f'I_{pre}_post', record=True)
    
    # Creating the gap junction from https://brian2.readthedocs.io/en/latest/examples/synapses.gapjunctions.html
    gap_model = '''
                w_gap : siemens
                I_gap_post = w_gap * (vs_pre - vs_post) : amp (summed)
                '''
    synapses_dict['bc_bc_gap'] = Synapses(cell_dict['bc'],
                                         cell_dict['bc'],
                                         gap_model)
    synapses_dict['bc_bc_gap'].connect(p=params['bc_bc_gap_p'])
    synapses_dict['bc_bc_gap'].w_gap = params['bc_bc_gap_w']
    
    network_dict = cell_dict | synapses_dict
    
    return network_dict

    

if __name__ == '__main__':
    # set_device('cpp_standalone')

    start_scope()
    
    params = parameters.baseline

    nw_dict = synapse_tester(params)

    pc = nw_dict['pc']
    inn = nw_dict['inn']
    bc = nw_dict['bc']
    inp = nw_dict['inp']
    inp_pc = nw_dict['inp_pc']
    inp_bc = nw_dict['inp_bc']
    pc_pc = nw_dict['pc_pc']
    pc_inn = nw_dict['pc_inn']
    pc_bc = nw_dict['pc_bc']
    bc_pc = nw_dict['bc_pc']
    bc_bc = nw_dict['bc_bc']
    inn_pc = nw_dict['inn_pc']
    inn_bc = nw_dict['inn_bc']
    inn_inn = nw_dict['inn_inn']
    bc_bc_gap = nw_dict['bc_bc_gap']

    inp_pc_monitor = StateMonitor(pc, 'I_inp', record=True)
    inp_bc_monitor = StateMonitor(bc, 'I_inp', record=True)
    pc_pc_monitor = StateMonitor(pc, 'I_pc', record=True)
    pc_inn_monitor = StateMonitor(inn, 'I_pc', record=True)
    pc_bc_monitor = StateMonitor(bc, 'I_pc', record=True)
    bc_pc_monitor = StateMonitor(pc, 'I_bc', record=True)
    bc_bc_monitor = StateMonitor(bc, 'I_bc', record=True)
    inn_pc_monitor = StateMonitor(pc, 'I_inn', record=True)
    inn_bc_monitor = StateMonitor(bc, 'I_inn', record=True)
    inn_inn_monitor = StateMonitor(inn, 'I_inn', record=True)
    
    run(2000*ms)

    synapse_types = ['inp_pc', 'inp_bc', 'pc_pc', 'pc_inn', 'pc_bc', 'bc_pc',
                     'bc_bc', 'inn_pc', 'inn_bc', 'inn_inn']
    
    fig, ax = plt.subplots(2, 5)
    
    ax_flat = ax.flatten()
    
    for idx, s in enumerate(synapse_types):
        pre, post = s.split('_')
        ax_flat[idx].plot(eval(f'{s}_monitor.t'), eval(f'{s}_monitor.I_{pre}.sum(axis=0)'))
        ax_flat[idx].set_title(f'Synapse: {s}\n' +
                               f'tau_inact: {nw_dict[f"{pre}_{post}"].tau_inact[0]}; ' + 
                               f'tau_rec: {nw_dict[f"{pre}_{post}"].tau_rec[0]}; ' + 
                               f'U_SE: {nw_dict[f"{pre}_{post}"].U_SE[0]}; ' + 
                               f'tau_facil: {params[f"{pre}_{post}_tau_facil"]}')

    for x in ax_flat:
        x.set_ylabel("Current (yamp)")
        x.set_xlabel("Time (s)")
        

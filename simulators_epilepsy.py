# -*- coding: utf-8 -*-
"""

"""

import brian2 as b2
import numpy as np
import priors
import torch
import summary_statistics
import pdb
import thetas_vetted

class Simulator():
    name = "baseline_conductance_based_01"
    def __init__(self,
                 dt : b2.units.fundamentalunits.Quantity,
                 duration : b2.units.fundamentalunits.Quantity,
                 network_constants : dict,
                 prior : dict):
        
        self.constants = network_constants
        self.dt = dt
        self.duration = duration
        self.prior = prior
        self.prior_names = list(prior['low'].keys())
        self.prior_units = priors.get_base_units(prior['low'].values())

    def make_network(self, params : dict):
        # (The rest of your function remains unchanged)
        net = b2.Network()

        """DEFINE THE CONSTANTS OF THE NETWORK"""
        # NEURON TYPE: Adaptive Exponential Integrate and Fire
        model_pc = '''dvs/dt = (-gL*(vs-EL)+gL*DeltaT*exp((vs-VT)/DeltaT)-w+I_total)/C : volt
                   dw/dt = (a*(vs-EL)-w)/tau_w : amp
                   I_total = - g_inp * (vs - EEx) - g_pc * (vs - EEx) - g_nin * (vs - EIn) - g_ain * (vs - EIn) + I_gap : amp
                   tau_w : second
                   b : amp
                   DeltaT : volt
                   a : siemens
                   C : farad
                   gL : siemens
                   EL : volt
                   VT : volt
                   Vr : volt
                   EIn : volt
                   EEx : volt
                   g_inp : siemens
                   g_pc: siemens
                   g_nin : siemens
                   g_ain : siemens
                   I_gap : amp'''

        model_reset='''vs=Vr
                       w=w+b
                    '''

        cell_types = ['pc', 'ain', 'nin']
        
        # SYNAPSE TYPE: Tsodyks-Markram
        # CELL Types: PC, FB-IN, IN
        cell_dict = {}
        for ct in cell_types:
            n = params[f'{ct}_n']
            cn = b2.NeuronGroup(n, model_pc, threshold='vs>0.0*mV', reset=model_reset, method='euler', name=ct)
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
            cell_dict[ct].EIn = -70 * b2.mV
            cell_dict[ct].EEx = 0 * b2.mV
            cell_dict[ct].vs = np.random.uniform(cell_dict[ct].EL[0] - 3 * b2.mV, cell_dict[ct].EL[0] + 3 *b2.mV, cell_dict[ct].vs.shape[0]) * b2.volt

        inp = b2.PoissonGroup(400, np.random.uniform(10*b2.Hz, 30*b2.Hz, 400)*b2.Hz,name='inp')
        cell_dict['inp'] = inp

        # Creating Synapses
        # From: https://brian2.readthedocs.io/en/latest/examples/frompapers.Tsodyks_Pawelzik_Markram_1998.html
        synapses_model =    """
                            dx/dt =  z/tau_rec   : 1 (clock-driven) # recovered
                            dy/dt = -y/tau_inact : 1 (clock-driven) # active
                            A_SE : siemens
                            U_SE : 1
                            tau_inact : second
                            tau_rec : second
                            z = 1 - x - y : 1 # inactive
                            """

        synapses_inp = synapses_model + f"g_inp_post = A_SE*y : siemens (summed)"
        synapses_pc = synapses_model + f"g_pc_post = A_SE*y : siemens (summed)"
        synapses_nin = synapses_model + f"g_nin_post = A_SE*y : siemens (summed)"
        synapses_ain = synapses_model + f"g_ain_post = A_SE*y : siemens (summed)"              

        synapses_action =   """
                            y += u*x # important: update y first
                            x += -u*x
                            """

        synapse_types = ['inp_pc', 'inp_nin', 'pc_pc', 'pc_ain', 'pc_nin', 'nin_pc',
                         'nin_nin', 'ain_pc', 'ain_nin', 'ain_ain']

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

            cs = b2.Synapses(cell_dict[pre],
                            cell_dict[post],
                            model=curr_synapse_eq,
                            on_pre=curr_synapses_action,
                            method="euler",
                            name=s)
        
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
            
            cs.x = 0.1

            synapses_dict[s] = cs

        # Creating the gap junction from https://brian2.readthedocs.io/en/latest/examples/synapses.gapjunctions.html
        gap_model = '''
                    w_gap : siemens
                    I_gap_post = w_gap * (vs_pre - vs_post) : amp (summed)
                    '''
        synapses_dict['nin_nin_gap'] = b2.Synapses(cell_dict['nin'],
                                             cell_dict['nin'],
                                             gap_model,
                                             name='nin_nin_gap')
        synapses_dict['nin_nin_gap'].connect(p=params['nin_nin_gap_p'])
        synapses_dict['nin_nin_gap'].w_gap = params['nin_nin_gap_w']
        
        network_dict = cell_dict | synapses_dict
        
        net.add(network_dict.values())
        
        return net
    
    def theta_merge(self, theta : dict):
        """CREATE PARAMS DICT AND ADD UNITS"""
        theta_units = [x * self.prior_units[idx] for idx, x in enumerate(np.array(theta))]
        params_dict = dict(zip(self.prior_names, theta_units))

        """MERGE PARAMS AND CONSTANTS INTO A DICT THAT FULLY DEFINES THE NETWORK"""
        params_full = {**params_dict, **self.constants}
        
        return params_full
    
    def get_output(self, nw, spikes, volt):
        mean_rate = summary_statistics.mean_rate(spikes, nw['pc'].N, self.duration)
        mean_entropy = summary_statistics.mean_entropy(spikes, np.arange(0, self.duration, 0.001)*b2.second)
        f, psd = summary_statistics.psd(nw['pc_vs'], self.dt)
        theta_power = psd[(f > 8*b2.Hz) & (f < 12*b2.Hz)].mean()
        gamma_power = psd[(f > 30*b2.Hz) & (f < 100*b2.Hz)].mean()
        fast_power = psd[(f > 100*b2.Hz) & (f < 150*b2.Hz)].mean()
        correlation = summary_statistics.average_correlation(spikes, kernel_width=1*b2.ms, cells=0.1)
        cv = summary_statistics.coefficient_of_variation(spikes)

        return torch.tensor([b2.asarray(mean_rate).mean(), mean_entropy, theta_power, gamma_power, fast_power, correlation, cv])
    
    def run(self, theta : torch.Tensor):
        # Merge theta and constants
        params = self.theta_merge(theta)

        # Make network
        nw = self.make_network(params)
        
        # Setup recordings
        pc_spikes = b2.SpikeMonitor(nw['pc'], name='pc_spikes')
        pc_vs = b2.StateMonitor(nw['pc'], 'vs', record=True, name='pc_vs')
        
        nw.add(pc_spikes, pc_vs)
        
        # Run Network
        b2.defaultclock.dt = self.dt
    
        duration = self.duration
        
        nw.run(duration)
    
        output = self.get_output(nw, nw['pc_spikes'], nw['pc_vs'])
    
        return output
    
    def run_ampa_gaba(self, theta : torch.Tensor):
        # Merge theta and constants
        params = self.theta_merge(theta)

        # Make network
        nw = self.make_network(params)
        
        # Setup recordings
        pc_spikes = b2.SpikeMonitor(nw['pc'], name='pc_spikes')
        pc_vs = b2.StateMonitor(nw['pc'], 'vs', record=True, name='pc_vs')
        
        nw.add(pc_spikes, pc_vs)
        
        # Run Network
        b2.defaultclock.dt = self.dt
    
        duration = self.duration
        
        nw.run(duration)
    
        output = self.get_output(nw, nw['pc_spikes'], nw['pc_vs'])
    
        return output

        # return torch.tensor([b2.asarray(mean_rate).mean(), mean_entropy, theta_power, gamma_power, fast_power, correlation, cv])

    def run_evaluate(self, theta : torch.Tensor):
        # Merge theta and constants
        params = self.theta_merge(theta)

        # Make network
        nw = self.make_network(params)
        
        #MONITORS FOR EVALUATION
        pc_spikes = b2.SpikeMonitor(nw['pc'], name='pc_spikes')
        pc_vs = b2.StateMonitor(nw['pc'], 'vs', record=True, name='pc_vs')
        ain_vs = b2.StateMonitor(nw['ain'], 'vs', record=True, name='ain_vs')
        nin_vs = b2.StateMonitor(nw['nin'], 'vs', record=True, name='nin_vs')
        pc_rate = b2.PopulationRateMonitor(nw['pc'], name='pc_rate')
        ain_rate = b2.PopulationRateMonitor(nw['ain'], name='ain_rate')
        nin_rate = b2.PopulationRateMonitor(nw['nin'], name='nin_rate')
        input_rate = b2.PopulationRateMonitor(nw['inp'], name='input_rate')
        ain_spikes = b2.SpikeMonitor(nw['ain'], name='ain_spikes')
        nin_spikes = b2.SpikeMonitor(nw['nin'], name='nin_spikes')
        input_spikes = b2.SpikeMonitor(nw['inp'], name='input_spikes')
        g_inp_m = b2.StateMonitor(nw['pc'], 'g_inp', record=True, name='g_inp_m')
        g_pc_m = b2.StateMonitor(nw['pc'], 'g_pc', record=True, name='g_pc_m')
        g_nin_m = b2.StateMonitor(nw['pc'], 'g_nin', record=True, name='g_nin_m')
        g_ain_m = b2.StateMonitor(nw['pc'], 'g_ain', record=True, name='g_ain_m')
        
        nw.add(pc_spikes, pc_vs, ain_vs, nin_vs, pc_rate, ain_rate, nin_rate, input_rate, ain_spikes, nin_spikes, input_spikes, g_inp_m, g_pc_m, g_nin_m, g_ain_m)
        
        # Run Network
        b2.defaultclock.dt = self.dt
    
        duration = self.duration

        nw.run(duration)

        output = self.get_output(nw, nw['pc_spikes'], nw['pc_vs'])
        
        return output, nw



b2.prefs.codegen.target = 'numpy'

params = priors.baseline_epilepsy['low']
param_names = list(params.keys())
param_units = priors.get_base_units(params.values())

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['svg.fonttype'] = 'none'
    """CREATE THE SIMULATOR"""
    # from brian2 import *
    # constants_dict = constants.dynamics_constant
    prior = priors.baseline_epilepsy
    
    
    
    prior_tensor = priors.prior_dict_to_tensor(prior)
    duration = 1*b2.second
    _dt = 0.1*b2.ms  # Needs underscore not to conflict with b2 internal dt

    simulator = Simulator(dt=_dt,
                                    duration=duration,
                                    network_constants=prior['constants'],
                                    prior=prior)

    """CREATE A THETA"""
    theta = thetas_vetted.baseline
    
    
    theta_tensor = thetas_vetted.theta_dict_to_tensor(theta)
    
    """RUN THETA"""
    output, nw = simulator.run_evaluate(theta_tensor)

    plt.figure()
    plt.plot(nw['pc_vs'].t, nw['pc_vs'].vs[0,:])

    # output, nw = simulator.run(theta_tensor)

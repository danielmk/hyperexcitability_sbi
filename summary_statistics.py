# -*- coding: utf-8 -*-
"""
summary_statistics.py implements functions that calculate summary statistics
from simulator output. Some functions are helpers and do not directly return
a single statistic of the network (e.g. relative_amplitude)
"""

# from brian2 import *
import brian2 as b2
import numpy as np
import scipy.stats
from numpy.fft import fft, fftshift, fftfreq
from scipy.signal import windows, periodogram, welch

def mean_rate(spike_monitor, n_neurons, duration):
    
    return spike_monitor.num_spikes / (n_neurons * duration)

def mean_entropy(spike_monitor, bins):
    entropy_list = []
    for x in spike_monitor.spike_trains().values():
        ch = np.histogram(np.diff(x), bins=bins)
        ce = scipy.stats.entropy(pk=ch[0])
        entropy_list.append(ce)

    return np.nanmean(np.array(entropy_list))

def relative_amplitude(vs_monitor,  dt, lowest_f, highest_f):
    sp = fftshift(fft(vs_monitor.vs.mean(axis=0)))
    freq = fftshift(fftfreq(len(vs_monitor.vs.mean(axis=0)), b2.asarray(dt)))
    
    indices_f = np.argwhere(np.logical_and(freq > lowest_f, freq < highest_f))
    
    amp = np.abs(sp)
    amp = amp[indices_f]
    relative_amp = amp / amp.sum()
    
    freq = freq[indices_f]

    # pdb.set_trace()

    return freq[:, 0], relative_amp[:, 0]

def psd(vs_monitor, dt):
    mean_membrane_voltage = vs_monitor.vs.mean(axis=0)
    sampling_rate = 1 / dt
    f, psd = periodogram(mean_membrane_voltage, sampling_rate, scaling='density')
    return f, psd

def psd_ampa_gaba(signal, dt):
    sampling_rate = 1 / dt
    f, psd = welch(signal, sampling_rate, nperseg=2000, noverlap=1500, scaling='density')
    return f, psd

def psd_lfp_collab(signal, dt):
    sampling_rate = 1 / dt
    # f, psd = welch(signal, sampling_rate, nperseg=1000, noverlap=500, scaling='density')
    f, psd = welch(signal, sampling_rate, nperseg=int(sampling_rate) / 2, scaling='density')
    return f, psd 

def total_power(vs_monitor, dt, low=4*b2.Hz, high=12*b2.Hz, lowest=1*b2.Hz, highest=200*b2.Hz):
    """Bounds from Butler & Paulsen 2015
    Theta: 4Hz - 12Hz
    Gamma: 30Hz - 100Hz
    Fast: 100Hz - 150Hz
    """
    freq, relative_amp = relative_amplitude(vs_monitor, dt, lowest, highest)

    bounds = np.argwhere(np.logical_and(freq >= low, freq <= high))
    
    power = relative_amp[bounds].sum()

    return power

def total_power_freq_amp(freq, relative_amp, dt, low=4*b2.Hz, high=12*b2.Hz, lowest=1*b2.Hz, highest=200*b2.Hz):
    """Bounds from Butler & Paulsen 2015
    Theta: 4Hz - 12Hz
    Gamma: 30Hz - 100Hz
    Fast: 100Hz - 150Hz
    """

    bounds = np.argwhere(np.logical_and(freq >= low, freq <= high))
    
    power = relative_amp[bounds].sum()

    return power

def average_correlation(spike_monitor, kernel_width=10*b2.ms, cells=0.1):
    n_units = spike_monitor.source.N
    dt = spike_monitor.source.dt
    duration = spike_monitor.source.t
    
    n_selection = int(n_units * cells)
    
    cells_selection = np.random.choice(n_units, n_selection, replace=False)
    # cells.sort()
    
    cells_idc = np.in1d(spike_monitor.i, cells_selection)
    
    units = spike_monitor.i[cells_idc]
    times = spike_monitor.t[cells_idc]
    # pdb.set_trace()
    # curr_binary = units_times_to_binary(spike_monitor.i, spike_monitor.t, n_units=n_units, dt=dt, total_time=duration)
    curr_binary = units_times_to_binary(units, times, n_units=n_units, dt=dt, total_time=duration)
    
    curr_binary = np.array(curr_binary, dtype=int)
    
    gaussian_sigma = kernel_width / dt
    gaussian_width = gaussian_sigma * 10

    kernel = windows.gaussian(b2.asarray(gaussian_width), b2.asarray(gaussian_sigma))

    kernel = kernel / kernel.sum()

    curr_binary_convolved = np.array([np.convolve(x, kernel, 'full') for x in curr_binary[cells_selection, :]])
    pr = np.corrcoef(x=curr_binary_convolved)
    ut_idc = np.triu_indices(pr.shape[0], 1)
    mean_pearson_r = np.nanmean(pr[ut_idc])
    
    # pdb.set_trace()
    
    return mean_pearson_r

def burst_ratio(spike_monitor, cutoff=5*b2.ms):

    burst_ratio_list = []
    for st in spike_monitor.spike_trains().values():
        diff_forward = np.diff(st, append=st[-1]+(cutoff+1*b2.ms))
        diff_backward = np.diff(st, append=-(cutoff+1*b2.ms))
        
        burst_spikes = np.logical_or(diff_forward*b2.second<5*b2.ms,  diff_backward*b2.second<5*b2.ms).sum()
        burst_ratio_list.append(burst_spikes/st.shape[0])
    
    return np.array(burst_ratio_list).mean()

def coefficient_of_variation(spike_monitor):
    
    m = []
    sd = []
    for x in spike_monitor.spike_trains().values():
        isi = np.diff(x)
        m.append(isi.mean())
        sd.append(np.std(isi))

    return np.nanmean(np.array(sd) / np.array(m))

def n_active(spike_trains):
    n_spikes = np.array([len(x) for x in spike_trains])
    n_active = (n_spikes > 0).sum()
    
    return n_active
    

"""HELPER"""
def units_times_to_binary(units, times, n_units=784, dt=0.0001, total_time=0.1):
    """Convert spike times and their presynaptic unit to binary matrix"""
    binary = np.zeros((n_units, int(total_time / dt)), dtype=bool)

    times_to_idc = np.array(times / dt, dtype = int)

    binary[units, times_to_idc] = 1

    return binary


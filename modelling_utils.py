import torch
import numpy as np
import timeit
import sys

inf = sys.float_info.max

#-------------------------------------------------------------------------------
def indices_array(m,n):
    return np.indices((m,n)).transpose(1,2,0)

#-------------------------------------------------------------------------------
def connections_all_to_all(dim_from, dim_to, weights):
    connections = []
    for i in range(dim_from):
        for j in range(dim_to):
            connections.append((i, j, weights[i, j]))

    return connections

#-------------------------------------------------------------------------------
def pt_to_np(pytorch_tensor):
    t = torch.load(pytorch_tensor, map_location=torch.device('cpu'))
    return t.cpu().detach().numpy()

#-------------------------------------------------------------------------------
def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    idx = x<thr
    x = np.clip(x,thr+epsilon,1e9)
    T = tau*np.log(x/(x-thr))
    T[idx] = tmax
    return T

#-------------------------------------------------------------------------------
def stack_spikes(x, offset=150):
    tmax = 100
    all_firing_times = current2firing_time(x, tmax=tmax)
    m, n, k = np.shape(all_firing_times)
    i = np.arange(m)
    i = np.reshape(i, (m, 1))

    all_firing_times = np.reshape(all_firing_times, (m, (n*k)))

    idx = (all_firing_times >= tmax).T

    all_firing_times = np.add(all_firing_times, (i * offset)).T

    firing_list = [[]] * (n * k)

    for i in range(n*k):
        single_neuron_times = all_firing_times[i][np.invert(idx[i])]
        firing_list[i] = single_neuron_times.tolist()

    return firing_list

#-------------------------------------------------------------------------------
def count_spikes(spiketrain):
    count = 0
    for i in range(np.shape(spiketrain)[0]):
        count += len(spiketrain[i])
    return count

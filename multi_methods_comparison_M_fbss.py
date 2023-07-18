# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:54:56 2021

@author: DaiHoang
"""
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.model_selection import train_test_split

from DNN_Nsignals_training_SNR import *

N_antenas = 15
SNRdB = 10
N_snapshots = 1000

def data_shuffle(X_train, y_train, N_samples=None):
    if N_samples is None:
        N_samples = int(len(X_train))
    shuf_idx = np.random.permutation(X_train.shape[0])
    if N_samples is not None:
        if type(N_samples) == int:
            shuf_idx = shuf_idx[:N_samples]
        elif type(N_samples) == float:
            shuf_idx = shuf_idx[:int(len(X_train)*N_samples)]
    return X_train[shuf_idx], y_train[shuf_idx]

    
#feature_input = 'fbss'
SNRdB_arr = [-10, 0, 10, 20, 30]
delta_SNRdB = 4 # Constant
#SNRdB_arr = [10]
# M_fbss = [7, 8, 9, 10, 11] # For 15 antennas
# M_fbss = [7, 13, 19] # For M0 = 25
M_fbss = [7, 11, 13] # For M0 = 25
#N_snapshots_arr = [10, 100, 1000, 10000, 100000] 

acc_fbss = []
acc_toep = np.zeros(len(SNRdB_arr))

AIC_toep = []
MDL_toep = []
#for s in range(len(N_snapshots_arr)):
for s in range(len(SNRdB_arr)):
    SNRdB = SNRdB_arr[s]
    #N_snapshots = N_snapshots_arr[s]

    # ECNet with various M
    print('\n<=========== ECNet ============>')
    acc_tmp = []
    for M in M_fbss:
        acc_tmp.append(testing('fbss', N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=M))
    acc_fbss.append(acc_tmp)
    print('\n<=========== Proposed ============>')
    acc_toep[s] = testing('toep', N_antenas, SNRdB, delta_SNRdB, N_snapshots)
        
acc_fbss = np.array(acc_fbss)
hist = {
        'acc_toep': acc_toep,
        'acc_fbss': acc_fbss
        }
#savemat('acc_comparison_snapshots_%d.mat'%SNRdB, hist)
savemat('acc_comparison_SNR_M_fbss_%d_snaps.mat'%N_snapshots, hist)

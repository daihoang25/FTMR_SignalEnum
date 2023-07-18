# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:22:33 2021

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
#N_snapshots = 10

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

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def EGM(eigens):
    M = len(eigens)
    threshold = (eigens[0] - eigens[-1])/(M - 1)
    hist = []
    for i in range(M-1):
        delta_i = eigens[i] - eigens[i+1]
        if delta_i <= threshold:
            hist.append(i)
    
    y_pred = group_consecutives(hist)[-1][0] + 1
    return hist, y_pred

def AIC(eigens):
    M = len(eigens)
    aic = []
    L_arr = []
    for k in range(M):
        numer = np.prod(eigens[k:])
        numer = np.power(numer, 1/(M - k))
        denom = np.sum(eigens[k:])/(M - k)
        L = -N_snapshots*(M - k)*np.log(numer/denom)
        L_arr.append(L)
        aic.append(L + k*(2*M - k))
    return np.argmin(aic)

def improved_AIC(eigens):
    M = len(eigens)
    aic = []
    for k in range(M):
        numer = np.prod(eigens[k:])
        numer = np.power(numer, 1/(M - k))
        denom = np.sum(eigens[k:])/(M - k)
        L = -N_snapshots*(M - k)*np.log(numer/denom)
        C = 2 + 0.001*np.log(N_snapshots)
        aic.append(L + 2*C*k*(M + 1 - (k + 1)/2))
    return np.argmin(aic)

def MDL(eigens):
    M = len(eigens)
    mdl = []
    for k in range(M):
        numer = np.prod(eigens[k:])
        numer = np.power(numer, 1/(M - k))
        denom = np.sum(eigens[k:])/(M - k)
        L = -N_snapshots*(M - k)*np.log(numer/denom)
        mdl.append(L + 0.5*k*(2*M - k)*np.log(N_snapshots))
    return np.argmin(mdl)

def LS_MDL(eigens):
    M = len(eigens)
    mdl = []
    rho = []
    for k in range(M):
        # supplimental elements
        to_k = np.mean(eigens[k:])
        num_alpha = np.sum(eigens[k:]**2) + (np.sum(eigens[k:]))**2
        den_alpha = (N_snapshots + 1)*(np.sum(eigens[k:]**2) - (np.sum(eigens[k:]))**2/len(eigens[k:]))
        alpha_k = num_alpha/den_alpha
        beta_k = min(alpha_k, 1)
        rho_k = beta_k*to_k + (1 - beta_k)*eigens[k:]
        rho.append(np.sum(rho_k))
        # LS-MDL main formula
        numer = np.mean(rho_k)
        denom = np.power(np.prod(rho_k), 1/len(rho_k))
        L = N_snapshots*len(rho_k)*np.log(numer/denom)
        mdl.append(L + 0.5*k*(k - 1)*np.log(N_snapshots))
    return np.argmin(mdl)

def TopSVD(eigens):
    M = len(eigens)
    ratio = []
    for k in range(M-1):
        ratio.append(eigens[k]/eigens[k+1])
    return np.argmax(ratio) + 1

def SORTE(eigens):
    M = len(eigens)
    delta = eigens[0:-1] - eigens[1:]
    S = np.zeros(M - 2)
    for K in range(len(S)):
        tmp_inner = np.mean(delta[K:M-1])
        var_K = np.mean((delta[K:M-1] - tmp_inner)**2)
        if var_K == 0:
            S[K] = np.Inf
        else:
            tmp_inner = np.mean(delta[K+1:M-1])
            var_K1 = np.mean((delta[K+1:M-1] - tmp_inner)**2)
            S[K] = var_K1/var_K
    return np.argmin(S[:-1]) + 1
    
#feature_input = 'fbss'
#SNRdB_arr = [-10, 0, 10, 20, 30]
N_snapshots_arr = [10, 100, 1000, 10000, 100000] 
delta_SNRdB = 4

#SNRdB_arr = [20]
acc_aic_fbss = np.zeros(len(N_snapshots_arr))
acc_mdl_fbss = np.zeros(len(N_snapshots_arr))
acc_ls_mdl_fbss = np.zeros(len(N_snapshots_arr))
acc_im_aic_fbss = np.zeros(len(N_snapshots_arr))
acc_topsvd_fbss = np.zeros(len(N_snapshots_arr))
acc_sorte_fbss = np.zeros(len(N_snapshots_arr))

acc_aic_toep = np.zeros(len(N_snapshots_arr))
acc_mdl_toep = np.zeros(len(N_snapshots_arr))
acc_im_aic_toep = np.zeros(len(N_snapshots_arr))
acc_topsvd_toep = np.zeros(len(N_snapshots_arr))
acc_sorte_toep = np.zeros(len(N_snapshots_arr))

#acc_ecnet = np.zeros(len(SNRdB_arr))
acc_ecnet = []
acc_toep = np.zeros(len(N_snapshots_arr))

AIC_toep = []
MDL_toep = []
for s in range(len(N_snapshots_arr)):
#for s in range(len(SNRdB_arr)):
    #SNRdB = SNRdB_arr[s]
    N_snapshots = N_snapshots_arr[s]
    print('\n<=========== Importing dataset (SNR = %ddB, delta = %ddB, N_antenas = %d, N_snapshots = %d) ============>'\
          %(SNRdB, delta_SNRdB, N_antenas, N_snapshots))
    
    print('\n<=========== AIC, MDL with FBSS method ============>')
    model_path, checkpoint_path_Nsignals, testset_path = path_reader('fbss', N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=11, training=False)
    data = np.load(testset_path, allow_pickle=True).item()
    X_test, y_test = dataset_preparation(data, SNRdB, feature_input='fbss', normalized_input=False, data_shuffling=False)
    
    N_samples = len(X_test)
    for i in range(N_samples):
        N_signals = int(np.where(y_test[i]==1)[0])
        acc_aic_fbss[s] += (N_signals == AIC(X_test[i]))/N_samples
        acc_mdl_fbss[s] += (N_signals == MDL(X_test[i]))/N_samples
        acc_ls_mdl_fbss[s] += (N_signals == MDL(X_test[i]))/N_samples
        acc_im_aic_fbss[s] += (N_signals == improved_AIC(X_test[i]))/N_samples
        acc_topsvd_fbss[s] += (N_signals == TopSVD(X_test[i]))/N_samples
        acc_sorte_fbss[s] += (N_signals == SORTE(X_test[i]))/N_samples
        
    print('\n<=========== AIC, MDL with Toep method ============>')
    model_path, checkpoint_path_Nsignals, testset_path = path_reader('toep', N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=None, training=False)
    data = np.load(testset_path, allow_pickle=True).item()
    X_test, y_test = dataset_preparation(data, SNRdB, feature_input='toep', normalized_input=False, data_shuffling=False)
    N_samples = len(X_test)
    
    sorte_idx = []
    for i in range(N_samples):
        N_signals = int(np.where(y_test[i]==1)[0])
        acc_aic_toep[s] += (N_signals == AIC(X_test[i]))/N_samples
        acc_mdl_toep[s] += (N_signals == LS_MDL(X_test[i]))/N_samples
        acc_im_aic_toep[s] += (N_signals == improved_AIC(X_test[i]))/N_samples
        acc_topsvd_toep[s] += (N_signals == TopSVD(X_test[i]))/N_samples
        acc_sorte_toep[s] += (N_signals == SORTE(X_test[i]))/N_samples
        
        if (N_signals != SORTE(X_test[i])):
            sorte_idx.append(i)
        
        AIC_toep.append(AIC(X_test[i]))
        MDL_toep.append(MDL(X_test[i]))

    # ECNet with various M
    print('\n<=========== ECNet ============>')
    #acc_ecnet[SNRdB] = []
    acc_tmp = []
    #for M in [7, 9, 11]:
    for M in [11]:
        acc_tmp.append(testing('fbss', N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=M))
    acc_ecnet.append(acc_tmp)
    print('\n<=========== Proposed ============>')
    acc_toep[s] = testing('toep', N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=None)
    
acc_ecnet = np.array(acc_ecnet)
hist = {
        'acc_aic_fbss': acc_aic_fbss,
        'acc_mdl_fbss': acc_mdl_fbss,
        'acc_ls_mdl_fbss': acc_ls_mdl_fbss,
        'acc_im_aic_fbss': acc_im_aic_fbss,
        'acc_topsvd_fbss': acc_topsvd_fbss,
        'acc_sorte_fbss': acc_sorte_fbss,
        'acc_aic_toep': acc_aic_toep,
        'acc_mdl_toep': acc_mdl_toep,
        'acc_im_aic_toep': acc_im_aic_toep,
        'acc_topsvd_toep': acc_topsvd_toep,
        'acc_sorte_toep': acc_sorte_toep,
        'acc_ecnet': acc_ecnet,
        'acc_toep': acc_toep
        }
savemat('acc_comparison_snapshots_%d.mat'%SNRdB, hist)
#savemat('acc_comparison_SNR_%d_snaps.mat'%N_snapshots, hist)

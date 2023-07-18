# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:30:52 2020

@author: DaiHoang
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.utils import np_utils
#from parameters import doa_samples
from scipy.linalg import cholesky, ldl, toeplitz
from tqdm import tqdm

c = 3e8
fc = 1e9
wavelength = c/fc
d = 0.5*wavelength
resolution = 0.1
doa_samples = np.arange(-60, 60, resolution)

def attenuation_coef():
    sign_real = 1 if np.random.random() < 0.5 else -1
    sign_imag = 1 if np.random.random() < 0.5 else -1
    return (sign_real*np.random.normal(0, 1) + sign_imag*1j*np.random.normal(0, 1))/np.sqrt(2)

def DoAs_samples(N_signals=4, DoAs_spacing=1):
    DoAs_truth = []
    while len(DoAs_truth) < N_signals: # to ensure no same DoA in this list
        DoA_tmp = np.random.uniform(-60, 60)
        # Assign the very first DoA value or only append DoAs out of spacing
        if len(DoAs_truth) == 0 or np.sum(np.abs(np.array(DoAs_truth) - DoA_tmp) > DoAs_spacing) == len(DoAs_truth):
            DoAs_truth.append(DoA_tmp)
    return DoAs_truth

def DoA_onehot(DoAs, resolution):
    DoA_samples = np.arange(-60, 60, resolution)
    spec_truth = np.zeros(len(DoA_samples))
    for i in range(len(DoA_samples)):
        for DoA in DoAs:
            if np.abs(DoA_samples[i] - DoA) < 1e-2: 
                spec_truth[i] = 1
    return spec_truth

def array_signal_calculator(SNRdB, delta_SNRdB, N_antenas, N_snapshots, DOAs, N_coherent=None):
    N_signals = len(DOAs)
    M = N_antenas//2
    if N_coherent is None: N_coherent = np.random.randint(0, high=N_signals+1)
    # position of array
    array_geom = np.expand_dims(np.array(np.arange(-M, M+1)), axis=-1)*d # position # = [0, 0,15, ..., 1.35]   
    # noise matrix
    N = (np.random.normal(size=(N_antenas, N_snapshots)) + 1j*np.random.normal(size=(N_antenas, N_snapshots)))/np.sqrt(2) # size = (N_antenas, N_snapshot)
    std_noise = np.std(N)
    
    snrdB = np.random.uniform(low=SNRdB - delta_SNRdB/2, high=SNRdB + delta_SNRdB/2);

    array_signal = 0
    S_hist = []
    for sig in range(N_signals):
        phase_shift_array = 2*np.pi*array_geom/wavelength*np.sin(DOAs[sig]*np.pi/180) 
        # size = (10, 1)
        A = np.exp(-1j*phase_shift_array) # steering vectors'matrix, size = (N_antenas, N_signals) = (N_antenas, 1)
        if sig <= N_coherent:
            if sig == 0: 
                S_0 = np.random.normal(size=(1, N_snapshots)) + 1j*np.random.normal(size=(1, N_snapshots)) # path-gain, size = (N_signals, N_snapshots) = (1, N_snapshots)
                S = 1*S_0
            else:
                S = attenuation_coef()*S_0
        else:
            S = np.random.normal(size=(1, N_snapshots)) + 1j*np.random.normal(size=(1, N_snapshots)) # path-gain, size = (N_signals, N_snapshots) = (1, N_snapshots)
        S = 10**(snrdB/20)*S/np.sqrt(2)
        array_signal += A.dot(S) 
    X = array_signal + N    
    return X, std_noise, N_coherent

def data_covariance(R, normalized=True, diag_element=False):
    # Collect elements in upper triangle of Covariance matrix
    cov_vector = []
    for r in range(R.shape[0]):
        cov_vector.extend(R[r, (r+1):])
    cov_vector_ = np.asarray(cov_vector)
    if diag_element:
        cov_vector_ = np.concatenate([np.diag(R).real, cov_vector_.real, cov_vector_.imag])
    else:
        cov_vector_ = np.concatenate([cov_vector_.real, cov_vector_.imag])
    if normalized:
        #cov_data = cov_vector_/np.linalg.norm(cov_vector_)
        cov_data = cov_vector_/np.linalg.norm(cov_vector_)
    else:
        cov_data = cov_vector_
    return cov_data

def data_toeplitz(X, SNRdB, N_antenas, N_snapshots):
    M = N_antenas//2
    SNR = 10**(SNRdB/10)

    # covariance matrix
    R = X.dot(np.matrix.getH(X))/N_snapshots # R.shape = (N_antenas, N_antenas)
    
    ### Construct F and G ###
    F = 0 
    # Construct full-row Toeplitz covariance matrix - F
    #for r in range(M+1):
    for r in range(R.shape[0]):
        Rm_tmp = R[r, :]
        Rm = toeplitz(Rm_tmp[:M+1][::-1], Rm_tmp[M:])
        F += Rm.dot(np.matrix.getH(Rm))
        
    # Collect eigenvalues from toeplitz algorithm
    #R_bar = F + np.identity(M+1).dot(np.matrix.getH(F)).dot(np.identity(M+1))
    
    ## Forward/Backward algorithm
    J = np.eye(M+1)[:, :: -1]
    R_bar = (F + J.dot(np.conj(F)).dot(J))/2
    eig_values, eig_vectors = data_eigens(R_bar)
        
    # Collect elements in upper triangle of Covariance matrix
    cov_data = data_covariance(R_bar)
    return cov_data, R_bar, eig_values, eig_vectors

def covariance_fbss(X, SNRdB, N_antenas, N_snapshots, M_fbss=None):
    # Reference: Forward/Backward Spatial Smoothing Techniques for Coherent Signal Indentification
    # Input: M_fbss is the number of antennas in each sub-array (not the total antennas)
    # N_sub_antenas <= N_antenas - N_signals + 1
    if M_fbss is None: M_fbss = N_antenas//2 # M
    L = N_antenas - M_fbss + 1 # L
       
    R_f = np.zeros((M_fbss, M_fbss), dtype=complex)
    for l in range(L):
        X_lf = X[l:l+M_fbss, :]
        #R_f += X_lf.dot(np.matrix.getH(X_lf))/(L*N_snapshots)
        R_f += X_lf.dot(np.matrix.getH(X_lf))/N_snapshots
        
    # Quick way to calculate R_b
    change = np.eye(M_fbss)
    R_b = np.dot(change[:, :: -1], np.conj(R_f).dot(change[:, :: -1]))
    
    '''
    R_b = np.zeros((N_sub_antenas, N_sub_antenas), dtype=complex)
    for i_sub in range(N_subarrays):
        X_tmp = X[::-1, :]
        X_lb = np.conj(X_tmp[i_sub:i_sub+N_sub_antenas, :])
        R_b += X_lb.dot(np.matrix.getH(X_lb))/(N_subarrays*N_snapshots)
    '''
    R_avg = (R_f + R_b)/2
    #return R_f # Uncomment this line to perform FOSS algorithm
    return R_avg

def data_fbss(X, SNRdB, N_antenas, N_snapshots, std_noise=None, M_fbss=None):  
    R_fbss = covariance_fbss(X, SNRdB, N_antenas, N_snapshots, M_fbss=M_fbss)
    if std_noise is not None: R_fbss += np.eye(R_fbss.shape[0])*(std_noise**2)
    eig_values, eig_vectors = data_eigens(R_fbss)
    # Collect elements in upper triangle of Cholesky deomposition
    #chol_vector, chol_norm = data_cholesky(R_fbss)
    ldl_L, ldl_D = data_LDL(R_fbss)
    return R_fbss, eig_values, eig_vectors, ldl_L, ldl_D

def EGM_liked_output(arr_sorted):
    delta_max = (arr_sorted[0] - arr_sorted[-1])/(len(arr_sorted) - 1)
    outputs = []
    for i in range(len(arr_sorted) - 1):
        delta = arr_sorted[i] - arr_sorted[i+1]
        if delta <= delta_max:
            outputs.append(delta)
    outputs = np.concatenate((outputs, np.zeros(len(arr_sorted) - len(outputs))))
    return outputs

def data_cholesky(R):
    # Collect elements in upper triangle of Cholesky deomposition
    chol = cholesky(R, lower=False)
    chol_vector = []
    for r in range(chol.shape[0]):
        chol_vector.extend(chol[r, (r+1):])
    chol_vector = np.asarray(chol_vector)
    chol_vector = np.concatenate([chol_vector.real, chol_vector.imag])
    chol_norm = np.linalg.norm(chol, axis=1, keepdims=False)
    return chol_vector, chol_norm

def data_LDL(R):
    # Collect elements in upper triangle of Cholesky deomposition
    chol, d, _ = ldl(R, lower=False)
    chol_vector = []
    for r in range(chol.shape[0]):
        chol_vector.extend(chol[r, (r+1):])
    chol_vector = np.asarray(chol_vector)
    chol_vector = np.concatenate([chol_vector.real, chol_vector.imag])
    d = -np.sort(-np.diag(d.real)) # Decending sort
    d = np.round(d, 5)
    return chol_vector, d

def data_eigens(R):
    eig_values, eig_vectors = np.linalg.eig(R)
    eig_values = eig_values.real
    # Decending sort
    eig_values = -np.sort(-np.real(eig_values)) 
    eig_vectors = eig_vectors[:, np.argsort(-np.real(eig_values))]
    return eig_values, eig_vectors

def data_root_music(R, N_antenas, N_signals, input_only=False):
    M = N_antenas//2
    eig_vals, eig_vect = data_eigens(R)
    Qn = eig_vect[:, N_signals:M+1]
    # Calculate the noise subspace
    C = Qn.dot(np.matrix.getH(Qn))
    # Construct the co-effiecent in polynomial equation (DNN input)
    f = []
    f_X = []
    for di in np.arange(-M, M+1)[::-1]:
        if di < 0:
            f_X.append(np.sum(np.diag(C, di)))
        f.append(np.sum(np.diag(C, di)))
    X = np.concatenate((np.real(f_X), np.imag(f_X)), axis = 0)
    X = X/np.linalg.norm(X)

    #X = data_covariance(C, normalized=False, diag_element=True)
    if input_only: return X
    # Root-Output of the DNN (DNN output)
    r = np.roots(f)
    #y = np.concatenate((np.real(r), np.imag(r)), axis = 0)
    abs_r = np.abs(r)
    r = r[np.argsort(-abs_r)]
    idx = np.where(np.abs(r) < 1)[0][:N_signals]
    
    DoAs_est = []
    for i in idx:
        #doa = np.math.asin(-np.angle(r[i])/np.pi)*180/np.pi
        doa = np.math.asin(-np.angle(r[i])/np.pi)
        DoAs_est.append(doa)
    y = np.sort(DoAs_est)
    return X, y

def angle_rounding(DoAs, resolution):
    DoAs_rounding = []
    DoA_samples = np.arange(-60, 60, resolution)
    for DoA in DoAs:
        for s in range(len(DoA_samples) - 1):
            if DoA >= DoA_samples[s] and DoA < DoA_samples[s+1]:
                DoA_mean = np.mean([DoA_samples[s], DoA_samples[s+1]])
                if DoA >= DoA_mean:
                    DoAs_rounding.append(DoA_samples[s+1])
                else:
                    DoAs_rounding.append(DoA_samples[s])
            if len(DoAs_rounding) >= len(DoAs): break
    return DoAs_rounding

def dataset_Nsignals_fbss(SNRdB, delta_SNRdB, N_antenas, N_snapshots, N_signals_min, N_signals_max, M_fbss, N_repeat=10000, random_flag=False):
    delta_DoA = 2
    data = {}
    data['eigen_fbss'] = []
    data['y_Nsignals'] = []
    data['DoAs'] = []

    for rep in tqdm(range(N_repeat)):
        # if rep%(N_repeat//10)==0: print('<=========== Repetition: %d ============>'%rep)
        if random_flag:
            N_signals = np.random.randint(N_signals_min, N_signals_max+1)
            N_coherent = np.random.randint(0, N_signals)
            DOAs = DoAs_samples(N_signals, delta_DoA)

            X, _, _ = array_signal_calculator(SNRdB, delta_SNRdB, N_antenas, N_snapshots, DOAs, N_coherent=N_coherent)
            # Data from FBSS algprithm
            _, eigen_fbss, _, _, _ = data_fbss(X, SNRdB, N_antenas, N_snapshots, std_noise=None, M_fbss=M_fbss)
            #data['X'].append(X)
            data['eigen_fbss'].append(eigen_fbss)            
            # Output for N_signals detection and DoA estimation
            data['y_Nsignals'].append(np_utils.to_categorical(N_signals, N_antenas//2 + 2))
            data['DoAs'].append(DOAs)

        else:
            for N_signals in range(N_signals_min, N_signals_max+1):
                DOAs = DoAs_samples(N_signals, delta_DoA)
                N_coherent = np.random.randint(0, N_signals)
                X, std_noise, _ = array_signal_calculator(SNRdB, delta_SNRdB, N_antenas, N_snapshots, DOAs, N_coherent=N_coherent)
                # Data from FBSS algprithm
                _, eigen_fbss, _, _, _ = data_fbss(X, SNRdB, N_antenas, N_snapshots, std_noise=None, M_fbss=M_fbss)
                #data['X'].append(X)
                data['eigen_fbss'].append(eigen_fbss)
                # Output for N_signals detection and DoA estimation
                data['y_Nsignals'].append(np_utils.to_categorical(N_signals, N_antenas//2 + 2))
                data['DoAs'].append(DOAs)
    return data

def dataset_Nsignals_toep(SNRdB, delta_SNRdB, N_antenas, N_snapshots, N_signals_min, N_signals_max, N_repeat=10000, random_flag=False):
    delta_DoA = 2
    data = {}
    data['eigen_toep'] = []
    data['y_Nsignals'] = []
    data['DoAs'] = []

    for rep in tqdm(range(N_repeat)):
        # if rep%(N_repeat//10)==0: print('<=========== Repetition: %d ============>'%rep)
        if random_flag:
            N_signals = np.random.randint(N_signals_min, N_signals_max+1)
            N_coherent = np.random.randint(0, N_signals)
            DOAs = DoAs_samples(N_signals, delta_DoA)

            X, _, _ = array_signal_calculator(SNRdB, delta_SNRdB, N_antenas, N_snapshots, DOAs, N_coherent=N_coherent)
            # Data from Toeplitz-FBSS algprithm
            _, _, eigen_toep, _ = data_toeplitz(X, SNRdB, N_antenas, N_snapshots)
            data['eigen_toep'].append(eigen_toep)          
            # Output for N_signals detection and DoA estimation
            data['y_Nsignals'].append(np_utils.to_categorical(N_signals, N_antenas//2 + 2))
            data['DoAs'].append(DOAs)

        else:
            for N_signals in range(N_signals_min, N_signals_max+1):
                DOAs = DoAs_samples(N_signals, delta_DoA)
                N_coherent = np.random.randint(0, N_signals)
                X, std_noise, _ = array_signal_calculator(SNRdB, delta_SNRdB, N_antenas, N_snapshots, DOAs, N_coherent=N_coherent)
                # Data from FBSS algprithm
                _, _, eigen_toep, _ = data_toeplitz(X, SNRdB, N_antenas, N_snapshots)
                #data['X'].append(X)
                data['eigen_toep'].append(eigen_toep)                   
                # Output for N_signals detection and DoA estimation
                data['y_Nsignals'].append(np_utils.to_categorical(N_signals, N_antenas//2 + 2))
                data['DoAs'].append(DOAs)
    return data


def dataset_generator():
    c = 3e8
    fc = 1e9
    wavelength = c/fc
    d = 0.5*wavelength

    # Generate dataset with various SNRs
    # delta_SNRdB_arr = [0, 2, 4, 8, 16]
    delta_SNRdB_arr = [4]
    SNRdB_arr = [10]
    N_antennas_arr = [35, 45, 55]
    # SNRdB_arr = [-10, 0, 10, 20, 30]    
    N_snapshots_arr = [1000] 
    
    #SNRdB_arr = [10]
    # N_antennas_arr = [25]
    # resolution = 0.2
    resolution_arr = [0.1]
    
    for SNRdB in SNRdB_arr:
        for N_antenas in N_antennas_arr:
            for N_snapshots in N_snapshots_arr:
                for delta_SNRdB in delta_SNRdB_arr:
                            
                    # Nsignals detection dataset
                    print('<=========== Generating dataset for Nsignals detection ============>')
                    ## Toeplitz-FBSS algorithm with various number of 
                    # Data for training
                    print('\n<=========== Dataset Toeplitz-FBSS ============>')
                    dataset_path = './train/dataset_Nsignals_clas_toep_%d_dB_%d_delta_%d_antenas_%d_snap.npy'\
                        %(SNRdB, delta_SNRdB, N_antenas, N_snapshots)
                    data = dataset_Nsignals_toep(SNRdB, delta_SNRdB, N_antenas, N_snapshots, N_signals_min=1, \
                                                        N_signals_max=6, N_repeat=20000)
                    np.save(dataset_path, data)
                    # Data for testing
                    print('\n<=========== Testset Toeplitz-FBSS ============>')
                    testset_path = './test/testset_Nsignals_clas_toep_%d_dB_%d_delta_%d_antenas_%d_snap.npy'\
                        %(SNRdB, delta_SNRdB, N_antenas, N_snapshots)
                    data = dataset_Nsignals_toep(SNRdB, delta_SNRdB, N_antenas, N_snapshots, N_signals_min=1, \
                                                        N_signals_max=6, N_repeat=10000, random_flag=True)
                    np.save(testset_path, data)

dataset_generator()

    
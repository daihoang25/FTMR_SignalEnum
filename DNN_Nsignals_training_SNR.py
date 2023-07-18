# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:42:54 2020

@author: Dai Hoang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv1D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Activation, Flatten, Dropout, LeakyReLU, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate, Permute, Subtract, add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

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

def dataset_preparation(data, SNRdB, feature_input, normalized_input=True, data_shuffling=True, N_samples=None):   
    print('<=========== Loading dataset (SNR = %ddB) ============>'%SNRdB)
    X_train, y_train = np.array(data['eigen_'+feature_input[0:4]]), np.array(data['y_Nsignals'])
    M = X_train.shape[1] - 1
    if data_shuffling:
        print('<=========== Shuffling dataset (SNR = %ddB) ============>'%SNRdB)
        X_train, y_train = data_shuffle(X_train, y_train)
    if (normalized_input):
        print('<=========== Normalize dataset (SNR = %ddB) ============>'%SNRdB)
        for i in range(len(X_train)):
            #X_train[i] = np.log(X_train[i]/X_train[i][-1]) # Super important
            X_train[i] = np.log(X_train[i]/(M+1))
            #X_train[i] = np.log(X_train[i])
            #X_train[i] /= np.linalg.norm(X_train[i])
            
    return X_train, y_train

def Nsignals_model_1(input_shape, output_shape, N_hidden=None):
    drop_rate = 0.01    
    if N_hidden is None: N_hidden = [32, 4, input_shape[0]] # Number of nodes in each hidden layers
    N_FC_blocks = 4
    inputs = Input(shape=input_shape, name='Nsignals_input') 
    #act_func = LeakyReLU(alpha=0.2)
    act_func = 'selu'
    x = inputs
    for fc in range(N_FC_blocks):
        x_ = x
        for h in range(len(N_hidden)):
            x_ = Dense(units=N_hidden[h], activation=act_func, name='FC_%d_block_%d'%(h, fc))(x_)
            #x = BatchNormalization()(x)
            if h == len(N_hidden) - 1:
                x_ = Dropout(drop_rate)(x_)
                x = add([inputs, x_])
                #x = BatchNormalization()(x)
    
    outputs = Dense(units=output_shape, activation='softmax', name='FC_Nsignals_4')(x)
    return Model(inputs, outputs)

def Nsignals_model(input_shape, output_shape, N_hidden=None):
    drop_rate = 0.01    
    # if N_hidden is None: N_hidden = [16, 32, 16, 8]
    M_1 = input_shape[0]
    if N_hidden is None: N_hidden = [int(M_1*2/3), int(M_1*1/3)]
    inputs = Input(shape=input_shape, name='Nsignals_input') 
    act_func = 'selu'
    #act_func = 'relu'
    for h in range(len(N_hidden)):
        if h == 0: 
            x = inputs
        x = Dense(units=N_hidden[h], activation=act_func, name='FC_%d'%h)(x)
        x = Dropout(drop_rate)(x)
    
    outputs = Dense(units=output_shape, activation='softmax', name='FC_Nsignals_4')(x)
    return Model(inputs, outputs)

def accuracy(y_true, y_pred):
    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)
    return np.sum(pred==true)/len(pred)

training_flag = True

def path_reader(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=11, training=True):
    if training == True:
        data_folder = './train/dataset'
    else:
        data_folder = './test/testset'
        
    if feature_input == 'fbss': 
    
        model_path = './saved_models/DNN_Nsignals_%s_%d_antenas_%d_dB_%d_delta_%d_snap_%d_Mfbss/cp.ckpt'\
            %(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss)
            
        checkpoint_path_Nsignals = os.path.abspath(model_path)
        dataset_path = data_folder + '_Nsignals_clas_%s_%d_dB_%d_delta_%d_antenas_%d_snap_%d_Mfbss.npy'\
            %(feature_input, SNRdB, delta_SNRdB, N_antenas, N_snapshots, M_fbss)
    #elif feature_input == 'toep':
    else:
        model_path = './saved_models/DNN_Nsignals_%s_%d_antenas_%d_dB_%d_delta_%d_snap/cp.ckpt'\
            %(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots)
            
        checkpoint_path_Nsignals = os.path.abspath(model_path)
        dataset_path = data_folder + '_Nsignals_clas_%s_%d_dB_%d_delta_%d_antenas_%d_snap.npy'\
            %(feature_input, SNRdB, delta_SNRdB, N_antenas, N_snapshots)
    return model_path, checkpoint_path_Nsignals, dataset_path


def training(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=None):
    model_path, checkpoint_path_Nsignals, dataset_path = path_reader(feature_input, \
                                                         N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss)
        
    checkpoint_folder_Nsignals = os.path.dirname(checkpoint_path_Nsignals)
    cp_callback_Nsignals = ModelCheckpoint(filepath=checkpoint_path_Nsignals, verbose=0, 
                              save_weights_only=True, save_best_only=True, period=1)
        
    print('<=========== Loading testset (SNR = %ddB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'\
          %(SNRdB, delta_SNRdB, N_antenas, N_snapshots))
    data = np.load(dataset_path, allow_pickle=True).item()

    X_train, y_train = dataset_preparation(data, SNRdB, feature_input, normalized_input=True, data_shuffling=True)
    #X_train = X_train/np.sum(X_train, axis=1)
    print('<=========== Training model (SNR = %ddB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'\
          %(SNRdB, delta_SNRdB, N_antenas, N_snapshots))
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:][0]
    loss_func = CategoricalCrossentropy(from_logits=False)
    model_2 = Nsignals_model(input_shape, output_shape)
    model_2.compile(loss=loss_func, optimizer=Adam(5e-4), metrics=['accuracy'])
    #plot_model(model_2, './model_2.png', show_shapes=True)
    hist_2 = model_2.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, 
                        verbose=0, callbacks=[
                            tfdocs.modeling.EpochDots(), 
                            cp_callback_Nsignals,
                            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)])
    #np.save('./history/DNN_N_signals_%s_%d_dB.npy'%(feature_input, SNRdB), hist_2.history)
    print('\n<=========== Saving classification model ============>')
    tf.saved_model.save(model_2, os.path.abspath(model_path))
    
def testing(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=None):
    model_path, checkpoint_path_Nsignals, testset_path = path_reader(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss, training=False)
    print('<=========== Loading testset (SNR = %ddB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'
          %(SNRdB, delta_SNRdB, N_antenas, N_snapshots))
    model_2 = tf.keras.models.load_model(model_path)   
    data = np.load(testset_path, allow_pickle=True).item()
    print('<=========== Testing model (SNR = %ddB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'\
          %(SNRdB, delta_SNRdB, N_antenas, N_snapshots))
    X_test, y_test = dataset_preparation(data, SNRdB, feature_input, normalized_input=True, data_shuffling=False)
    y_pred = model_2.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('Accuracy: %.2f'%accuracy(y_test, y_pred))
    return acc

def training_func():
    feature_input = 'toep'
    # feature_input = 'fbss'
    #feature_input = 'toep_foss'
    
    SNRdB_arr = [-10, 0, 10, 20, 30]
    # SNRdB_arr = [10]
    
    # delta_SNRdB_arr = [0, 2, 8, 16]
    delta_SNRdB_arr = [4]
    
    # N_snapshots_arr = [10, 100, 1000, 10000, 100000] 
    N_snapshots_arr = [10, 1000]    
    N_antenas_arr = [15]
    # N_antenas_arr = [25]

    results = {}
    
    for N_antenas in N_antenas_arr:
        for SNRdB in SNRdB_arr:
            for N_snapshots in N_snapshots_arr:
                for delta_SNRdB in delta_SNRdB_arr:
                    results[SNRdB] = []
                    results_tmp = []
                    # model path
                    if training_flag:
                        #if feature_input == 'toep':
                        if 'toep' in feature_input:
                            training(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots)
                            acc = testing(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=None)
                            results_tmp.append(acc)
            
                        else:
                            for M_fbss in np.arange(7, 13, 2): # For M0 = 15
                            # for M_fbss in [7, 13, 19]:
                                training(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss)
                                acc = testing(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=M_fbss)
                                results_tmp.append(acc)
                        results[SNRdB] = results_tmp
                    else:
                        #if feature_input == 'toep':
                        if 'toep' in feature_input:
                            acc = testing(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=None)
                            results_tmp.append(acc)
            
                        else:
                            for M_fbss in np.arange(7, 13, 2): # For M0 = 15
                            # for M_fbss in [7, 13, 19]:
                                acc = testing(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, M_fbss=M_fbss)
                                results_tmp.append(acc)
                        results[SNRdB] = results_tmp
                
training_func()

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:49:33 2017

@author: QingDog
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import EarlyStopping
import time
import scipy.io as sio


np.random.seed(0)
plt.close('all')

""" Define Functions """

def divide_train_test(features,target,train_ratio):
    train_size = int(len(features)*train_ratio)
    features_train, target_train= features[0:train_size,:], target[0:train_size]
    features_test, target_test = features[train_size:len(features),:], target[train_size:len(features)]
    return features_train, features_test, target_train, target_test


def create_dataset(dataset, look_back=1):
	datafeatures = []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		datafeatures.append(a)
	return np.array(datafeatures)
 
 
def subsampling(data_array, slice_id):
    data_array_subs = data_array[0::slice_id]
    return data_array_subs
    

def calc_new_dist_spdlim(Spdlim, s):
    Spdlim_transition = np.squeeze(np.array(np.where(np.diff(Spdlim)!=0)))
    I = len(Spdlim_transition)
    Distance = np.zeros(len(Spdlim))
    Distance[0:Spdlim_transition[0]] = np.fliplr(np.expand_dims(s[0:Spdlim_transition[0]],axis=0))[0,:]
    for i in range(1,I):
        Distance[Spdlim_transition[i-1]+1:Spdlim_transition[i]] = np.fliplr(np.expand_dims(s[Spdlim_transition[i-1]+1:Spdlim_transition[i]],axis=0))[0,:]-s[Spdlim_transition[i-1]]
    return Distance

  
def consecutive_ones(array):
    """ Counts consecutive ones of a boolean array """
    temp = np.diff(np.insert(array.astype(int),0,0))
    start_idx = np.array(np.where(temp == 1))[0,:]
    end_idx = np.array(np.where(temp==-1))[0,:]
    if len(end_idx)<len(start_idx):
        end_idx = np.append(end_idx, len(array))
    count = end_idx - start_idx
    return start_idx, end_idx, count
    
""" Here comes the part of the analysis for CCW Drivers!! """
     
source_path = '../../Datasets/AFP_Simulation_csv/CCW_Drivers/'
feature_files = listdir(source_path+'features')
target_files = listdir(source_path+'target')

for p in range(len(target_files)):
    curr_feature_file = feature_files[p]
    curr_target_file = target_files[p]
    
    features = np.array(pd.read_csv(source_path+'features/'+curr_feature_file, header=None))
    target = np.array(pd.read_csv(source_path+'target/'+curr_target_file, header=None))
    
    
    """ Scaling, Processing and Transformation into the correct formats """
    look_back = 5
    features_lag = np.array([])
    for numFeat in range(features.shape[1]):
        features_temp = np.expand_dims(features[:,numFeat],axis=1)
        laggedFeat = create_dataset(features_temp, look_back)
        features_lag = np.hstack([features_lag, laggedFeat]) if features_lag.size else laggedFeat
    
    target_lag = target[:-look_back]
    
    scaler_feat = MinMaxScaler(feature_range=(-1, 1))
    scaler_target = MinMaxScaler(feature_range=(-1,1))
    features_scale = scaler_feat.fit_transform(features_lag)
    target_scale = scaler_target.fit_transform(target_lag.reshape(-1,1))
    
    ### Shaping features into 3d - tensor
    nb_samples = len(features_scale)
    X_list = [np.expand_dims(np.atleast_2d(features_scale[i:look_back+i,:]), axis=0) for i in range(nb_samples-look_back)]           #creates 270 samples, each is 20 timesteps long and consists of values i:i+20 (each sample is shifted by one in the future)
    features = np.asarray(np.concatenate(X_list, axis=0), 'float32')                                 
    target = np.asarray(target_scale, 'float32')
    
    ### Dividing into "Train" (actually design) and "Test" (actually non-design) data
    train_size = int(len(features)*0.80)
    test_size = len(features)-train_size
    
    features_train, target_train= features[0:train_size,:], target[0:train_size]
    features_test, target_test = features[train_size:len(features),:], target[train_size:len(features)]
    
    
    """ Define LSTM algorithm """
    # (Full) Batch gradient descent
    batch_size = train_size
    R2_val_best = -np.inf
    
    ID = -1
    Ntrials = 1
    for no_hidden_layers in range(1,3):
        for cells in range(2,22,2):
            ID +=1
            print("Iteration %i with %i cells" %(ID, cells))
            for i in range(1,Ntrials+1):
    #            print("Trial ",i)
    
                if no_hidden_layers == 1:
                    model = Sequential()
                    model.add(LSTM(cells, input_shape=(look_back, features.shape[2]), return_sequences = False, stateful = False))
                else:
                    if int(cells) > 2:
                        model = Sequential()
                        model.add(LSTM(int(cells/2), input_shape=(look_back, features.shape[2]), return_sequences = True, stateful = False))
                        model.add(LSTM(int(cells/2), input_shape=(look_back, features.shape[2]), return_sequences = False, stateful = False))
                    else:
                        continue
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='rmsprop')
                early_stopping = EarlyStopping(monitor = 'val_loss', patience=20)
                
                
                # Training
#                time_start = time.time()
                model.fit(features_train, target_train, 
                          nb_epoch=320, 
                          batch_size=batch_size, 
                          verbose=0,          
                          validation_split=0.25,
                          callbacks=[early_stopping]
                          )
                
                target_trainPred = model.predict(features_train, batch_size = batch_size)
                target_testPred = model.predict(features_test, batch_size = batch_size)
                      
                ### Evaluation
                R2_trn = r2_score(target_train[:int(0.75*len(target_train))], target_trainPred[:int(0.75*len(target_train))])
                R2_val = r2_score(target_train[int(-0.25*len(target_train)):], target_trainPred[int(-0.25*len(target_train)):])
            
                model.reset_states()
                if R2_val > R2_val_best:
                    ID_best = ID
                    lstm_opt = model
                    R2_val_best = R2_val
#                    print("Highest accuracy achieved with %i cells:" %(cells))
#                    print("Training error: \t R2_trn: %.2f" %(R2_trn))
#                    print("Validation error: \t R2_val: %.2f" %(R2_val))
    
#    time_finish = time.time()    
    ### Saving results of best LSTM:
    target_trainPred = lstm_opt.predict(features_train, batch_size = batch_size)
    target_testPred = lstm_opt.predict(features_test, batch_size = batch_size)
    
    ### Evaluation
    target_train = scaler_target.inverse_transform(target_train)
    target_trainPred = scaler_target.inverse_transform(target_trainPred)
    target_test = scaler_target.inverse_transform(target_test)
    target_testPred = scaler_target.inverse_transform(target_testPred)  
    
    RMSE_trn = np.sqrt(mean_squared_error(target_train[:int(0.75*len(target_train))], target_trainPred[:int(0.75*len(target_train))]))
    RMSE_val = np.sqrt(mean_squared_error(target_train[int(-0.25*len(target_train)):], target_trainPred[int(-0.25*len(target_train)):]))
    RMSE_tst = np.sqrt(mean_squared_error(target_test, target_testPred))
    
    R2_trn = r2_score(target_train[:int(0.75*len(target_train))], target_trainPred[:int(0.75*len(target_train))])
    R2_val = r2_score(target_train[int(-0.25*len(target_train)):], target_trainPred[int(-0.25*len(target_train)):])
    R2_tst = r2_score(target_test, target_testPred) 
    
    path = '../../Datasets/'
    destination_path = 'Results/CCW_Drivers/LSTM/'
    destination_filename = 'data_' + target_files[p][-15:-12]+'_'+target_files[p][-11:-3]
    sio.savemat(destination_path+destination_filename, {'RMSE_trn':RMSE_trn, 'RMSE_val': RMSE_val, 'RMSE_tst': RMSE_tst, 
                                                        'R2_trn': R2_trn, 'R2_val': R2_val, 'R2_tst': R2_tst,
                                                        'prediction': scaler_target.inverse_transform(model.predict(features)), 'Net ID': ID_best})
    model_name = 'lstm_'+target_files[p][-15:-12]+'_'+target_files[p][-11:-3]    
    lstm_opt.save(destination_path+model_name+'.h5')
    
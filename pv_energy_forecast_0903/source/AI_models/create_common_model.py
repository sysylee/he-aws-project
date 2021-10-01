import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import random 
import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, GRU, RNN, Dropout, BatchNormalization, Conv1D, Flatten, Activation

from tensorflow.keras import Model
from tensorflow.keras import regularizers

from keras.optimizers import Adam
from ast import literal_eval

# from keras.callbacks import ModelCheckpoint
# from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

import keras.backend as k



def create_dnn_model(model_param_dict, input_win_len, output_win_len, num_feature):

    # dictionary 형태의 model_param 을 변수 선언 및 값 선언하기    
    for key, value in model_param_dict.items():
        temp_value =  model_param_dict[key]
        
        if type(temp_value) == str and '[' in temp_value: # dict 형태면 str -> dict 형태로 변환
            globals()['%s'%key] = literal_eval(temp_value)
        else:
            globals()['%s'%key] = temp_value
                    
    num_layer = len(hidden)
    input_shape = (input_win_len, num_feature)
 
    model = Sequential()

    for i in range(num_layer):
        if i == 0:
            model.add(Dense(hidden[i], input_shape = input_shape, activation =activation , kernel_initializer = kernel_init_method))
        else:
            model.add(Dense(hidden[i], kernel_initializer= kernel_init_method))
            
        if use_batch_normalization:
            model.add(BatchNormalization())
        if use_dropout:
            model.add(Dropout(dropout_rate))
            
    model.add(Dense(output_win_len, activation = 'linear'))
    model.compile(loss = loss , optimizer = Adam(lr= learning_rate))
    
    model.summary()    
    return model

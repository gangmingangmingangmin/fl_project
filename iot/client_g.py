#!/usr/bin/python3.7

import flwr as fl
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
import pandas as pd
import time
import os
import boto3
#load data and fit
##############################################
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
import time
import sys
##############################################
if len(sys.argv[1]) == 3:
  gid = int(sys.argv[1][0])-1
  cid = int(sys.argv[1][2:])-1
else:
  cid = int(sys.argv[1][3:])-1
##############################################
#loader
class TimeSeriesLoader:
    def __init__(self,file_n,div,n_clients):
        if n_clients>9:
          self.start_index=gid*div+cid*12
        else:
          self.start_index=cid*div
        min_n = min(file_n,self.start_index+div)# 파일 크기 안넘도록
        self.num_files = min_n-self.start_index
        self.files_indices = np.arange(self.num_files)
        self.shuffle_chunks()
    def num_chunks(self):
        return self.num_files

    def shuffle_chunks(self):
        np.random.shuffle(self.files_indices)

    def get_chunk(self, idx):
        # model.fit does train the model incrementally. ie. Can call m
        assert (idx >= 0) and (idx < self.num_files)

        #data load from s3
        ind = self.files_indices[idx]+self.start_index # data index
        s3 = boto3.resource('s3',region_name='ap-northeast-2')
        bucket = 'federatedlearning2'

        obj=s3.Object(bucket,'ts_file'+str(ind)+'.pkl')

        objd = obj.get()['Body'].read()
        df_ts = pickle.loads(objd)

        #data load from local
        '''
        with open('/home/ec2-user/data/ts_data/ts_file'+str(ind)+'.pkl','rb') as f:
            df_ts = pickle.load(f)
        '''
        num_records = len(df_ts.index)

        features = df_ts.drop('y', axis=1).values
        target = df_ts['y'].values

        # reshape for input into LSTM. Batch major format.
        features_batchmajor = np.array(features).reshape(num_records, -1, 1)
        return features_batchmajor, target

# Load and compile Keras model
ts_inputs = tf.keras.Input(shape=(1008,1))
x = layers.LSTM(units=10)(ts_inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=['mse'])


# Define Flower client
class flClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self,parameters,config):
        print(config)
        tss=TimeSeriesLoader(config['file_n'],config['div'],config['n_clients'])
        BATCH_SIZE = 128
        NUM_EPOCHS = config['epoch']
        NUM_CHUNKS = tss.num_chunks()
        rnd = config['round']-1
        NUM_CHUNKS_LIST=[]
        index=0
        r=NUM_CHUNKS//config['n_round']
        for i in range(config['n_round']):
            NUM_CHUNKS_LIST.append([index,index+r])
            index = index+r
        NUM_CHUNKS_LIST[-1][-1] = NUM_CHUNKS
        x_len=0
        for epoch in range(NUM_EPOCHS):
            print('epoch #{}'.format(epoch))
            for i in range(NUM_CHUNKS_LIST[rnd][0],NUM_CHUNKS_LIST[rnd][1]):
                X, y = tss.get_chunk(i)
                x_len+=len(X)
                model.fit(x=X, y=y, batch_size=BATCH_SIZE)
        
        return model.get_weights(), x_len, {}
    def evaluate(self, parameters, config):
      return 0,0,{"no evaluation":0}
    '''
    # client side evaluation
    def evaluate(self, parameters, config):
        # testdata
        df_val_ts = pd.read_pickle('/home/ec2-user/ts_file0.pkl')
        features = df_val_ts.
        drop('y', axis=1).values
        features_arr = np.array(features)

        # reshape for input into LSTM. Batch major format.
        num_records = len(df_val_ts.index)
        features_batchmajor = features_arr.reshape(num_records, -1, 1)

        model.set_weights(parameters)
        # Scaled to work with Neural networks.
        with open('/home/ec2-user/scaler_train.pickle','rb') as f:
            scaler = pickle.load(f)

        y_pred = model.predict(features_batchmajor).reshape(-1, )
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1 ,)

        y_act = df_val_ts['y'].values
        y_act = scaler.inverse_transform(y_act.reshape(-1, 1)).reshape(-1 ,)
        return mean_squared_error(y_act, y_pred), len(features_batchmajor),  {"MSE": mean_squared_error(y_act, y_pred)}
    '''
# Start Flower client
fl.client.start_numpy_client(server_address="172.31.18.91:8080", client=flClient())

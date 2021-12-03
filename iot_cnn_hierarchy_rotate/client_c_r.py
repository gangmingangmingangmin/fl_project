#!/usr/bin/python3.7

import flwr as fl
import math
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
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
          self.start_index=gid*div+cid*5555
        else:
          self.start_index=cid*div
        #min_n = min(file_n,self.start_index+div)# 파일 크기 안넘도록
        #self.num_files = min_n-self.start_index
        self.num_files = div
        self.files_indices = np.arange(self.num_files)
        self.shuffle_chunks()
        
        #데이터 로드
        s3 = boto3.resource('s3',region_name='ap-northeast-2')
        bucket = 'federatedlearning2'

        #python 2 버전에서 dump한 파일이기때문에 encoding, python 3 버전은 bytes 사용
        obj = s3.Object(bucket,'mnist/X_train.pickle')
        objd=obj.get()['Body'].read()
        X_train = pickle.loads(objd,encoding='bytes')

        obj = s3.Object(bucket,'mnist/y_train.pickle')
        objd=obj.get()['Body'].read()
        y_train = pickle.loads(objd,encoding='bytes')

        X_train = X_train.reshape(X_train.shape[0],img_row,img_col,1)
        X_train = X_train.astype('float32')/255
        y_train = tf.keras.utils.to_categorical(y_train,10)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_train[50000:]
        self.y_val = y_train[50000:]
    def num_chunks(self):
        return self.num_files

    def shuffle_chunks(self):
        np.random.shuffle(self.files_indices)

    def get_chunk(self):
        # model.fit does train the model incrementally. ie. Can call m
        #assert (idx >= 0) and (idx < self.num_files)

        #data load from s3
        ind = self.num_files+self.start_index # data index
        
        X = self.X_train[self.start_index:ind]
        y = self.y_train[self.start_index:ind]

        return X,y


#model for cnn
img_row = 28
img_col = 28
input_shape = (img_row,img_col,1)
num_classes = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Define Flower client
class flClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self,parameters,config):
        # round 찾기
        rnd = config['round']-1
        cid +=1
        if cid ==9:
          cid = 0

        tss=TimeSeriesLoader(config['file_n'],config['div'],config['n_clients'])
        BATCH_SIZE = 128
        NUM_EPOCHS = config['epoch']
        NUM_CHUNKS = tss.num_chunks()
        
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
            #for i in range(NUM_CHUNKS_LIST[rnd][0],NUM_CHUNKS_LIST[rnd][1]):
            #for i in range(NUM_CHUNKS):
            X, y = tss.get_chunk()
            x_len+=len(X)
            print(X.shape)
            print(tss.X_val.shape)
            print(y.shape)
            print('@@@@@@@@@@@@@@@@@@@')
            model.fit(x=X, y=y, batch_size=BATCH_SIZE, validation_data = (tss.X_val, tss.y_val))
    
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
fl.client.start_numpy_client(server_address="172.31.18.242:8080", client=flClient())

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

#load data and fit
##############################################
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
import time
##############################################
# 데이터 로더
class TimeSeriesLoader:
    def __init__(self):
        self.k = ['ts_file34.pkl']
        self.num_files = len(self.k)
        self.files_indices = np.arange(self.num_files)
        self.shuffle_chunks()
    def num_chunks(self):
        return self.num_files

    def shuffle_chunks(self):
        np.random.shuffle(self.files_indices)

    def get_chunk(self, idx):
        # model.fit does train the model incrementally. ie. Can call m
        assert (idx >= 0) and (idx < self.num_files)

        #data load
        ind = self.files_indices[idx] # data index

        #s3 = boto3.resource('s3',region_name='ap-northeast-2')
        #bucket = 'federatedlearing'

        #obj=s3.Object(bucket,self.k[ind])

        #objd = obj.get()['Body'].read()
        with open('/home/hadoop/federated/flower/ts_data/ts_file34.pkl','rb') as d:
            df_ts = pickle.load(d)

        num_records = len(df_ts.index)

        features = df_ts.drop('y', axis=1).values
        target = df_ts['y'].values

        # reshape for input into LSTM. Batch major format.
        features_batchmajor = np.array(features).reshape(num_records, -1, 1)
        return features_batchmajor, target
tss=TimeSeriesLoader()
# Load and compile Keras model
ts_inputs = tf.keras.Input(shape=(1008,1))
x = layers.LSTM(units=10)(ts_inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=['mse'])
# testdata
df_val_ts = pd.read_pickle('/home/hadoop/federated/flower/ts_val_data/ts_file0.pkl')


features = df_val_ts.drop('y', axis=1).values
features_arr = np.array(features)

# reshape for input into LSTM. Batch major format.
num_records = len(df_val_ts.index)
features_batchmajor = features_arr.reshape(num_records, -1, 1)



# Define Flower client
class flClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self,parameters,config):
        BATCH_SIZE = 128
        NUM_EPOCHS = config['epoch']
        NUM_CHUNKS = tss.num_chunks()
        begin = time.time()
        for epoch in range(NUM_EPOCHS):
            print('epoch #{}'.format(epoch))
            for i in range(NUM_CHUNKS):
                X, y = tss.get_chunk(i)

                model.fit(x=X, y=y, batch_size=BATCH_SIZE)
        
        return model.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # Scaled to work with Neural networks.
        with open('/home/hadoop/federated/flower/scaler_train.pickle','rb') as f:
            scaler = pickle.load(f)

        y_pred = model.predict(features_batchmajor).reshape(-1, )
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1 ,)

        y_act = df_val_ts['y'].values
        y_act = scaler.inverse_transform(y_act.reshape(-1, 1)).reshape(-1 ,)
        return mean_squared_error(y_act, y_pred), len(features_batchmajor),  {"loss": mean_squared_error(y_act, y_pred)}

# Start Flower client
fl.client.start_numpy_client(server_address="172.31.24.139:8080", client=flClient())

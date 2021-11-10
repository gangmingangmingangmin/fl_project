#!/usr/bin/python3.7

import flwr as fl
from typing import Callable, Dict, Optional, Tuple
import sys
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import pickle
MIN_AVAILABLE_CLIENTS=int(sys.argv[1])
ssid = sys.argv[2][3]
NUM_ROUND=1


#data load from boto3
def divide_list(arr,n):
    for i in range(0,len(arr),n):
        yield arr[i:i+n]

#read file from s3
import boto3
client = boto3.client('s3')
paginator = client.get_paginator('list_objects_v2')
response_iterator = paginator.paginate(Bucket='federatedlearning2')
file_n =0
for page in response_iterator:
    for content in page['Contents']:
        if content['Key'][-4:]=='.pkl':
            file_n+=1

#read file from local
#import os
#path = '/home/ec2-user/data/ts_data'
#file_list=os.listdir(path)
#file_n=0
#for content in file_list:
#    if content[-4:]=='.pkl':
#        file_n+=1

# 데이터 수를 노드 수에 맞추기 위한 코드
file_n = 108 #고정

if file_n % MIN_AVAILABLE_CLIENTS > 0:
    DIV = file_n // MIN_AVAILABLE_CLIENTS +1
else:
    DIV = file_n//MIN_AVAILABLE_CLIENTS
print(file_n,DIV)
#strategy

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    def fit_config(rnd:int) -> Dict[str,str]:
        config = {"epoch" : 1, "round":rnd, "file_n":file_n, "div":DIV, "n_round":NUM_ROUND,"n_clients": MIN_AVAILABLE_CLIENTS} 
        return config
    return fit_config

#server side evaluation
def get_eval_fn(model):
    
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        with open('parameters'+ssid+'.pickle','wb') as f:
            pickle.dump(weights,f)
        #s3 upload
        client.upload_file('./parameters'+ssid+'.pickle','federatedlearning2','parameters'+ssid+'.pickle')
        return 0, {"r2_score":0}

    return evaluate

# model
ts_inputs = tf.keras.Input(shape=(1008,1))
x = layers.LSTM(units=10)(ts_inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=['mse'])

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 10% of available clients for the next round
    min_fit_clients=9,  # Minimum number of clients to be sampled for the next round
    min_available_clients=9,  # Minimum number of clients that need to be connected to the server before a training round can start
    min_eval_clients=9, # default = 2
    on_fit_config_fn=get_on_fit_config_fn(),
    eval_fn = get_eval_fn(model)
)

import time
#federated learning
start_time = time.time()
fl.server.start_server(config={"num_rounds": NUM_ROUND},strategy=strategy)
end_time = time.time()

print('processing time : '+str(end_time-start_time))

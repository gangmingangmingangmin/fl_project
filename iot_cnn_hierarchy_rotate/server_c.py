#!/usr/bin/python3.7

import flwr as fl
import warnings
warnings.simplefilter(action='ignore',category = FutureWarning)
from typing import Callable, Dict, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import sys
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
import pandas as pd
import numpy as np
import pickle
np.random.seed(10)

MIN_AVAILABLE_CLIENTS=int(sys.argv[1])
ssid = sys.argv[2][3] # sub server id
NUM_ROUND=1
NUM_EPOCHS = 1

#data load from boto3
def divide_list(arr,n):
    for i in range(0,len(arr),n):
        yield arr[i:i+n]




# 데이터 수를 노드 수에 맞추기 위한 코드
file_n = 50000 #고정 50000

# 나머지 분배코드, 사용x
if file_n % MIN_AVAILABLE_CLIENTS > 0:
    DIV = file_n // MIN_AVAILABLE_CLIENTS +1
else:
    DIV = file_n//MIN_AVAILABLE_CLIENTS
print(file_n,DIV)

#fit strategy
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    def fit_config(rnd:int) -> Dict[str,str]:
        config = {"epoch" : NUM_EPOCHS, "round":rnd, "file_n":file_n, "div":DIV, "n_round":NUM_ROUND,"n_clients": MIN_AVAILABLE_CLIENTS}
        return config
    return fit_config
#server side evaluation
def get_eval_fn(model):
    
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        #read file from s3
        import boto3
        client = boto3.client('s3')
        bucket = 'federatedlearning2'
        s3 = boto3.resource('s3',region_name = 'ap-northeast-2')

        model.set_weights(weights)
        with open('parameters_c'+ssid+'.pickle','wb') as f:
          pickle.dump(weights,f)
        #s3 upload
        client.upload_file('./parameters_c'+ssid+'.pickle','federatedlearning2','parameters_c'+ssid+'.pickle')
        return 0, {"err":0}

    return evaluate

#model for cnn
img_row = 28
img_col = 28
input_shape = (img_row,img_col,1)


batch_size = 128
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
fl.server.start_server(config={"num_rounds": NUM_ROUND},strategy=strategy)


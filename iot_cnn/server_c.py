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
from mlxtend.data import loadlocal_mnist
np.random.seed(10)

MIN_AVAILABLE_CLIENTS=int(sys.argv[1])
NUM_ROUND=1
NUM_EPOCHS = 20

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

        obj = s3.Object(bucket,'mnist/mnist.pkl')
        objd=obj.get()['Body'].read()
        #python 2 버전에서 dump한 파일이기때문에 encoding, python 3 버전은 bytes 사용
        (X_train,y_train),(X_test,y_test) = pickle.loads(objd,encoding = 'bytes') 
        X_test = X_test.reshape(X_test.shape[0],img_row,img_col,1)
        X_test = X_test.astype('float32')/255
        y_test = tf.keras.utils.to_categorical(y_test,10)
        print(X_test.shape)
        model.set_weights(weights)  # Update model with the latest parameters
        #x_test, y_test
        predict = model.predict(X_test)
        y_pred = np.argmax(predict,axis=1)
        y_label = np.argmax(y_test,axis=1)
        #classification_report, confusion_matrix
        cr = classification_report(y_label,y_pred,digits = 4)
        cm = confusion_matrix(y_label,y_pred)
        f = open('/home/ec2-user/result.txt','w')
        f.write("classification_report : "+str(cr)+"\n")
        f.write("confusion_matrix : "+str(cm))
        f.close()
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
    min_fit_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients to be sampled for the next round
    min_available_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
    min_eval_clients=MIN_AVAILABLE_CLIENTS, # default = 2
    on_fit_config_fn=get_on_fit_config_fn(),
    eval_fn = get_eval_fn(model)
)

import time
#federated learning
fl.server.start_server(config={"num_rounds": NUM_ROUND},strategy=strategy)


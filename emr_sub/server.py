#!/usr/bin/python3.7

import flwr as fl
from typing import Callable, Dict
import sys

MIN_AVAILABLE_CLIENTS=int(sys.argv[1])
NUM_ROUND=2


#data load from boto3
def divide_list(arr,n):
    for i in range(0,len(arr),n):
        yield arr[i:i+n]

#read file from s3
#import boto3
#client = boto3.client('s3')
#paginator = client.get_paginator('list_objects_v2')
#response_iterator = paginator.paginate(Bucket='federatedlearing')
#file_n =0
#for page in response_iterator:
#    for content in page['Contents']:
#        if content['Key'][-4:]=='.pkl':
#            file_n+=1

#read file from local
import os
path = '/home/ec2-user/data/ts_data'
file_list=os.listdir(path)
file_n=0
for content in file_list:
    if content[-4:]=='.pkl':
        file_n+=1

if file_n % MIN_AVAILABLE_CLIENTS > 0:
    DIV = file_n // MIN_AVAILABLE_CLIENTS +1
else:
    DIV = file_n//MIN_AVAILABLE_CLIENTS
print(file_n,DIV)
#strategy

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    def fit_config(rnd:int) -> Dict[str,str]:
        config = {"epoch" : 1, "round":rnd, "file_n":file_n, "div":DIV, "n_round":NUM_ROUND}
        return config
    return fit_config


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 10% of available clients for the next round
    min_fit_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients to be sampled for the next round
    min_available_clients=MIN_AVAILABLE_CLIENTS,  # Minimum number of clients that need to be connected to the server before a training round can start
    min_eval_clients=MIN_AVAILABLE_CLIENTS, # default = 2
    on_fit_config_fn=get_on_fit_config_fn(),
)

import time
#federated learning
start_time = time.time()
fl.server.start_server(config={"num_rounds": NUM_ROUND},strategy=strategy)
end_time = time.time()

print('processing time : '+str(end_time-start_time))

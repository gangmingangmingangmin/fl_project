# import packages
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

# read the dataset into python
df = pd.read_csv('/home/hadoop/federated/household/household_power_consumption.txt', delimiter=';')

df['date_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df['Sub_metering_1']  = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')
df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])
df = df.dropna(subset=['Sub_metering_1'])
df = df.dropna(subset=['Sub_metering_2'])
df = df.dropna(subset=['Sub_metering_3'])

df['date_time'] = pd.to_datetime(df['date_time'])

df = df.loc[:, ['date_time','Sub_metering_1','Sub_metering_2','Sub_metering_3','Global_active_power']]
df.sort_values('date_time', inplace=True, ascending=True)
df = df.reset_index(drop=True)

test_cutoff_date = df['date_time'].max() - timedelta(days=7)
val_cutoff_date = test_cutoff_date - timedelta(days=14)
df_test = df[df['date_time'] > test_cutoff_date]
df_val = df[(df['date_time'] > val_cutoff_date) & (df['date_time'] <= test_cutoff_date)]
df_train = df[df['date_time'] <= val_cutoff_date]


def create_ts_files(dataset,
                    label, 
                    start_index, 
                    end_index, 
                    history_length, 
                    step_size, 
                    target_step, 
                    num_rows_per_file, 
                    data_folder):
    assert step_size > 0
    assert start_index >= 0
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    time_lags = sorted(range(target_step+1, target_step+history_length+1, step_size), reverse=True)
    col_names = [f'x_lag{i}' for i in time_lags] + ['y']
    start_index = start_index + history_length
    if end_index is None:
        end_index = len(dataset) - target_step
    
    rng = range(start_index, end_index)
    num_rows = len(rng)
    print(num_rows)
    num_files = math.ceil(num_rows/num_rows_per_file)
    
    # for each file.
    print(f'Creating {num_files} files.')
    for i in range(num_files):
        filename = f'{data_folder}/ts_file{i}.pkl'
        
        if i % 10 == 0:
            print(f'{filename}')
            
        # get the start and end indices.
        ind0 = i*num_rows_per_file
        ind1 = min(ind0 + num_rows_per_file, end_index)
        data_list = []
        
        # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
        
        for j in range(ind0, ind1):
            indices = range(j-1, j-history_length-1, -step_size)
            data = np.append( dataset[sorted(indices)] , np.array(label[j+target_step]) )
            # append data to the list.
            data_list.append(data)
        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        df_ts.to_pickle(filename)
            
    return len(col_names)-1


global_active_power=df_train['Global_active_power'].values
sub_metering1 = df_train['Sub_metering_1'].values
sub_metering2 = df_train['Sub_metering_2'].values
sub_metering3 = df_train['Sub_metering_3'].values

#scaling

scaler = MinMaxScaler(feature_range=(0,1))
global_active_power_scaled = scaler.fit_transform(global_active_power.reshape(-1,1)).reshape(-1,)
with open('scaler_train_sub.pickle','wb') as f:
    pickle.dump(scaler,f)

scaler = MinMaxScaler(feature_range=(0,1))
sub_metering1_scaled = scaler.fit_transform(sub_metering1.reshape(-1,1)).reshape(-1,)
with open('sub_metering1_scaler.pickle','wb') as f:
    pickle.dump(scaler,f)

scaler = MinMaxScaler(feature_range=(0,1))
sub_metering2_scaled = scaler.fit_transform(sub_metering2.reshape(-1,1)).reshape(-1,)
with open('sub_metering2_scaler.pickle','wb') as f:
    pickle.dump(scaler,f)

scaler = MinMaxScaler(feature_range=(0,1))
sub_metering3_scaled = scaler.fit_transform(sub_metering3.reshape(-1,1)).reshape(-1,)
with open('sub_metering3_scaler.pickle','wb') as f:
    pickle.dump(scaler,f)

history_length = 7*24*60
step_size = 10
target_step = 10

for i,v in enumerate([sub_metering1_scaled,sub_metering2_scaled,sub_metering3_scaled]):
    if i==2:
        num_timesteps = create_ts_files(v,global_active_power_scaled,start_index=0,end_index=None,history_length=history_length,step_size=step_size,target_step=target_step,num_rows_per_file=128*100,data_folder='sub_metering'+str(i+1)+'_scaled_data')



global_active_power_val=df_train['Global_active_power'].values
sub_metering1_val = df_val['Sub_metering_1'].values
sub_metering2_val = df_val['Sub_metering_2'].values
sub_metering3_val = df_val['Sub_metering_3'].values

with open('scaler_train_sub.pickle','rb') as f:
    scaler = pickle.load(f)
global_active_power_val_scaled = scaler.transform(global_active_power_val.reshape(-1,1)).reshape(-1,)

with open('sub_metering1_scaler.pickle','rb') as f:
    scaler = pickle.load(f)
sub_metering1_val_scaled = scaler.transform(sub_metering1_val.reshape(-1,1)).reshape(-1,)

with open('sub_metering2_scaler.pickle','rb') as f:
    scaler = pickle.load(f)
sub_metering2_val_scaled = scaler.transform(sub_metering2_val.reshape(-1,1)).reshape(-1,)

with open('sub_metering3_scaler.pickle','rb') as f:
    scaler = pickle.load(f)
sub_metering3_val_scaled = scaler.transform(sub_metering3_val.reshape(-1,1)).reshape(-1,)

history_length = 7*24*60
step_size=10
target_step=10

for i,v in enumerate([sub_metering1_val_scaled,sub_metering2_val_scaled,sub_metering3_val_scaled]):
    if i==2:
        num_timesteps = create_ts_files(v,global_active_power_scaled,start_index=0,end_index=None,history_length=history_length,step_size=step_size,target_step=target_step,num_rows_per_file=128*100,data_folder='sub_metering'+str(i+1)+'_val_scaled_data')
    

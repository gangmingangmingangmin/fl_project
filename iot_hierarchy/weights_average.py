import boto3
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, r2_score

client = boto3.client('s3')
paginator = client.get_paginator('list_objects_v2')

response_iterator = paginator.paginate(
    Bucket='federatedlearning2'
)
file_list=[]
for page in response_iterator:
    for content in page['Contents']:
        if content['Key'][:10] =='parameters':
            file_list.append(content['Key'])
s3 = boto3.resource('s3',region_name='ap-northeast-2')
bucket = 'federatedlearning2'

weights = []
for i,item in enumerate(file_list):
    obj=s3.Object(bucket,item)
    objd = obj.get()['Body'].read()
    objO = pickle.loads(objd)
    weights.append(objO)
    if i==0:
      break # cluster 수에 맞게 멈추기
# weights average
avg_weight=list()
for weight in zip(*weights):
  layer_mean = tf.math.reduce_sum(weight,axis=0)/len(weights)
  avg_weight.append(layer_mean)

# model
ts_inputs = tf.keras.Input(shape=(1008,1))
x = layers.LSTM(units=10)(ts_inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=['mse'])

model.set_weights(avg_weight)  # Update model with the latest parameters

# testdata
df_val_ts = pd.read_pickle('/home/ec2-user/fl_project/data/ts_file0.pkl')
features = df_val_ts.drop('y', axis=1).values
features_arr = np.array(features)

# reshape for input into LSTM. Batch major format.
num_records = len(df_val_ts.index)
features_batchmajor = features_arr.reshape(num_records, -1, 1)

# Scaled to work with Neural networks.
with open('/home/ec2-user/fl_project/data/scaler_train.pickle','rb') as f:
    scaler = pickle.load(f)

y_pred = model.predict(features_batchmajor).reshape(-1, )
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1 ,)

y_act = df_val_ts['y'].values
y_act = scaler.inverse_transform(y_act.reshape(-1, 1)).reshape(-1 ,)
loss = mean_squared_error(y_act, y_pred)
accuracy =r2_score(y_act, y_pred)
f = open('/home/ec2-user/result.txt','w')
f.write("mse : "+str(loss)+"\n")
f.write("r2 : "+str(accuracy))
f.close()


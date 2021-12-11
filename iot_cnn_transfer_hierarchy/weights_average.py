import boto3
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix

client = boto3.client('s3')
paginator = client.get_paginator('list_objects_v2')

response_iterator = paginator.paginate(
    Bucket='federatedlearning2'
)
file_list=[]
for page in response_iterator:
    for content in page['Contents']:
        if content['Key'][:12] =='parameters_c':
            file_list.append(content['Key'])
s3 = boto3.resource('s3',region_name='ap-northeast-2')
bucket = 'federatedlearning2'

weights = []
for i,item in enumerate(file_list):
    obj=s3.Object(bucket,item)
    objd = obj.get()['Body'].read()
    objO = pickle.loads(objd)
    weights.append(objO)
    if i==1:
      break # cluster 수에 맞게 멈추기
# weights average
avg_weight=list()
for weight in zip(*weights):
  layer_mean = tf.math.reduce_sum(weight,axis=0)/len(weights)
  avg_weight.append(layer_mean)

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

with open('/home/ec2-user/fl_project/data/full.pkl','rb') as f:
  w = pickle.load(f)

new_weights = w[:4] + avg_weight[4:]
model.set_weights(new_weights)  # Update model with the latest parameters

#데이터 로드

#python 2 버전에서 dump한 파일이기때문에 encoding, python 3 버전은 bytes 사용
obj = s3.Object(bucket,'mnist/X_test.pickle')
objd=obj.get()['Body'].read()
X_test = pickle.loads(objd,encoding='bytes')

obj = s3.Object(bucket,'mnist/y_test.pickle')
objd=obj.get()['Body'].read()
y_test = pickle.loads(objd,encoding='bytes')

X_test = X_test.reshape(X_test.shape[0],img_row,img_col,1)
X_test = X_test.astype('float32')/255
y_test = tf.keras.utils.to_categorical(y_test,10)
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

print('finish')
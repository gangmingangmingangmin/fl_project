import boto3
import pickle
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
print(file_list)
s3 = boto3.resource('s3',region_name='ap-northeast-2')
bucket = 'federatedlearning2'
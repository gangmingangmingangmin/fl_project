SERVERS="
15.164.237.132
3.36.13.222
15.164.207.230
3.37.45.61
15.164.203.27
13.125.134.212
15.164.182.203
54.180.129.229
3.37.28.85
"

for m in $SERVERS
do
 ssh -i ../../../fedtest.pem  hadoop@$m nohup python /home/hadoop/federated/flower/client.py &
done

#!/bin/sh
PID=`ps -ef | grep “/home/ec2-user/server.py” | grep -v “grep | awk '{print $2}'`
echo “kill $PID”
kill -9 $PID

PID=`ps -ef | grep “/home/ec2-user/client.py” | grep -v “grep | awk '{print $2}'`
echo “kill $PID”
kill -9 $PID
### Dataset path in HDFS
```
hdfs://namenode:9000/user/flume/raw/data/spool/
```

### Mounted path to spark master containers
```
./apps:  /opt/spark-apps
./data:  /opt/spark-data
```

### To submit spark task
```
spark/bin/spark-submit --master spark://spark-master:7077   --deploy-mode client  spark-apps/customer_analysis.py
```
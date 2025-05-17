# E-Commerce_Platform_Sales_Analytics_and_Optimization_Pipeline

Follow instructions below to build and run the pipeline

## 0. Pre-requisite
Make sure Docker and Docker Compose are installed

## 1. Build Customized Image
Build Apache Flume image
```bash
cd infrastructure/flume/docker-flume
docker build -t my-flume:latest .
```
Build Spark Cluster image
```bash
cd infrastructure/spark_cluster
docker build -t cluster-apache-spark:3.0.2  .
```

## 2. Run Containerized Pipeline Components
In repo directory
```bash
docker compose up
```
Use `docker ps` to check and make sure containers are running

## 3. Pipeline Stages
### 3.1 Data Ingestion
Put the dataset file `customer_shopping_data.csv` into directory `raw_dataset/landing`

Apache Flume will automatically pick up the file and mark it `customer_shopping_data.csv.COMPLETED`

### 3.2 Data Processing
Run following command to execute batch processing task
```bash
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077   --deploy-mode client  ../spark-apps/batch_processing.py
```

### 3.3 Data Analytics
Run following command to execute data prediction task
```bash
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077   --deploy-mode client  ../spark-apps/prediction.py
```

### 3.4 Data Visualization
Wait until the two tasks above to complete, which will be shown in the console

Go to `http://[your-ip-address]:5601/` to view visualization Kibana dashboard

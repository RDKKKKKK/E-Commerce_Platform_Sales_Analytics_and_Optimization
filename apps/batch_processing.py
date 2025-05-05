#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
客户消费行为分析与个性化营销策略批处理程序
使用Spark框架进行大规模数据处理
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, desc, month, year, dayofweek, when, round, lit
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Customer Shopping Analysis") \
    .getOrCreate()

# Load CSV data
df = spark.read.csv("hdfs://namenode:9000/user/flume/raw/data/spool/customer_shopping_data.csv.1746246540086.tmp", header=True, inferSchema=True)

# Data preprocessing
# 1. Calculate total amount for each transaction
df = df.withColumn("total_amount", col("quantity") * col("price"))

# 2. Convert date format - specify the format for "dd/mm/yyyy" style dates
df = df.withColumn("invoice_date",
                   F.to_timestamp(col("invoice_date"), "d/M/yyyy"))

# 3. Handle null dates by creating a flag
df = df.withColumn("valid_date", col("invoice_date").isNotNull())

# 4. Extract date-related features with null handling
df = df.withColumn("month", when(col("valid_date"), month(col("invoice_date"))).otherwise(lit(None))) \
    .withColumn("year", when(col("valid_date"), year(col("invoice_date"))).otherwise(lit(None))) \
    .withColumn("day_of_week", when(col("valid_date"), dayofweek(col("invoice_date"))).otherwise(lit(None)))

# Register as temporary view for SQL queries
df.createOrReplaceTempView("shopping_data")


# ========= Batch Analysis 1: Customer Segmentation =========
def analyze_customer_segments():
    print("Performing customer segmentation analysis...")

    customer_stats = df.groupBy("customer_id", "gender", "age") \
        .agg(
        count("invoice_no").alias("transaction_count"),
        sum("total_amount").alias("total_spent"),
        avg("total_amount").alias("avg_transaction_value")
    ) \
        .withColumn("total_spent", round(col("total_spent"), 2)) \
        .withColumn("avg_transaction_value", round(col("avg_transaction_value"), 2))

    customer_rfm = customer_stats.withColumn(
        "customer_segment",
        when(col("total_spent") > 1000, "High Value Customer")
        .when(col("transaction_count") > 10, "Frequent Customer")
        .when(col("avg_transaction_value") > 100, "High Ticket Customer")
        .otherwise("General Customer")
    )

    customer_rfm = customer_rfm.withColumn(
        "age_group",
        when(col("age") < 18, "Under 18")
        .when((col("age") >= 18) & (col("age") < 25), "18-24")
        .when((col("age") >= 25) & (col("age") < 35), "25-34")
        .when((col("age") >= 35) & (col("age") < 45), "35-44")
        .when((col("age") >= 45) & (col("age") < 55), "45-54")
        .when((col("age") >= 55) & (col("age") < 65), "55-64")
        .otherwise("65+")
    )

    customer_rfm.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/customer_segments")

    segment_counts = customer_rfm.groupBy("customer_segment").count().orderBy(desc("count"))
    segment_counts.show()

    age_gender_analysis = customer_rfm.groupBy("age_group", "gender") \
        .agg(
        count("customer_id").alias("customer_count"),
        round(avg("total_spent"), 2).alias("avg_total_spent")
    ) \
        .orderBy("age_group", "gender")

    age_gender_analysis.show(20)
    age_gender_analysis.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/age_gender_analysis")


# ========= Batch Analysis 2: Product Category Analysis =========
def analyze_product_categories():
    print("Performing product category analysis...")

    category_analysis = df.groupBy("category") \
        .agg(
        sum("quantity").alias("total_quantity_sold"),
        round(sum("total_amount"), 2).alias("total_revenue"),
        count("invoice_no").alias("transaction_count"),
        round(avg("price"), 2).alias("avg_price")
    ) \
        .orderBy(desc("total_revenue"))

    category_analysis.show()
    category_analysis.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/category_analysis")

    # Only include records with valid dates for time-based analysis
    valid_date_df = df.filter(col("valid_date") == True)

    monthly_category = valid_date_df.groupBy("month", "category") \
        .agg(round(sum("total_amount"), 2).alias("monthly_revenue")) \
        .orderBy("month", desc("monthly_revenue"))

    monthly_category.show()
    monthly_category.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/monthly_category_sales")

    # Use the filtered dataframe for age_group analysis as well
    preference_analysis = df.groupBy("category", "gender", "age_group") \
        .agg(round(sum("total_amount"), 2).alias("total_spent")) \
        .orderBy("category", "gender", "age_group")

    preference_analysis.show()
    preference_analysis.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/category_preferences")


# ========= Batch Analysis 3: Payment Method Analysis =========
def analyze_payment_methods():
    print("Performing payment method analysis...")

    payment_analysis = df.groupBy("payment_method") \
        .agg(
        count("invoice_no").alias("transaction_count"),
        round(sum("total_amount"), 2).alias("total_revenue"),
        round(avg("total_amount"), 2).alias("avg_transaction_value")
    ) \
        .orderBy(desc("transaction_count"))

    payment_analysis.show()
    payment_analysis.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/payment_analysis")

    payment_age_analysis = df.groupBy("age_group", "payment_method") \
        .agg(count("invoice_no").alias("usage_count")) \
        .orderBy("age_group", desc("usage_count"))

    payment_age_analysis.show()
    payment_age_analysis.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/payment_age_analysis")


# ========= Batch Analysis 4: Time Pattern Analysis =========
def analyze_time_patterns():
    print("Performing time pattern analysis...")

    # Only include records with valid dates for time-based analysis
    valid_date_df = df.filter(col("valid_date") == True)

    daily_sales = valid_date_df.groupBy("day_of_week") \
        .agg(
        round(sum("total_amount"), 2).alias("total_revenue"),
        count("invoice_no").alias("transaction_count")
    ) \
        .orderBy("day_of_week")

    daily_sales.show()
    daily_sales.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/daily_sales_pattern")

    monthly_sales = valid_date_df.groupBy("month") \
        .agg(
        round(sum("total_amount"), 2).alias("total_revenue"),
        count("invoice_no").alias("transaction_count")
    ) \
        .orderBy("month")

    monthly_sales.show()
    monthly_sales.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/monthly_sales_pattern")


# ========= Batch Analysis 5: Purchase Basket Analysis (Simplified) =========
def analyze_purchase_baskets():
    print("Performing purchase basket analysis...")

    basket_df = df.select("invoice_no", "category")

    basket_pairs = basket_df.alias("a").join(
        basket_df.alias("b"),
        (col("a.invoice_no") == col("b.invoice_no")) & (col("a.category") != col("b.category"))
    ).select(
        col("a.category").alias("category1"),
        col("b.category").alias("category2")
    )

    pair_counts = basket_pairs.groupBy("category1", "category2") \
        .count() \
        .orderBy(desc("count"))

    pair_counts.show(20)
    pair_counts.coalesce(1).write.mode("overwrite").option("header", "true").csv("output/category_pairs")


# ========= Execute All Batch Analyses =========
def main():
    global df

    df = df.withColumn(
        "age_group",
        when(col("age") < 18, "Under 18")
        .when((col("age") >= 18) & (col("age") < 25), "18-24")
        .when((col("age") >= 25) & (col("age") < 35), "25-34")
        .when((col("age") >= 35) & (col("age") < 45), "35-44")
        .when((col("age") >= 45) & (col("age") < 55), "45-54")
        .when((col("age") >= 55) & (col("age") < 65), "55-64")
        .otherwise("65+")
    )

    # Print data schema and check for nulls before analysis
    #print("Data Schema:")
    '''df.printSchema()

    print("Null Values Counts:")
    null_counts = []
    for column_name in df.columns:
        null_count = df.filter(col(column_name).isNull()).count()
        null_counts.append((column_name, null_count))

    for column_name, null_count in null_counts:
        print(f"{column_name}: {null_count}")'''

    # 设置Spark不使用科学计数法
    spark.conf.set("spark.sql.legacy.allowNegativeScaleOfDecimal", "true")
    spark.conf.set("spark.sql.legacy.executer.respectNegativeScaleOfDecimal", "true")

    # 确保数值列以常规格式显示（避免科学计数法）
    #for column_name in df.schema.fieldNames():
      #  if df.schema[column_name].dataType.typeName() in ["double", "decimal", "float"]:
        #    df = df.withColumn(column_name, F.format_number(col(column_name), 2))

    #print("Sample data:")
    #df.show(5)

    analyze_customer_segments()
    analyze_product_categories()
    analyze_payment_methods()
    analyze_time_patterns()


    print("Batch processing completed. All outputs saved to elastic search.\n" \
        "——————————————————————————————————————————————————————————————————————\n" \
        "Results are visualized by Kibana, please check at: http://168.138.187.230:5601")

    spark.stop()


if __name__ == "__main__":
    main()
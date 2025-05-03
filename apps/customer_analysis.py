from pyspark.sql import SparkSession
from pyspark.sql.functions import when
import matplotlib.pyplot as plt

# Initializing Spark Session
spark = SparkSession.builder.appName("CustomerSegmentation").getOrCreate()
# Loading The Dataset
customer_df = spark.read.csv("hdfs://namenode:9000/user/flume/raw/data/spool/E-commerceCustomerBehavior-Sheet.csv.1746246584074.tmp", header=True, inferSchema=True)

# Handling missing values
customer_df = customer_df.na.drop()

# Create age groups
customer_df_age_groups = customer_df.withColumn("Age Group",
                       when(customer_df["Age"] < 35, "Under 35")
                       .when((customer_df["Age"] >= 35) & (customer_df["Age"] <= 50), "Between 35-50")
                       .otherwise("Over 50"))

# Group by Membership Type, Age Group, and City
customer_segments = customer_df_age_groups.groupBy("Membership Type", "Age Group", "City").agg({
    "Total Spend": "mean",
    "Items Purchased": "sum"
})

# Group by City separately
city_segments = customer_df_age_groups.groupBy("City").agg({
    "Total Spend": "mean",
    "Items Purchased": "sum"
})

# Show all segments in a single DataFrame
print('Customer Segmentation Result :  ')
customer_segments.show()

# Show city segments separately
city_segments.show()

# Filtering customers at risk based on Days Since Last Purchase and Satisfaction Level
at_risk_customers = customer_df.filter((customer_df["Days Since Last Purchase"] > 30) & (customer_df["Satisfaction Level"] == "Unsatisfied"))

# Show customers at risk
print('Customers at Risk Result : ')
at_risk_customers.show()



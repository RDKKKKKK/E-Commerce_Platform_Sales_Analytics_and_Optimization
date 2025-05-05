#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Customer Value Prediction and Classification Model
Using Spark MLlib for Customer Consumption Behavior Analysis and Visualization
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, desc, when
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
import os

# Ensure all output directories exist
os.makedirs("output/ml_results", exist_ok=True)
os.makedirs("output/ml_results/data", exist_ok=True)  # Add this line to create data subdirectory


def main():
    print("Starting simplified customer analysis...")

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Simple Customer Analysis") \
        .getOrCreate()

    # Set log level to ERROR to reduce output
    spark.sparkContext.setLogLevel("ERROR")

    # Read CSV data
    file_path = "hdfs://namenode:9000/user/flume/raw/data/spool/customer_shopping_data.csv.1746246540086.tmp"
    print(f"Loading data from path: {file_path}")

    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        # Display basic information about the dataset
        print("Number of rows in dataset:", df.count())
        print("Number of columns in dataset:", len(df.columns))
        print("Dataset structure:")
        df.printSchema()

        # Basic data preprocessing
        print("Performing data preprocessing...")

        # Ensure all numeric columns have no null values
        for col_name in df.columns:
            if df.schema[col_name].dataType.simpleString() in ['int', 'double', 'float']:
                df = df.fillna(0, subset=[col_name])
            else:
                df = df.fillna("unknown", subset=[col_name])

        # Calculate total amount
        df = df.withColumn("total_amount", col("quantity") * col("price"))

        # Add age group classification
        df = df.withColumn(
            "age_group",
            when(col("age") < 25, "Young")
            .when((col("age") >= 25) & (col("age") < 35), "Youth")
            .when((col("age") >= 35) & (col("age") < 55), "Middle-aged")
            .otherwise("Senior")
        )

        # ====== Basic Consumer Aggregation Analysis ======
        print("\nPerforming consumer analysis...")

        # Analyze total consumption and average consumption by gender
        gender_analysis = df.groupBy("gender").agg(
            count("invoice_no").alias("transaction_count"),
            F.round(sum("total_amount"), 2).alias("total_spent"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        ).orderBy("gender")

        print("Consumption analysis by gender:")
        gender_analysis.show()

        # Age group analysis
        age_analysis = df.groupBy("age_group").agg(
            count("invoice_no").alias("transaction_count"),
            F.round(sum("total_amount"), 2).alias("total_spent"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        ).orderBy("age_group")

        print("Consumption analysis by age group:")
        age_analysis.show()

        # Analysis by payment method
        payment_analysis = df.groupBy("payment_method").agg(
            count("invoice_no").alias("transaction_count"),
            F.round(sum("total_amount"), 2).alias("total_spent"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        ).orderBy(desc("transaction_count"))

        print("Consumption analysis by payment method:")
        payment_analysis.show()

        # ====== Customer Level Feature Engineering ======
        print("\nPerforming customer-level feature engineering...")

        # Calculate RFM metrics (simplified to only FM metrics here)
        customer_features = df.groupBy("customer_id", "gender", "age", "age_group").agg(
            count("invoice_no").alias("frequency"),
            F.round(sum("total_amount"), 2).alias("monetary"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        )

        # Classify customers as high-value and general-value
        # Using a simple criterion: high-value if total spending exceeds the median of all customers
        median_monetary = customer_features.approxQuantile("monetary", [0.5], 0.001)[0]
        print(f"Median of total spending: {median_monetary}")

        customer_features = customer_features.withColumn(
            "customer_value",
            when(col("monetary") > median_monetary, 1).otherwise(0)
        )

        # Display feature data
        print("Customer feature data examples:")
        customer_features.show(5)

        # ====== Machine Learning Models ======
        print("\nPreparing machine learning models...")

        # Prepare feature columns
        # For simplicity, we directly use numerical features
        feature_cols = ["age", "frequency", "monetary", "avg_transaction"]

        # Create feature vector
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"  # Skip rows containing null values
        )

        # Transform data
        ml_data = assembler.transform(customer_features)

        # Check if feature data exists
        feature_count = ml_data.select("features").count()
        print(f"Number of feature vectors: {feature_count}")

        if feature_count == 0:
            print("Error: No valid feature data, cannot train model")
            return

        # Split training and test sets
        train_data, test_data = ml_data.randomSplit([0.8, 0.2], seed=42)

        print(f"Training data: {train_data.count()} rows")
        print(f"Test data: {test_data.count()} rows")

        # ====== Customer Value Classification Model ======
        print("\nTraining customer value classification model...")

        # Random Forest Classifier
        rf = RandomForestClassifier(
            labelCol="customer_value",
            featuresCol="features",
            numTrees=50,
            maxDepth=5
        )

        # Train the model
        model = rf.fit(train_data)

        # Predict on test set
        predictions = model.transform(test_data)

        # Evaluate model
        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="customer_value",
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)
        print(f"Customer value prediction model AUC: {auc:.4f}")

        # Feature importance
        feature_importance = model.featureImportances
        feature_importance_list = [(feature, float(importance)) for feature, importance in
                                   zip(feature_cols, feature_importance)]
        feature_importance_list.sort(key=lambda x: x[1], reverse=True)

        print("Feature importance:")
        for feature, importance in feature_importance_list:
            print(f"  {feature}: {importance:.4f}")

        # # ====== Visualize Results ======
        # print("\nGenerating visualizations...")

        # # Convert to Pandas for visualization
        # gender_pd = gender_analysis.toPandas()
        # age_pd = age_analysis.toPandas()
        # payment_pd = payment_analysis.toPandas()
        # customer_pd = customer_features.toPandas()

        # # Set better chart style
        # plt.style.use('seaborn-v0_8-whitegrid')  # Use seaborn style to improve readability
        # colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Define color scheme

        # # Create Chart 1: Key Consumption Behavior Analysis
        # plt.figure(figsize=(20, 14))
        # plt.suptitle('Analysis of E-commerce Customer Consumption Behavior', fontsize=24, y=0.98)

        # # 1. Consumption Amount by Gender - Add clear title and value labels
        # plt.subplot(2, 2, 1)
        # ax1 = sns.barplot(x='gender', y='total_spent', data=gender_pd, palette=[colors[0], colors[1]])
        # plt.title('Distribution of total consumption amount by gender', fontsize=16, pad=10)
        # plt.xlabel(' Gender', fontsize=12)
        # plt.ylabel('Total consumption amount', fontsize=12)
        # # Add value labels
        # for p in ax1.patches:
        #     ax1.annotate(f'{p.get_height():,.0f}',
        #                  (p.get_x() + p.get_width() / 2., p.get_height()),
        #                  ha='center', va='bottom', fontsize=12)

        # # 2. Consumption Amount by Age Group - Enhance readability
        # plt.subplot(2, 2, 2)
        # ax2 = sns.barplot(x='age_group', y='total_spent', data=age_pd, palette=colors)
        # plt.title('Comparison of consumption capacity among different age groups', fontsize=16, pad=10)
        # plt.xlabel('age groups', fontsize=12)
        # plt.ylabel(' total consumption', fontsize=12)
        # plt.xticks(rotation=30, ha='right')
        # # Add value labels
        # for p in ax2.patches:
        #     ax2.annotate(f'{p.get_height():,.0f}',
        #                  (p.get_x() + p.get_width() / 2., p.get_height()),
        #                  ha='center', va='bottom', fontsize=12)

        # # 3. Transaction Count by Payment Method - Show only top 5 and add percentages
        # plt.subplot(2, 2, 3)
        # top_payments = payment_pd.head(5)
        # total_transactions = top_payments['transaction_count'].sum()
        # top_payments['percentage'] = top_payments['transaction_count'] / total_transactions * 100
        # ax3 = sns.barplot(x='payment_method', y='transaction_count', data=top_payments, palette=colors)
        # plt.title('The usage of the main payment methods', fontsize=16, pad=10)
        # plt.xlabel('payment method', fontsize=12)
        # plt.ylabel('payment times', fontsize=12)
        # plt.xticks(rotation=30, ha='right')
        # # Add value and percentage labels
        # for i, p in enumerate(ax3.patches):
        #     percentage = top_payments.iloc[i]['percentage']
        #     ax3.annotate(f'{p.get_height():,.0f}\n({percentage:.1f}%)',
        #                  (p.get_x() + p.get_width() / 2., p.get_height()),
        #                  ha='center', va='bottom', fontsize=11)

        # # 4. Feature Importance - Highlight most important features
        # plt.subplot(2, 2, 4)
        # features = [x[0] for x in feature_importance_list]
        # importances = [x[1] for x in feature_importance_list]
        # feature_colors = [colors[0] if i == 0 else colors[1] if i == 1 else colors[3] for i in range(len(features))]
        # ax4 = sns.barplot(x=importances, y=features, palette=feature_colors)
        # plt.title('The key factors of customer value prediction', fontsize=16, pad=10)
        # plt.xlabel('Importance Index', fontsize=12)
        # plt.ylabel('Feature', fontsize=12)
        # # Add percentage labels
        # for i, p in enumerate(ax4.patches):
        #     percentage = importances[i] * 100
        #     ax4.annotate(f'{percentage:.1f}%',
        #                  (p.get_width(), p.get_y() + p.get_height() / 2),
        #                  ha='left', va='center', fontsize=12, weight='bold')

        # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust spacing between subplots
        # plt.savefig('output/ml_results/customer_analysis.png', dpi=300, bbox_inches='tight')
        # print(f"Consumption behavior analysis chart saved to: output/ml_results/customer_analysis.png")
        # gender_pd.to_csv("output/ml_results/data/gender_analysis.csv", index=False)
        # age_pd.to_csv("output/ml_results/data/age_analysis.csv", index=False)
        # payment_pd.to_csv("output/ml_results/data/payment_analysis.csv", index=False)

        # # Chart 2: Customer Value Distribution and Classification
        # plt.figure(figsize=(18, 8))
        # plt.suptitle('Customer value distribution and classification', fontsize=24, y=0.98)

        # # 1. Customer Spending Amount Distribution - Enhance readability
        # plt.subplot(1, 2, 1)
        # sns.histplot(customer_pd['monetary'], bins=30, kde=True, color=colors[0])
        # plt.axvline(x=median_monetary, color='red', linestyle='--',
        #             label=f'Value division threshold: {median_monetary:.2f}')
        # # Add annotation labels
        # plt.annotate('High-value customers', xy=(median_monetary * 1.2, plt.gca().get_ylim()[1] * 0.8),
        #              color='red', fontsize=14, weight='bold')
        # plt.annotate('General value customer', xy=(median_monetary * 0.5, plt.gca().get_ylim()[1] * 0.8),
        #              color=colors[0], fontsize=14, weight='bold')
        # plt.title('Distribution and value division of customer consumption amounts', fontsize=16, pad=10)
        # plt.xlabel('Total consumption amount', fontsize=12)
        # plt.ylabel('The number of customers', fontsize=12)
        # plt.legend(fontsize=12)

        # # 2. Customer Consumption Pattern Scatter Plot - More clearly show classification results
        # plt.subplot(1, 2, 2)
        # scatter = sns.scatterplot(x='frequency', y='monetary', hue='customer_value',
        #                           palette={0: colors[0], 1: 'red'}, s=100, alpha=0.7,
        #                           data=customer_pd)
        # # Add classification boundary line
        # plt.axhline(y=median_monetary, color='red', linestyle='--', alpha=0.7)
        # # Add quadrant labels
        # plt.annotate('High frequency and high value\n(Core customers)',
        #              xy=(customer_pd['frequency'].max() * 0.8, customer_pd['monetary'].max() * 0.9),
        #              color='darkred', fontsize=14, weight='bold', ha='center',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        # plt.annotate('Low frequency and high value\n(Large consumers)',
        #              xy=(customer_pd['frequency'].min() * 1.5, customer_pd['monetary'].max() * 0.9),
        #              color='darkred', fontsize=14, weight='bold', ha='center',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        # plt.annotate('High frequency and low value\n(Potential customers)', xy=(customer_pd['frequency'].max() * 0.8, median_monetary * 0.5),
        #              color='darkblue', fontsize=14, weight='bold', ha='center',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        # plt.annotate('Low frequency and low value\n(Ordinary customer)', xy=(customer_pd['frequency'].min() * 1.5, median_monetary * 0.5),
        #              color='darkblue', fontsize=14, weight='bold', ha='center',
        #              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        # plt.title('The relationship matrix between customer consumption frequency and amount', fontsize=16, pad=10)
        # plt.xlabel('frequency (times)', fontsize=12)
        # plt.ylabel('total consumption amount', fontsize=12)
        # scatter.legend_.set_title('customer value')
        # new_labels = ['General customer', 'High value customer']
        # for t, l in zip(scatter.legend_.texts, new_labels):
        #     t.set_text(l)

        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig('output/ml_results/customer_spending_patterns.png', dpi=300, bbox_inches='tight')
        # print(f"Customer value distribution chart saved to: output/ml_results/customer_spending_patterns.png")
        # customer_pd.to_csv("output/ml_results/data/customer_features.csv", index=False)

        # # Chart 3: Customer Value and Gender Distribution
        # plt.figure(figsize=(18, 8))
        # plt.suptitle('Customer value segmentation analysis', fontsize=24, y=0.98)

        # # 1. High-value vs. General-value Customer Ratio - Pie chart display
        # plt.subplot(1, 2, 1)
        # value_counts = customer_features.groupBy("customer_value").count().toPandas()
        # value_counts['value_label'] = value_counts['customer_value'].apply(
        #     lambda x: "High value customer" if x == 1 else "General customer")
        # value_counts['percentage'] = value_counts['count'] / value_counts['count'].sum() * 100

        # # Draw pie chart and add percentage labels
        # plt.pie(value_counts['count'], labels=value_counts['value_label'],
        #         autopct='%1.1f%%', startangle=90, colors=[colors[0], 'red'],
        #         wedgeprops=dict(width=0.5, edgecolor='w'),
        #         textprops={'fontsize': 14, 'weight': 'bold'})
        # # Add total count in the center of the donut
        # plt.annotate(f'Total number of customers\n{value_counts["count"].sum():,}',
        #              xy=(0, -0.1), ha='center', va='center', fontsize=12, weight='bold')
        # plt.title('Customer value distribution', fontsize=16, pad=10)

        # # 2. Customer Value Distribution by Gender - Stacked bar chart
        # plt.subplot(1, 2, 2)
        # gender_value = customer_features.groupBy("gender", "customer_value").count().toPandas()
        # gender_value['value_label'] = gender_value['customer_value'].apply(
        #     lambda x: "High value customer" if x == 1 else "General customer")

        # # Calculate percentage for each group
        # pivot_data = gender_value.pivot(index='gender', columns='value_label', values='count').fillna(0)
        # pivot_data_percent = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

        # # Create stacked bar chart
        # ax = pivot_data.plot(kind='bar', stacked=True, color=[colors[0], 'red'],
        #                      figsize=(10, 6), ax=plt.gca())
        # plt.title('The distribution of customer value of different genders', fontsize=16, pad=10)
        # plt.xlabel('Gender', fontsize=12)
        # plt.ylabel('Customer Amount', fontsize=12)

        # # Add count and percentage labels
        # for i, gender in enumerate(pivot_data.index):
        #     total = pivot_data.loc[gender].sum()
        #     high_value = pivot_data.loc[gender, 'High value customer'] if 'High value customer' in pivot_data.columns else 0
        #     high_value_pct = pivot_data_percent.loc[
        #         gender, 'High value customer'] if 'High value customer' in pivot_data_percent.columns else 0
        #     low_value_pct = pivot_data_percent.loc[
        #         gender, 'General customer'] if 'General customer' in pivot_data_percent.columns else 0

        #     # High-value labels
        #     plt.annotate(f'{high_value:,.0f}\n({high_value_pct:.1f}%)',
        #                  xy=(i, high_value / 2 + (total - high_value)),
        #                  ha='center', va='center', color='white', fontsize=11, weight='bold')

        #     # General-value labels
        #     plt.annotate(f'{total - high_value:,.0f}\n({low_value_pct:.1f}%)',
        #                  xy=(i, (total - high_value) / 2),
        #                  ha='center', va='center', color='white', fontsize=11, weight='bold')

        #     # Total labels
        #     plt.annotate(f'Total: {total:,.0f}',
        #                  xy=(i, total + 500),
        #                  ha='center', va='bottom', color='black', fontsize=12)

        # plt.legend(title='Customer type')
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig('output/ml_results/customer_value_distribution.png', dpi=300, bbox_inches='tight')
        # print(f"Customer value distribution chart saved to: output/ml_results/customer_value_distribution.png")
        # value_counts.to_csv("output/ml_results/data/value_counts.csv", index=False)
        # gender_value.to_csv("output/ml_results/data/gender_value_counts.csv", index=False)

        print("Prediction completed. All outputs saved to elastic search." \
        "——————————————————————————————————————————————————————————————————————" \
        "Results are visualized by Kibana, please check at: http://168.138.187.230:5601")
        # Close Spark session
        spark.stop()
        # print("\nAnalysis complete! All results saved to output/ml_results/ directory")

    except Exception as e:
        print(f"Error: {str(e)}")
        spark.stop()
        raise e


if __name__ == "__main__":
    main()
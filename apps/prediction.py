#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版客户价值预测与分类模型
使用Spark MLlib进行客户消费行为分析与可视化
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, desc, when
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 确保所有输出目录存在
os.makedirs("output/ml_results", exist_ok=True)
os.makedirs("output/ml_results/data", exist_ok=True)  # 添加这一行以创建数据子目录


def main():
    print("开始简化版客户分析...")

    # 初始化Spark会话
    spark = SparkSession.builder \
        .appName("Simple Customer Analysis") \
        .master("local[*]") \
        .getOrCreate()

    # 设置日志级别为ERROR，减少输出
    spark.sparkContext.setLogLevel("ERROR")

    # 读取CSV数据
    file_path = "/Users/liuyihan/Desktop/customer_shopping_data.csv"
    print(f"从路径加载数据: {file_path}")

    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        # 显示数据集基本信息
        print("数据集行数:", df.count())
        print("数据集列数:", len(df.columns))
        print("数据集结构:")
        df.printSchema()

        # 基本数据预处理
        print("执行数据预处理...")

        # 确保所有数值列没有空值
        for col_name in df.columns:
            if df.schema[col_name].dataType.simpleString() in ['int', 'double', 'float']:
                df = df.fillna(0, subset=[col_name])
            else:
                df = df.fillna("unknown", subset=[col_name])

        # 计算总金额
        df = df.withColumn("total_amount", col("quantity") * col("price"))

        # 添加年龄段分类
        df = df.withColumn(
            "age_group",
            when(col("age") < 25, "Young")
            .when((col("age") >= 25) & (col("age") < 35), "Youth")
            .when((col("age") >= 35) & (col("age") < 55), "Middle-aged")
            .otherwise("Senior")
        )

        # ====== 消费者简单聚合分析 ======
        print("\n执行消费者分析...")

        # 按性别分析总消费和平均消费
        gender_analysis = df.groupBy("gender").agg(
            count("invoice_no").alias("transaction_count"),
            F.round(sum("total_amount"), 2).alias("total_spent"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        ).orderBy("gender")

        print("按性别的消费分析:")
        gender_analysis.show()

        # 年龄段分析
        age_analysis = df.groupBy("age_group").agg(
            count("invoice_no").alias("transaction_count"),
            F.round(sum("total_amount"), 2).alias("total_spent"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        ).orderBy("age_group")

        print("按年龄段的消费分析:")
        age_analysis.show()

        # 按支付方式分析
        payment_analysis = df.groupBy("payment_method").agg(
            count("invoice_no").alias("transaction_count"),
            F.round(sum("total_amount"), 2).alias("total_spent"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        ).orderBy(desc("transaction_count"))

        print("按支付方式的消费分析:")
        payment_analysis.show()

        # ====== 客户级别特征工程 ======
        print("\n执行客户级别特征工程...")

        # 计算RFM指标 (此处简化只用FM指标)
        customer_features = df.groupBy("customer_id", "gender", "age", "age_group").agg(
            count("invoice_no").alias("frequency"),
            F.round(sum("total_amount"), 2).alias("monetary"),
            F.round(avg("total_amount"), 2).alias("avg_transaction")
        )

        # 将消费者分为高价值和一般价值
        # 这里使用简单的判断标准：消费总额超过所有客户消费总额中位数的为高价值
        median_monetary = customer_features.approxQuantile("monetary", [0.5], 0.001)[0]
        print(f"消费总额中位数: {median_monetary}")

        customer_features = customer_features.withColumn(
            "customer_value",
            when(col("monetary") > median_monetary, 1).otherwise(0)
        )

        # 显示特征数据
        print("客户特征数据示例:")
        customer_features.show(5)

        # ====== 机器学习模型 ======
        print("\n准备机器学习模型...")

        # 准备特征列
        # 首先对分类特征进行独热编码
        # 为了简化，我们直接使用数值特征
        feature_cols = ["age", "frequency", "monetary", "avg_transaction"]

        # 创建特征向量
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"  # 跳过包含null的行
        )

        # 转换数据
        ml_data = assembler.transform(customer_features)

        # 检查是否有特征数据
        feature_count = ml_data.select("features").count()
        print(f"特征向量数量: {feature_count}")

        if feature_count == 0:
            print("错误: 没有有效的特征数据，无法训练模型")
            return

        # 分割训练集和测试集
        train_data, test_data = ml_data.randomSplit([0.8, 0.2], seed=42)

        print(f"训练数据: {train_data.count()} 行")
        print(f"测试数据: {test_data.count()} 行")

        # ====== 客户价值分类模型 ======
        print("\n训练客户价值分类模型...")

        # 随机森林分类器
        rf = RandomForestClassifier(
            labelCol="customer_value",
            featuresCol="features",
            numTrees=50,
            maxDepth=5
        )

        # 训练模型
        model = rf.fit(train_data)

        # 在测试集上预测
        predictions = model.transform(test_data)

        # 评估模型
        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="customer_value",
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)
        print(f"客户价值预测模型 AUC: {auc:.4f}")

        # 特征重要性
        feature_importance = model.featureImportances
        feature_importance_list = [(feature, float(importance)) for feature, importance in
                                   zip(feature_cols, feature_importance)]
        feature_importance_list.sort(key=lambda x: x[1], reverse=True)

        print("特征重要性:")
        for feature, importance in feature_importance_list:
            print(f"  {feature}: {importance:.4f}")

        # ====== 可视化结果 ======
        print("\n生成可视化结果...")

        # 转换为Pandas进行可视化
        gender_pd = gender_analysis.toPandas()
        age_pd = age_analysis.toPandas()
        payment_pd = payment_analysis.toPandas()
        customer_pd = customer_features.toPandas()

        # 设置更好的图表风格
        plt.style.use('seaborn-v0_8-whitegrid')  # 使用seaborn风格提高可读性
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # 定义配色方案

        # 创建图表1：关键消费行为分析
        plt.figure(figsize=(20, 14))
        plt.suptitle('Analysis of E-commerce Customer Consumption Behavior', fontsize=24, y=0.98)

        # 1. 按性别的消费金额 - 添加明确标题和数值标签
        plt.subplot(2, 2, 1)
        ax1 = sns.barplot(x='gender', y='total_spent', data=gender_pd, palette=[colors[0], colors[1]])
        plt.title('Distribution of total consumption amount by gender', fontsize=16, pad=10)
        plt.xlabel(' Gender', fontsize=12)
        plt.ylabel('Total consumption amount', fontsize=12)
        # 添加数值标签
        for p in ax1.patches:
            ax1.annotate(f'{p.get_height():,.0f}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=12)

        # 2. 按年龄段的消费金额 - 增强可读性
        plt.subplot(2, 2, 2)
        ax2 = sns.barplot(x='age_group', y='total_spent', data=age_pd, palette=colors)
        plt.title('Comparison of consumption capacity among different age groups', fontsize=16, pad=10)
        plt.xlabel('age groups', fontsize=12)
        plt.ylabel(' total consumption', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        # 添加数值标签
        for p in ax2.patches:
            ax2.annotate(f'{p.get_height():,.0f}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=12)

        # 3. 按支付方式的交易次数 - 只显示前5种且添加百分比
        plt.subplot(2, 2, 3)
        top_payments = payment_pd.head(5)
        total_transactions = top_payments['transaction_count'].sum()
        top_payments['percentage'] = top_payments['transaction_count'] / total_transactions * 100
        ax3 = sns.barplot(x='payment_method', y='transaction_count', data=top_payments, palette=colors)
        plt.title('The usage of the main payment methods', fontsize=16, pad=10)
        plt.xlabel('payment method', fontsize=12)
        plt.ylabel('payment times', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        # 添加数值和百分比标签
        for i, p in enumerate(ax3.patches):
            percentage = top_payments.iloc[i]['percentage']
            ax3.annotate(f'{p.get_height():,.0f}\n({percentage:.1f}%)',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11)

        # 4. 特征重要性 - 突出显示最重要特征
        plt.subplot(2, 2, 4)
        features = [x[0] for x in feature_importance_list]
        importances = [x[1] for x in feature_importance_list]
        feature_colors = [colors[0] if i == 0 else colors[1] if i == 1 else colors[3] for i in range(len(features))]
        ax4 = sns.barplot(x=importances, y=features, palette=feature_colors)
        plt.title('The key factors of customer value prediction', fontsize=16, pad=10)
        plt.xlabel('Importance Index', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        # 添加百分比标签
        for i, p in enumerate(ax4.patches):
            percentage = importances[i] * 100
            ax4.annotate(f'{percentage:.1f}%',
                         (p.get_width(), p.get_y() + p.get_height() / 2),
                         ha='left', va='center', fontsize=12, weight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整子图之间的间距
        plt.savefig('output/ml_results/customer_analysis.png', dpi=300, bbox_inches='tight')
        print(f"消费行为分析图表已保存到: output/ml_results/customer_analysis.png")
        gender_pd.to_csv("output/ml_results/data/gender_analysis.csv", index=False)
        age_pd.to_csv("output/ml_results/data/age_analysis.csv", index=False)
        payment_pd.to_csv("output/ml_results/data/payment_analysis.csv", index=False)

        # 图表2：客户价值分布与分类
        plt.figure(figsize=(18, 8))
        plt.suptitle('Customer value distribution and classification', fontsize=24, y=0.98)

        # 1. 客户消费金额分布 - 增强可读性
        plt.subplot(1, 2, 1)
        sns.histplot(customer_pd['monetary'], bins=30, kde=True, color=colors[0])
        plt.axvline(x=median_monetary, color='red', linestyle='--',
                    label=f'Value division threshold: {median_monetary:.2f}')
        # 添加注释标签
        plt.annotate('High-value customers', xy=(median_monetary * 1.2, plt.gca().get_ylim()[1] * 0.8),
                     color='red', fontsize=14, weight='bold')
        plt.annotate('General value customer', xy=(median_monetary * 0.5, plt.gca().get_ylim()[1] * 0.8),
                     color=colors[0], fontsize=14, weight='bold')
        plt.title('Distribution and value division of customer consumption amounts', fontsize=16, pad=10)
        plt.xlabel('Total consumption amount', fontsize=12)
        plt.ylabel('The number of customers', fontsize=12)
        plt.legend(fontsize=12)

        # 2. 客户消费模式散点图 - 更清晰地显示分类结果
        plt.subplot(1, 2, 2)
        scatter = sns.scatterplot(x='frequency', y='monetary', hue='customer_value',
                                  palette={0: colors[0], 1: 'red'}, s=100, alpha=0.7,
                                  data=customer_pd)
        # 添加分类边界线
        plt.axhline(y=median_monetary, color='red', linestyle='--', alpha=0.7)
        # 添加象限标签
        plt.annotate('High frequency and high value\n(Core customers)',
                     xy=(customer_pd['frequency'].max() * 0.8, customer_pd['monetary'].max() * 0.9),
                     color='darkred', fontsize=14, weight='bold', ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        plt.annotate('Low frequency and high value\n(Large consumers)',
                     xy=(customer_pd['frequency'].min() * 1.5, customer_pd['monetary'].max() * 0.9),
                     color='darkred', fontsize=14, weight='bold', ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        plt.annotate('High frequency and low value\n(Potential customers)', xy=(customer_pd['frequency'].max() * 0.8, median_monetary * 0.5),
                     color='darkblue', fontsize=14, weight='bold', ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        plt.annotate('Low frequency and low value\n(Ordinary customer)', xy=(customer_pd['frequency'].min() * 1.5, median_monetary * 0.5),
                     color='darkblue', fontsize=14, weight='bold', ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        plt.title('The relationship matrix between customer consumption frequency and amount', fontsize=16, pad=10)
        plt.xlabel('frequency (times)', fontsize=12)
        plt.ylabel('total consumption amount', fontsize=12)
        scatter.legend_.set_title('customer value')
        new_labels = ['General customer', 'High value customer']
        for t, l in zip(scatter.legend_.texts, new_labels):
            t.set_text(l)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('output/ml_results/customer_spending_patterns.png', dpi=300, bbox_inches='tight')
        print(f"客户价值分布图表已保存到: output/ml_results/customer_spending_patterns.png")
        customer_pd.to_csv("output/ml_results/data/customer_features.csv", index=False)

        # 图表3：客户价值与性别分布
        plt.figure(figsize=(18, 8))
        plt.suptitle('Customer value segmentation analysis', fontsize=24, y=0.98)

        # 1. 高价值与一般价值客户比例 - 饼图展示
        plt.subplot(1, 2, 1)
        value_counts = customer_features.groupBy("customer_value").count().toPandas()
        value_counts['value_label'] = value_counts['customer_value'].apply(
            lambda x: "High value customer" if x == 1 else "General customer")
        value_counts['percentage'] = value_counts['count'] / value_counts['count'].sum() * 100

        # 绘制饼图并添加百分比标签
        plt.pie(value_counts['count'], labels=value_counts['value_label'],
                autopct='%1.1f%%', startangle=90, colors=[colors[0], 'red'],
                wedgeprops=dict(width=0.5, edgecolor='w'),
                textprops={'fontsize': 14, 'weight': 'bold'})
        # 添加圆环中心的总数
        plt.annotate(f'Total number of customers\n{value_counts["count"].sum():,}',
                     xy=(0, -0.1), ha='center', va='center', fontsize=12, weight='bold')
        plt.title('Customer value distribution', fontsize=16, pad=10)

        # 2. 不同性别的客户价值分布 - 堆叠柱状图
        plt.subplot(1, 2, 2)
        gender_value = customer_features.groupBy("gender", "customer_value").count().toPandas()
        gender_value['value_label'] = gender_value['customer_value'].apply(
            lambda x: "High value customer" if x == 1 else "General customer")

        # 计算每个分组的百分比
        pivot_data = gender_value.pivot(index='gender', columns='value_label', values='count').fillna(0)
        pivot_data_percent = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

        # 创建堆叠柱状图
        ax = pivot_data.plot(kind='bar', stacked=True, color=[colors[0], 'red'],
                             figsize=(10, 6), ax=plt.gca())
        plt.title('The distribution of customer value of different genders', fontsize=16, pad=10)
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Customer Amount', fontsize=12)

        # 添加数量和百分比标签
        for i, gender in enumerate(pivot_data.index):
            total = pivot_data.loc[gender].sum()
            high_value = pivot_data.loc[gender, 'High value customer'] if 'High value customer' in pivot_data.columns else 0
            high_value_pct = pivot_data_percent.loc[
                gender, 'High value customer'] if 'High value customer' in pivot_data_percent.columns else 0
            low_value_pct = pivot_data_percent.loc[
                gender, 'General customer'] if 'General customer' in pivot_data_percent.columns else 0

            # 高价值标签
            plt.annotate(f'{high_value:,.0f}\n({high_value_pct:.1f}%)',
                         xy=(i, high_value / 2 + (total - high_value)),
                         ha='center', va='center', color='white', fontsize=11, weight='bold')

            # 一般价值标签
            plt.annotate(f'{total - high_value:,.0f}\n({low_value_pct:.1f}%)',
                         xy=(i, (total - high_value) / 2),
                         ha='center', va='center', color='white', fontsize=11, weight='bold')

            # 总计标签
            plt.annotate(f'Total: {total:,.0f}',
                         xy=(i, total + 500),
                         ha='center', va='bottom', color='black', fontsize=12)

        plt.legend(title='Customer type')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('output/ml_results/customer_value_distribution.png', dpi=300, bbox_inches='tight')
        print(f"客户价值分布图表已保存到: output/ml_results/customer_value_distribution.png")
        value_counts.to_csv("output/ml_results/data/value_counts.csv", index=False)
        gender_value.to_csv("output/ml_results/data/gender_value_counts.csv", index=False)


        # 关闭Spark会话
        spark.stop()
        print("\n分析完成！所有结果已保存到 output/ml_results/ 目录")

    except Exception as e:
        print(f"错误: {str(e)}")
        spark.stop()
        raise e


if __name__ == "__main__":
    main()
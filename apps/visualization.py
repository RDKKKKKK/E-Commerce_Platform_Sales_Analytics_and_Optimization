import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import glob

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100

# Create output directory
if not os.path.exists("visualization"):
    os.makedirs("visualization")

# Set custom color scheme
colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
          "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]


# Helper function: Find and read CSV file
def find_and_read_csv(directory):
    """Find CSV files in the directory and read the first one"""
    files = glob.glob(f"{directory}/*.csv") or glob.glob(f"{directory}/part-*")
    if not files:
        print(f"Warning: No CSV files found in {directory}")
        return pd.DataFrame()

    print(f"Reading file: {files[0]}")
    return pd.read_csv(files[0])


# 1. Customer Segmentation Charts
def create_customer_segment_charts():
    print("Creating customer segmentation charts...")

    # Read customer segmentation data
    customer_segments = find_and_read_csv("output/customer_segments")
    age_gender_analysis = find_and_read_csv("output/age_gender_analysis")

    if customer_segments.empty or age_gender_analysis.empty:
        print("Cannot create customer segmentation charts, data is missing")
        return

    # 1.1 Customer Segment Pie Chart
    segments = customer_segments.groupby('customer_segment').size().reset_index(name='count')
    plt.figure(figsize=(10, 8))
    plt.pie(segments['count'], labels=segments['customer_segment'], autopct='%1.1f%%',
            startangle=90, colors=colors, shadow=False, wedgeprops={'edgecolor': 'w'})
    plt.title('Customer Segment Distribution', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("visualization/customer_segments_pie.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 1.2 Age-Gender Average Spending Heatmap
    try:
        pivot_data = age_gender_analysis.pivot(index='age_group', columns='gender', values='avg_total_spent')

        # Ensure correct age group order
        age_order = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        # Filter out age groups that don't exist in the data
        age_order = [age for age in age_order if age in pivot_data.index]
        if age_order:
            pivot_data = pivot_data.reindex(age_order)

        plt.figure(figsize=(12, 8))
        custom_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#E6F3FC", "#0662B0"])
        heatmap = sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap=custom_cmap,
                              linewidths=.5, cbar_kws={'label': 'Average Spending'})
        plt.title('Average Spending by Age Group and Gender', fontsize=16)
        plt.xlabel('Gender', fontsize=14)
        plt.ylabel('Age Group', fontsize=14)
        plt.tight_layout()
        plt.savefig("visualization/age_gender_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap: {e}")

    # 1.3 Average Spending by Age Group Bar Chart
    try:
        age_spending = customer_segments.groupby('age_group')['total_spent'].mean().reset_index()

        # Ensure correct age group order
        age_order = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        # Filter out age groups that don't exist in the data
        valid_ages = [age for age in age_order if age in age_spending['age_group'].values]
        if valid_ages:
            age_spending = age_spending.set_index('age_group').reindex(valid_ages).reset_index()

        plt.figure(figsize=(12, 6))
        bars = plt.bar(age_spending['age_group'], age_spending['total_spent'], color=colors[0])
        plt.title('Average Spending by Age Group', fontsize=16)
        plt.xlabel('Age Group', fontsize=14)
        plt.ylabel('Average Spending', fontsize=14)
        plt.xticks(rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig("visualization/age_spending_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")


# 2. Product Category Analysis Charts
def create_product_category_charts():
    print("Creating product category analysis charts...")

    # Read category analysis data
    category_analysis = find_and_read_csv("output/category_analysis")
    monthly_category = find_and_read_csv("output/monthly_category_sales")

    if category_analysis.empty:
        print("Cannot create product category charts, category_analysis data is missing")
        return

    # 2.1 Top 10 Categories Horizontal Bar Chart
    top_categories = category_analysis.sort_values('total_revenue', ascending=False).head(10)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_categories['category'], top_categories['total_revenue'], color=colors[:len(top_categories)])
    plt.title('Top 10 Categories by Revenue', fontsize=16)
    plt.xlabel('Revenue', fontsize=14)
    plt.ylabel('Category', fontsize=14)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 100, bar.get_y() + bar.get_height() / 2.,
                 f'{width:.2f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("visualization/top10_categories.png", dpi=300, bbox_inches='tight')
    plt.close()

    if not monthly_category.empty:
        # 2.2 Monthly Sales Trend for Top 5 Categories
        try:
            top5_categories = category_analysis.sort_values('total_revenue', ascending=False).head(5)[
                'category'].tolist()
            monthly_top5 = monthly_category[monthly_category['category'].isin(top5_categories)]

            if not monthly_top5.empty and 'month' in monthly_top5.columns:
                plt.figure(figsize=(14, 8))
                for i, category in enumerate(top5_categories):
                    category_data = monthly_top5[monthly_top5['category'] == category]
                    if not category_data.empty:
                        plt.plot(category_data['month'], category_data['monthly_revenue'],
                                 marker='o', linewidth=2, label=category, color=colors[i % len(colors)])

                plt.title('Monthly Sales Trend for Top 5 Categories', fontsize=16)
                plt.xlabel('Month', fontsize=14)
                plt.ylabel('Revenue', fontsize=14)
                plt.xticks(range(1, 13))
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig("visualization/monthly_top5_trend.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error creating monthly trend chart: {e}")

    # 2.3 Category Sales vs. Quantity Bubble Chart
    try:
        if 'total_quantity_sold' in category_analysis.columns and 'avg_price' in category_analysis.columns:
            plt.figure(figsize=(12, 8))
            plt.scatter(category_analysis['total_quantity_sold'],
                        category_analysis['total_revenue'],
                        s=category_analysis['avg_price'] * 20,
                        alpha=0.6,
                        c=[colors[i % len(colors)] for i in range(len(category_analysis))],
                        edgecolors='w')

            plt.title('Category Sales Volume vs Revenue vs Price', fontsize=16)
            plt.xlabel('Quantity Sold', fontsize=14)
            plt.ylabel('Revenue', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)

            # Add labels for Top 5 categories
            for i, row in category_analysis.sort_values('total_revenue', ascending=False).head(5).iterrows():
                plt.annotate(row['category'],
                             xy=(row['total_quantity_sold'], row['total_revenue']),
                             xytext=(10, 5), textcoords='offset points',
                             fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig("visualization/category_bubble_chart.png", dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error creating bubble chart: {e}")


# 3. Payment Method Analysis Charts
def create_payment_method_charts():
    print("Creating payment method analysis charts...")

    # Read payment method analysis data
    payment_analysis = find_and_read_csv("output/payment_analysis")
    payment_age_analysis = find_and_read_csv("output/payment_age_analysis")

    if payment_analysis.empty:
        print("Cannot create payment method charts, payment_analysis data is missing")
        return

    # 3.1 Payment Method Distribution Pie Chart
    plt.figure(figsize=(10, 8))
    explode = [0.1 if i == payment_analysis['transaction_count'].idxmax() else 0 for i in range(len(payment_analysis))]

    plt.pie(payment_analysis['transaction_count'],
            labels=payment_analysis['payment_method'],
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            colors=colors,
            shadow=False,
            wedgeprops={'edgecolor': 'w'})

    plt.title('Payment Method Distribution', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("visualization/payment_method_pie.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3.2 Average Transaction Value by Payment Method
    plt.figure(figsize=(12, 6))
    bars = plt.bar(payment_analysis['payment_method'],
                   payment_analysis['avg_transaction_value'],
                   color=colors[:len(payment_analysis)])

    plt.title('Average Transaction Value by Payment Method', fontsize=16)
    plt.xlabel('Payment Method', fontsize=14)
    plt.ylabel('Average Transaction Value', fontsize=14)
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("visualization/payment_avg_transaction.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3.3 Age Group vs Payment Method Heatmap
    if not payment_age_analysis.empty:
        try:
            # Pivot data for heatmap
            payment_pivot = payment_age_analysis.pivot(index='age_group', columns='payment_method',
                                                       values='usage_count')

            # Ensure correct age group order
            age_order = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            # Filter out age groups that don't exist in the data
            valid_ages = [age for age in age_order if age in payment_pivot.index]
            if valid_ages:
                payment_pivot = payment_pivot.reindex(valid_ages)

            plt.figure(figsize=(14, 8))
            heat_colors = sns.color_palette("YlOrRd", as_cmap=True)
            sns.heatmap(payment_pivot, annot=True, fmt='g', cmap=heat_colors,
                        linewidths=.5, cbar_kws={'label': 'Usage Count'})

            plt.title('Payment Method Preference by Age Group', fontsize=16)
            plt.xlabel('Payment Method', fontsize=14)
            plt.ylabel('Age Group', fontsize=14)
            plt.tight_layout()
            plt.savefig("visualization/age_payment_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating payment method heatmap: {e}")


# 4. Time Pattern Analysis Charts
def create_time_pattern_charts():
    print("Creating time pattern analysis charts...")

    # Read time pattern analysis data
    daily_sales = find_and_read_csv("output/daily_sales_pattern")
    monthly_sales = find_and_read_csv("output/monthly_sales_pattern")

    if daily_sales.empty and monthly_sales.empty:
        print("Cannot create time pattern charts, daily_sales and monthly_sales data is missing")
        return

    # 4.1 Weekday Sales Pattern Line Chart
    if not daily_sales.empty and 'day_of_week' in daily_sales.columns:
        try:
            # Add weekday names
            weekday_names = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday',
                             7: 'Saturday'}
            daily_sales['weekday_name'] = daily_sales['day_of_week'].map(weekday_names)

            # Create dual Y-axis chart
            fig, ax1 = plt.subplots(figsize=(12, 7))

            color1 = colors[0]
            ax1.set_xlabel('Day of Week', fontsize=14)
            ax1.set_ylabel('Total Revenue', color=color1, fontsize=14)
            ax1.plot(daily_sales['weekday_name'], daily_sales['total_revenue'], color=color1, marker='o', linewidth=3)
            ax1.tick_params(axis='y', labelcolor=color1)

            ax2 = ax1.twinx()  # Create shared x-axis with second y-axis
            color2 = colors[1]
            ax2.set_ylabel('Transaction Count', color=color2, fontsize=14)
            ax2.plot(daily_sales['weekday_name'], daily_sales['transaction_count'], color=color2, marker='s',
                     linewidth=3)
            ax2.tick_params(axis='y', labelcolor=color2)

            plt.title('Weekly Sales and Transaction Pattern', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.3)
            fig.tight_layout()
            plt.savefig("visualization/weekday_sales_pattern.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating weekday sales line chart: {e}")

    # 4.2 Monthly Sales and Transaction Count Chart
    if not monthly_sales.empty and 'month' in monthly_sales.columns:
        try:
            # Add month names
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                           7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            monthly_sales['month_name'] = monthly_sales['month'].map(month_names)

            # Create bar and line chart combination
            fig, ax1 = plt.subplots(figsize=(14, 8))

            x = np.arange(len(monthly_sales))
            width = 0.6

            # Bar chart for total revenue
            bars = ax1.bar(x, monthly_sales['total_revenue'], width, color=colors[2], alpha=0.7)
            ax1.set_xlabel('Month', fontsize=14)
            ax1.set_ylabel('Total Revenue', fontsize=14)
            ax1.set_xticks(x)
            ax1.set_xticklabels(monthly_sales['month_name'], rotation=45)

            # Add value labels to bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=9)

            # Create shared x-axis with second y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Transaction Count', fontsize=14, color=colors[3])
            ax2.plot(x, monthly_sales['transaction_count'], color=colors[3], marker='o', linewidth=3)
            ax2.tick_params(axis='y', labelcolor=colors[3])

            plt.title('Monthly Sales and Transaction Count', fontsize=16)
            plt.grid(False)
            fig.tight_layout()
            plt.savefig("visualization/monthly_sales_pattern.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 4.3 Revenue vs Transaction Count Scatter Plot
            plt.figure(figsize=(10, 8))

            # Create scatter plot with point sizes based on average transaction value
            monthly_sales['avg_transaction'] = monthly_sales['total_revenue'] / monthly_sales['transaction_count']

            scatter = plt.scatter(monthly_sales['transaction_count'],
                                  monthly_sales['total_revenue'],
                                  s=monthly_sales['avg_transaction'] * 3,
                                  c=[colors[i % len(colors)] for i in range(len(monthly_sales))],
                                  alpha=0.7,
                                  edgecolors='w')

            plt.title('Revenue vs Transaction Count Relationship', fontsize=16)
            plt.xlabel('Transaction Count', fontsize=14)
            plt.ylabel('Total Revenue', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)

            # Add month labels
            for i, row in monthly_sales.iterrows():
                plt.annotate(row['month_name'],
                             xy=(row['transaction_count'], row['total_revenue']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9)

            plt.tight_layout()
            plt.savefig("visualization/sales_transactions_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating monthly sales charts: {e}")


# 5. Dashboard Overview
def create_dashboard():
    print("Creating dashboard overview...")

    # Read analysis data
    customer_segments = find_and_read_csv("output/customer_segments")
    category_analysis = find_and_read_csv("output/category_analysis")
    payment_analysis = find_and_read_csv("output/payment_analysis")
    monthly_sales = find_and_read_csv("output/monthly_sales_pattern")

    if customer_segments.empty or category_analysis.empty or payment_analysis.empty or monthly_sales.empty:
        print("Cannot create dashboard overview, some data is missing")
        return

    try:
        # Create 2x2 dashboard layout
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Customer Segment Pie Chart
        segments = customer_segments.groupby('customer_segment').size().reset_index(name='count')
        axes[0, 0].pie(segments['count'], labels=segments['customer_segment'], autopct='%1.1f%%',
                       startangle=90, colors=colors[:len(segments)], wedgeprops={'edgecolor': 'w'})
        axes[0, 0].set_title('Customer Segment Distribution', fontsize=16)
        axes[0, 0].axis('equal')

        # 2. Top 5 Categories by Revenue
        top_categories = category_analysis.sort_values('total_revenue', ascending=False).head(5)
        bars = axes[0, 1].barh(top_categories['category'], top_categories['total_revenue'],
                               color=colors[:len(top_categories)])
        axes[0, 1].set_title('Top 5 Categories by Revenue', fontsize=16)
        axes[0, 1].set_xlabel('Revenue')

        # 3. Payment Method Distribution
        explode = [0.1 if i == payment_analysis['transaction_count'].idxmax() else 0
                   for i in range(len(payment_analysis))]
        axes[1, 0].pie(payment_analysis['transaction_count'], labels=payment_analysis['payment_method'],
                       autopct='%1.1f%%', startangle=90, explode=explode, colors=colors[:len(payment_analysis)],
                       wedgeprops={'edgecolor': 'w'})
        axes[1, 0].set_title('Payment Method Distribution', fontsize=16)
        axes[1, 0].axis('equal')

        # 4. Monthly Sales Trend
        if 'month' in monthly_sales.columns:
            axes[1, 1].plot(monthly_sales['month'], monthly_sales['total_revenue'],
                            marker='o', linewidth=3, color=colors[0])
            axes[1, 1].set_title('Monthly Sales Trend', fontsize=16)
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Revenue')
            axes[1, 1].set_xticks(monthly_sales['month'])

        plt.tight_layout(pad=5.0)
        plt.savefig("visualization/dashboard_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating dashboard overview: {e}")


# Special function to fix gender category preference chart
def create_gender_category_chart():
    print("Creating gender category preference chart...")

    # Read necessary data files
    category_analysis = find_and_read_csv("output/category_analysis")
    category_preferences = find_and_read_csv("output/category_preferences")

    if category_analysis.empty or category_preferences.empty:
        print("Cannot create gender category chart, data is missing")
        return

    # Print debug information
    print(f"Category analysis columns: {category_analysis.columns.tolist()}")
    print(f"Category preferences columns: {category_preferences.columns.tolist()}")
    if not category_preferences.empty:
        print(f"Category preferences unique categories: {category_preferences['category'].unique()}")
        print(f"Category preferences unique genders: {category_preferences['gender'].unique()}")

    # Get top 5 categories
    top5_categories = category_analysis.sort_values('total_revenue', ascending=False).head(5)['category'].tolist()
    print(f"Top 5 categories: {top5_categories}")

    # Filter category preferences for top 5 categories
    top5_pref = category_preferences[category_preferences['category'].isin(top5_categories)]
    print(f"Filtered preferences data points: {len(top5_pref)}")

    if top5_pref.empty:
        print("No data found for top 5 categories in category_preferences")
        return

    # Group data by category and gender
    gender_pref = top5_pref.groupby(['category', 'gender'])['total_spent'].sum().reset_index()
    print(f"Grouped data points: {len(gender_pref)}")

    # Get unique categories and genders in the filtered data
    categories = gender_pref['category'].unique()
    genders = gender_pref['gender'].unique()
    print(f"Categories in grouped data: {categories}")
    print(f"Genders in grouped data: {genders}")

    # Create the chart
    plt.figure(figsize=(14, 8))

    # Set up for grouped bar chart
    x = np.arange(len(categories))
    width = 0.35 / len(genders) if len(genders) > 1 else 0.35  # Adjust width based on number of genders

    # Plot bars for each gender
    for i, gender in enumerate(genders):
        gender_data = gender_pref[gender_pref['gender'] == gender]
        # Create a dictionary for easier lookup
        gender_dict = dict(zip(gender_data['category'], gender_data['total_spent']))
        # Get values for each category
        values = [gender_dict.get(cat, 0) for cat in categories]

        offset = width * i - width * (len(genders) - 1) / 2 if len(genders) > 1 else 0
        plt.bar(x + offset, values, width, label=gender, color=colors[i % len(colors)])

    plt.title('Gender Preference for Top 5 Categories', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Total Spending', fontsize=14)
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the chart
    plt.savefig("visualization/gender_category_preference.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Gender category preference chart created successfully")


# Execute all chart creation functions
def main():
    print("Starting data visualization creation...")

    # Display files in output directory
    print("Checking files in output directory:")
    for subdir in ["customer_segments", "age_gender_analysis", "category_analysis",
                   "monthly_category_sales", "category_preferences", "payment_analysis",
                   "payment_age_analysis", "daily_sales_pattern", "monthly_sales_pattern"]:
        files = glob.glob(f"output/{subdir}/*")
        print(f"output/{subdir}: {len(files)} files")
        if len(files) > 0:
            print(f"  Example: {files[0]}")

    # Create all visualizations
    create_customer_segment_charts()
    create_product_category_charts()
    create_payment_method_charts()
    create_time_pattern_charts()
    create_dashboard()

    # Add this line to create the gender category chart separately
    create_gender_category_chart()

    print("All data visualizations completed! Stored in visualization directory")

    # Check created charts
    vis_files = glob.glob("visualization/*.png")
    if vis_files:
        print(f"Successfully created {len(vis_files)} charts:")
        for file in vis_files:
            print(f"  - {file}")
    else:
        print("Warning: No charts were created. Check the error messages above.")


if __name__ == "__main__":
    main()
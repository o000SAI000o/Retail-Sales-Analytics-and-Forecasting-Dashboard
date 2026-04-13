import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load the cleaned dataset
df = pd.read_csv("C:/Users/Vishwajeet/OneDrive/Desktop/retail-data-analytics/data/sales/sales.csv")


## Convert Date column to datetime
df['date'] = pd.to_datetime(df['date'])

""" 2. Sales Trends Over Time
Let's analyze monthly sales trends to understand seasonality."""
#create new column for month and year
#Use .dt.to_period('M') to group by month.
df['month'] = df['date'].dt.to_period('M')

#group by monthly sales and calculate total sales
monthly_sales = df.groupby('month')['sales'].sum()

#plot the mpnthly sales trend
plt.figure(figsize=(12,6))
#kind='line'- indicates that a line plot should be created.
monthly_sales.plot(kind='line',marker='o',color='blue')
plt.title("monthly sales trend")
plt.xlabel("month")
plt.ylabel("total sales")
plt.grid(True)
plt.xticks(rotation=45) #The line plt.xticks(rotation=45) is a command used in the Matplotlib library to modify the appearance of the x-axis tick labels in a plot. Specifically, this command rotates the labels on the x-axis by 45 degrees.
plt.show()

"""3. Top-Selling Products
Identify the top 10 best-selling products by total revenue."""
## Calculate total sales by product
top_products = df.groupby('product_id')['sales'].sum().sort_values(ascending=False).head(10)

#plot the top-selling products
plt.figure(figsize=(14,7))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Total sales")
plt.ylabel("product")
plt.grid(True)
plt.show()
""" Tips:

Bar plots are great for ranking products.
Use Seaborn's palette for eye-catching visuals.
"""

"""
 4. Customer Behavior Analysis
Let's analyze purchase patterns by day of the week to find peak buying times.
"""
#create new column for a day of week
df['DayofWeek'] = df['date'].dt.day_name()

#group by day of the week and calculate total sales
day_sales = df.groupby('DayofWeek')['sales'].sum().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

#plot sales by day of the week 
plt.figure(figsize=(10,6))
sns.barplot(x=day_sales.index, y=day_sales.values, palette='coolwarm')
plt.title("Sales by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

"""Use .dt.day_name() for day of the week extraction.
Coolwarm palette effectively shows high vs. low sales days.
"""
"""5. Correlation Analysis
Find correlations between Sales, Quantity, Discounts, etc."""

# Select only numeric columns for correlation matrix calculation
numeric_df = df.select_dtypes(include=['number'])

#calculate the correlation matrix
correlation = numeric_df.corr()

#plot correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()

"""Correlation helps identify relationships between variables.
Look for strong positive/negative correlations for insights.
"""

"""6. Save Visualizations for Power BI
We'll use these visualizations in Power BI later."""
# Save cleaned data and EDA insights
import os

df.to_csv("../data/cleaned_sales_data.csv", index=False)
plt.savefig("../visuals/monthly_sales_trend.png")
plt.savefig("../visuals/top_selling_products.png")
plt.savefig("../visuals/sales_by_day.png")
plt.savefig("../visuals/correlation_heatmap.png")
print("eda performed and fig are saved successfully!")










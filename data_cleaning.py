#1️⃣ Get the Dataset & Set Up Tools
import pandas as pd

#load a dataset
df = pd.read_csv("C:/Users/Vishwajeet/OneDrive/Desktop/retail-data-analytics/data/sales/sales.csv")

#display the first five rows #testing
print(df['sales'].head(50))

#2️⃣ Data Cleaning & Preprocessing
"""We'll clean the dataset to:
Remove missing values
Handle outliers for accurate predictions
Format columns (e.g., dates, categories) for analysis"""

#Let's first inspect the data to understand its structure.
# Inspect data types and missing values
print(df.info())
print(df.isnull().sum())  # Count of missing values per column
print(df.describe())  # Summary statistics
#Focus on columns with NaN (missing values).

""" 2. Handle Missing Values
Replace or drop missing values to clean the dataset:"""
#drop rows with missing values
df.dropna(subset=['sales', 'revenue'], inplace=True)
# Alternatively, fill missing values with 0 or mean
# df['revenue'].fillna(0, inplace=True)
#Tip: If a column is critical (e.g., Revenue), consider filling it with the mean or median instead of dropping it.

#3. Convert Data Types
#Ensure dates are in the correct format for time-series analysis:
# Convert Date column to datetime
df['date'] = pd.to_datetime(df['date'])
print(df.info()) # Confirm the change

#4. Remove Outliers
#Outliers can skew your analysis, especially for sales and revenue data.
"""Outliers are extreme values that can distort statistical 
analyses and lead to misleading results, particularly in financial 
data such as sales and revenue."""
# Remove outliers in Sales column
""" first quartile (Q1) and the third quartile (Q3) of the 'Sales' column using 
the quantile method. The first quartile represents the 25th percentile, while the 
third quartile represents the 75th percentile. These values are used to compute the 
interquartile range (IQR), which is the difference between Q3 and Q1. The IQR measures 
the spread of the middle 50% of the data The DataFrame df is then filtered to include 
only the rows where the 'Sales' values fall within the calculated lower and upper bounds. 
This effectively removes any outliers from the 'Sales' column. Finally, the describe method
is called on the filtered DataFrame to print summary statistics"""
q1 = df['sales'].quantile(0.01)
q3 = df['sales'].quantile(0.99)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df = df[(df['sales'] >= lower_bound) & (df['sales'] <= upper_bound)]
print(df['sales'].describe())    # Check statistics after outlier removal

"""5. Save the Cleaned Data
Store the cleaned data for exploratory data analysis (EDA):"""
# Save cleaned data for EDA
df.to_csv("C:/Users/Vishwajeet/OneDrive/Desktop/retail-data-analytics/data/sales/sales.csv", index=False)
print("Data cleaning completed and saved!")
print(df.columns)
print(df['sales'].unique())    # Check unique values in sales column
print(df['revenue'].unique())  # Check unique values in revenue column
print(df['sales'].dtype)       # Check data type of sales column
print(df['sales'].head(10))
# Check min and max of sales
print(df['sales'].min(), df['sales'].max())

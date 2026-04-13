#Phase 4: Sales Forecasting
"""We’ll build two forecasting models:

Linear Regression – To find relationships between sales and other variables
Time Series Analysis – To capture seasonal patterns and trends
"""
""" 1. Set Up the Forecasting Script
Create a new file: scripts/forecasting.py
Start by loading the cleaned dataset:"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#load the cleaned dataset
df = pd.read_csv("../data/cleaned_sales_data.csv")


# Convert Date column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
 
"""2. Prepare Data for Linear Regression
We’ll predict sales using previous sales as a feature.
"""
#create a lag feature (previous sales)
df['previous_sales'] = df['sales'].shift(1)
df.dropna(inplace=True)

#split data into training and testing sets
x = df[['previous_sales']]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""🎯 Tips:

Lag features help capture time dependencies in sales data.
Use .shift(1) to take the previous day's sales as the input feature."""

# 3. Build and Train Linear Regression Model
#initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train,y_train)

#make predictions
y_pred = model.predict(X_test)

#evaluate model performance 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

"""🎯 Tips:

Mean Squared Error (MSE) shows the average prediction error. Lower is better.
R-squared Score shows how well the model fits the data. Closer to 1 is better.
"""
# 4. Visualize Predictions vs Actual Sales
#plot predictions vs actual sales
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual sales', color='blue')
plt.plot(y_pred, label='Predicted Sales', color='lime')
plt.title('Actual vs predicted sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
"""🎯 Tips:
This visualization demonstrates model accuracy and trends.
Use different colors for actual vs predicted to highlight differences."""
 
"""5. Time Series Analysis with Moving Average
Now, let’s smooth out sales trends using a Moving Average:"""
#calculate 3 month moving average
df['3_month_MA'] = df['sales'].rolling(window=3).mean()

#plot moving average vs actual sales
plt.figure(figsize=(12,6))
plt.plot(df['sales'], label="Actual sales", color='blue')
plt.plot(df['3_month_MA'], label='3-month moving average', color='orange')
plt.title('Sales vs Moving Average')
plt.xlabel('date')
plt.ylabel('sales')
plt.legend()
plt.grid(True)
plt.show()

"""🎯 Tips:

Moving Average helps in smoothing out short-term fluctuations.
It's useful for identifying long-term trends.
"""
# 6. Forecast Future Sales
# Predict sales for the next month
last_sales = df['sales'].iloc[-1] #get the last recorded sales value
next_month_prediction = model.predict([[last_sales]]).reshape(-1, 1)
print(f"Predicted Sales for Next Month: {next_month_prediction[0]}")
"""🎯 Tips:
This prediction adds value to the dashboard by showing future sales trends.
You can extend this to predict multiple months ahead."""

# 7. Save Forecasting Results for Power BI
#We’ll visualize these forecasts in Power BI later.
# Save forecast data for Power BI
df['predicted_sales'] = model.predict(x)
df.to_csv("../data/forecasted_sales_data.csv")

"""Run forecasting.py and check:

Mean Squared Error and R-squared Score
Plot of Actual vs Predicted Sales
Moving Average trend
Predicted sales for the next month"""
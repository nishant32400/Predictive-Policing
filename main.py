import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import mean_squared_error, r2_score

# Loading dataset

data = pd.read_csv("data.csv")

# Feature selection

crime_columns = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']

data['total_crime'] = data[crime_columns].sum(axis=1)

features = ['long', 'lat']

X = data[features]

y = data['total_crime']

# Data scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Splitting data

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, y_pred_lin)

lin_r2 = r2_score(y_test, y_pred_lin)

# Ridge Regression

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)

ridge_r2 = r2_score(y_test, y_pred_ridge)

# Lasso Regression

lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, y_pred_lasso)

lasso_r2 = r2_score(y_test, y_pred_lasso)

# Crime Prediction for New Locations

new_locations = np.array([[77.2, 28.6], [77.1, 28.7]]) # Example coordinates

new_locations_scaled = scaler.transform(new_locations)

predicted_crimes = lin_reg.predict(new_locations_scaled)

# Display predictions with crime types

for i, loc in enumerate(new_locations):

nearest_index = np.argmin(np.sum((X.values - loc) ** 2, axis=1))

crime_info = data.iloc[nearest_index][crime_columns].to_dict()

print(f"Predicted total crimes at location {loc}: {predicted_crimes[i]:.2f}")

print("Crime Breakdown:", crime_info)

print("---")

# Printing results

print(f"Linear Regression MSE: {lin_mse:.2f}, R²: {lin_r2:.2f}")

print(f"Ridge Regression MSE: {ridge_mse:.2f}, R²: {ridge_r2:.2f}")

print(f"Lasso Regression MSE: {lasso_mse:.2f}, R²: {lasso_r2:.2f}")

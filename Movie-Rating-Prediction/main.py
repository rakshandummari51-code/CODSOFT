# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
movies = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

# Display first 5 rows
print("First 5 Rows:")
print(movies.head())

# Dataset information
print("\nDataset Information:")
print(movies.info())

# Check missing values
print("\nMissing Values:")
print(movies.isnull().sum())

# Keep important columns
movies = movies[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]

# Remove missing values
movies.dropna(inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()

movies['Genre'] = label_encoder.fit_transform(movies['Genre'])
movies['Director'] = label_encoder.fit_transform(movies['Director'])
movies['Actor 1'] = label_encoder.fit_transform(movies['Actor 1'])
movies['Actor 2'] = label_encoder.fit_transform(movies['Actor 2'])
movies['Actor 3'] = label_encoder.fit_transform(movies['Actor 3'])

# Features and target
X = movies.drop('Rating', axis=1)
y = movies['Rating']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()

# Sample prediction
sample = pd.DataFrame(
    [[1, 1, 1, 1, 1]],
    columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
)

predicted_rating = model.predict(sample)

print("\nPredicted Movie Rating:")
print(round(predicted_rating[0], 2))
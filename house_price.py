import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset from CSV
df = pd.read_csv("Housing.csv")

# Features and Target
X = df[['area', 'bedrooms']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Predicted Prices:", predictions)
print("MSE:", mean_squared_error(y_test, predictions))

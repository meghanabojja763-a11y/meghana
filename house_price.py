import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv("Housing.csv")
print(df.head())
X = df[['area', 'bedrooms','bathrooms']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train,X_test,y_train,y_test)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predicted Prices:", predictions)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
new_data = pd.DataFrame({'area': [1800, 2200], 'bedrooms': [3, 4]})
predicted_prices = model.predict(new_data)
print("Predicted Prices for new houses:", predicted_prices)

# 📦 Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 🧠 1. Load the dataset
df = pd.read_csv("boston.csv")

# 🧐 2. Explore the data (optional)
print(df.head())
print(df.info())

# 🎯 3. Define features (X) and target (y)
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# ✂️ 4. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 5. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔮 6. Make predictions on the test set
y_pred = model.predict(X_test)

# 📊 7. Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# 📈 8. Visualize Actual vs Predicted prices
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted House Prices")
plt.plot([0, 50], [0, 50], '--r')
plt.show()

# 📋 9. Display feature coefficients
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coeff_df.sort_values(by="Coefficient", ascending=False))

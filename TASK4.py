import pandas as pd

# Load data
df = pd.read_csv('/content/sample_data/advertising.csv')

# Inspect first rows
print(df.head())

# Check info and missing data
print(df.info())
print(df.isnull().sum())

X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']                       # Target

from sklearn.model_selection import train_test_split

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))


import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Sales Prediction')
plt.show()

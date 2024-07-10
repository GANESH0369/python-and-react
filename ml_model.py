import numpy as np
import joblib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate a simple dataset for regression
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model to a .pkl file
joblib_file = "ml_model.pkl"
joblib.dump(model, joblib_file)

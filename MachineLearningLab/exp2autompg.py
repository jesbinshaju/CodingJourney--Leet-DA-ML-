import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# -------------------------------------------------
# Load Auto MPG Dataset
# -------------------------------------------------
df = pd.read_csv("auto-mpg.csv")

# -------------------------------------------------
# Data Preprocessing
# -------------------------------------------------
# Remove car name column
df.drop("car name", axis=1, inplace=True)
# Replace '?' in horsepower
df["horsepower"] = df["horsepower"].replace("?", np.nan)
# Remove missing rows
df.dropna(inplace=True)
# Convert horsepower to float
df["horsepower"] = df["horsepower"].astype(float)

# -------------------------------------------------
# Split Features and Target
# -------------------------------------------------
X = df.drop("mpg", axis=1).values
y = df["mpg"].values

# -------------------------------------------------
# Train Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Linear Regression Using Gradient Descent
# -------------------------------------------------
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=5000):

        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        # Initialize parameters
        self.w = np.zeros(n)
        self.b = 0

        self.losses = []

        for epoch in range(self.epochs):
          # Prediction
            y_pred = np.dot(X, self.w) + self.b
            # Error
            error = y_pred - y
            # MSE Loss
            loss = np.mean(error ** 2)
            self.losses.append(loss)
            # Gradients
            dw = (2/m) * np.dot(X.T, error)
            db = (2/m) * np.sum(error)
            # Update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# -------------------------------------------------
# Polynomial Feature Generation From Scratch
# Degree = 2
# -------------------------------------------------
def polynomial_features(X):
    m, n = X.shape
    X_poly = X.copy()
    # Add squared terms
    for i in range(n):
        X_poly = np.column_stack(
            (
                X_poly,
                X[:, i] ** 2
            )
        )
    return X_poly

# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------
def evaluate_model(name, y_true, y_pred, model):

    print("\n==============================")
    print(name)
    print("==============================")

    print("MAE  :", mean_absolute_error(
        y_true,
        y_pred
    ))
    print("MSE  :", mean_squared_error(
        y_true,
        y_pred
    ))
    print("RMSE :", np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    ))
    print("R2 Score :", r2_score(
        y_true,
        y_pred
    ))
    print("\nBias:")
    print(model.b)
    print("\nWeights:")
    print(model.w)

# -------------------------------------------------
# Linear Regression Training
# -------------------------------------------------

linear_model = LinearRegressionGD(
    learning_rate=0.01,
    epochs=5000
)
linear_model.fit(
    X_train,
    y_train
)
linear_prediction = linear_model.predict(
    X_test
)

evaluate_model(
    "Linear Regression",
    y_test,
    linear_prediction,
    linear_model
)

# -------------------------------------------------
# Polynomial Regression
# -------------------------------------------------
# Create polynomial features
X_train_poly = polynomial_features(
    X_train
)
X_test_poly = polynomial_features(
    X_test
)

poly_model = LinearRegressionGD(
    learning_rate=0.001,
    epochs=10000
)

poly_model.fit(
    X_train_poly,
    y_train
)

poly_prediction = poly_model.predict(
    X_test_poly
)
evaluate_model(
    "Polynomial Regression Degree 2",
    y_test,
    poly_prediction,
    poly_model
)

# -------------------------------------------------
# Model Comparison
# -------------------------------------------------
print("\n\n========== MODEL COMPARISON ==========")
linear_r2 = r2_score(
    y_test,
    linear_prediction
)
poly_r2 = r2_score(
    y_test,
    poly_prediction
)
if poly_r2 > linear_r2:
    print("Polynomial Regression performs better")
else:
    print("Linear Regression performs better")
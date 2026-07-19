import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("breast_cancer.csv")

# Remove ID column
df.drop("id", axis=1, inplace=True)

# Convert diagnosis
# M = 1
# B = 0

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

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
# Sigmoid Function
# -------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------------------------------------
# Logistic Regression using MLE
# -------------------------------------------------

class LogisticRegressionMLE:

    def __init__(self, learning_rate=0.01, epochs=5000):

        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):

        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        self.losses = []

        for epoch in range(self.epochs):

            # Linear Function
            z = np.dot(X, self.w) + self.b

            # Sigmoid
            p = sigmoid(z)

            # Negative Log Likelihood
            loss = -np.mean(
                y * np.log(p + 1e-10)
                + (1 - y) * np.log(1 - p + 1e-10)
            )

            self.losses.append(loss)

            # Gradient
            dw = (1 / m) * np.dot(X.T, (p - y))
            db = (1 / m) * np.sum(p - y)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X):

        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def predict(self, X):

        p = self.predict_probability(X)
        return (p >= 0.5).astype(int)

# -------------------------------------------------
# Logistic Regression using MAP
# -------------------------------------------------

class LogisticRegressionMAP:

    def __init__(self,
                 learning_rate=0.01,
                 epochs=5000,
                 lam=0.1):

        self.lr = learning_rate
        self.epochs = epochs
        self.lam = lam

    def fit(self, X, y):

        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        self.losses = []

        for epoch in range(self.epochs):

            z = np.dot(X, self.w) + self.b

            p = sigmoid(z)





            # MAP Loss
            loss = (
                -np.mean(
                    y * np.log(p + 1e-10)
                    + (1 - y) * np.log(1 - p + 1e-10)
                )
                + (self.lam / (2 * m)) * np.sum(self.w ** 2)
            )

            self.losses.append(loss)

            # Gradient
            dw = (1 / m) * np.dot(X.T, (p - y)) + (self.lam / m) * self.w
            db = (1 / m) * np.sum(p - y)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X):

        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def predict(self, X):

        p = self.predict_probability(X)
        return (p >= 0.5).astype(int)

# -------------------------------------------------
# Accuracy Function
# -------------------------------------------------

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# -------------------------------------------------
# Train MLE Model
# -------------------------------------------------

mle = LogisticRegressionMLE(
    learning_rate=0.01,
    epochs=5000
)

mle.fit(X_train, y_train)

pred1 = mle.predict(X_test)

print("\n==============================")
print("MLE Logistic Regression")
print("==============================")

print("\nAccuracy :", accuracy(y_test, pred1))

print("\nBias:")
print(mle.b)

print("\nWeights:")
print(mle.w)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred1))

print("\nClassification Report")
print(classification_report(y_test, pred1))

print("Precision :", precision_score(y_test, pred1))
print("Recall    :", recall_score(y_test, pred1))
print("F1 Score  :", f1_score(y_test, pred1))

# -------------------------------------------------
# Train MAP Model
# -------------------------------------------------

map_model = LogisticRegressionMAP(
    learning_rate=0.01,
    epochs=5000,
    lam=0.1
)

map_model.fit(X_train, y_train)

pred2 = map_model.predict(X_test)

print("\n==============================")
print("MAP Logistic Regression")
print("==============================")

print("\nAccuracy :", accuracy(y_test, pred2))

print("\nBias:")
print(map_model.b)

print("\nWeights:")
print(map_model.w)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred2))

print("\nClassification Report")
print(classification_report(y_test, pred2))

print("Precision :", precision_score(y_test, pred2))
print("Recall    :", recall_score(y_test, pred2))
print("F1 Score  :", f1_score(y_test, pred2))


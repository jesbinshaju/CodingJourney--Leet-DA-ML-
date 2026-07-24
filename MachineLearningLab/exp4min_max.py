import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("breast_cancer.csv")
df.drop("id", axis=1, inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sigmoid Function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Logistic Regression From Scratch
class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=5000):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        self.losses = []
        for epoch in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            p = sigmoid(z)
            loss = -np.mean(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
            self.losses.append(loss)
            dw = (1 / m) * np.dot(X.T, p - y)
            db = (1 / m) * np.sum(p - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X):
        return sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        return (self.predict_probability(X) >= 0.5).astype(int)

# Evaluation Function
def evaluate(name, X_train, X_test, y_train, y_test):
    model = LogisticRegressionGD(learning_rate=0.01, epochs=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Weights:", model.w)
    print("Bias:", model.b)
    return accuracy_score(y_test, y_pred)

# 1. Without Feature Scaling
accuracy_no_scaling = evaluate("Logistic Regression - Without Scaling", X_train, X_test, y_train, y_test)

# 2. Standardization
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)
accuracy_standard = evaluate("Logistic Regression - Standardization", X_train_standard, X_test_standard, y_train, y_test)

# 3. Min-Max Scaling
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)
accuracy_minmax = evaluate("Logistic Regression - Min-Max Scaling", X_train_minmax, X_test_minmax, y_train, y_test)

# Comparison
print("\n========== MODEL COMPARISON ==========")
print("Without Scaling       :", accuracy_no_scaling)
print("Standardization       :", accuracy_standard)
print("Min-Max Scaling       :", accuracy_minmax)

accuracies = [accuracy_no_scaling, accuracy_standard, accuracy_minmax]
names = ["Without Scaling", "Standardization", "Min-Max Scaling"]
print("\nBest Method:", names[np.argmax(accuracies)])
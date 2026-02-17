import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulated voting dataset
np.random.seed(42)
normal_votes = np.random.normal(loc=50, scale=5, size=(1000, 1))
anomalies = np.random.normal(loc=80, scale=1, size=(50, 1))

X = np.vstack((normal_votes, anomalies))
y = np.array([0]*1000 + [1]*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train)

y_pred = model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

print(classification_report(y_test, y_pred))

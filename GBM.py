import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv("insurance2.csv")


X = data.drop(["insuranceclaim", "charges"], axis=1)
y = data["insuranceclaim"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler_cols = ["age", "bmi", "children"]
scaler = StandardScaler()

X_train[scaler_cols] = scaler.fit_transform(X_train[scaler_cols])

X_test[scaler_cols] = scaler.transform(X_test[scaler_cols])


model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)


m_pred = model.predict(X_test)
print("Initial Test Accuracy:", accuracy_score(y_test, m_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, m_pred))


param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2], 
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)

print("\n--- Best Model Results ---")
print("Best Params:", grid_search.best_params_)

print("Final Test Accuracy:", round(accuracy_score(y_test, best_pred) * 100, 2), "%")
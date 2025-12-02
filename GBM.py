import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report


data = pd.read_csv("insurance2.csv")


scaler_col = ["age","bmi","children","charges"]

scaler = StandardScaler()
data[scaler_col] = scaler.fit_transform(data[scaler_col])

X = data.drop("insuranceclaim",axis=1)
y = data["insuranceclaim"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1)
model.fit(X_train,y_train)

m_pred = model.predict(X_test)


print("Confusion Matrix :- \n",confusion_matrix(y_test,m_pred))


param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [1, 3, 5]
}

grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)
best_score = cross_val_score(best_model, X_train, y_train, cv=5)
print("Tuned Model Accuracy :- ", round(np.mean(best_score)*100,2))
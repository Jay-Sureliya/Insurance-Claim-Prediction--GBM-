import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
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

print("Accuracy is :- ",round(accuracy_score(y_test,m_pred)*100,2),"%")
print("Confusion Matrix :- \n",confusion_matrix(y_test,m_pred))
print("Classification Report :- \n",classification_report(y_test,m_pred))
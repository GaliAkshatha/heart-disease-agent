import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib

df = pd.read_csv("data/combined.csv")

df.drop(['MaxHR', 'ExerciseAngina'], axis=1, inplace=True, errors='ignore')
df.dropna(inplace=True)

expected_features = ['Age','Sex','Height','Weight','RestingBP','DiastolicBP',
                     'Cholesterol','Glucose','Smoking','AlcoholIntake','PhysicalActivity']

df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

X = df.drop('HeartDisease', axis=1)
y = np.where(df['HeartDisease'] > 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

xgb = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train_resampled, y_train_resampled)
y_pred = xgb.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(xgb, "model/heart_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(list(X.columns), "model/feature_columns.pkl")

print("Model Saved Successfully!")

import shap
import joblib
import pandas as pd

model = joblib.load("../model/heart_model.pkl")
features = joblib.load("../model/feature_columns.pkl")

df = pd.read_csv("combined.csv")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[features])

shap.summary_plot(shap_values, df[features], feature_names=features)

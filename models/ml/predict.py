import joblib
import pandas as pd
import shap
import numpy as np

def load_all():
    model = joblib.load("../model/heart_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    features = joblib.load("../model/feature_columns.pkl")
    return model, scaler, features

def predict_patient(input_data, view="patient"):
    model, scaler, features = load_all()

    df = pd.DataFrame([input_data])
    df["BMI"] = df["Weight"] / ((df["Height"]/100)**2)
    df = df[features]

    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    # Patient explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    shap_df = pd.DataFrame({
        "Feature": features,
        "Value": df.iloc[0].values,
        "Impact": shap_values[0]
    }).sort_values(by="Impact", key=abs, ascending=False).head(3)

    if view == "doctor":
        return shap_df

    # Patient friendly text
    factors = []
    for _, row in shap_df.iterrows():
        direction = "increased" if row["Impact"] > 0 else "decreased"
        factors.append(f"{row['Feature']} ({round(row['Value'], 2)}) {direction} risk.")

    return {
        "prediction": int(prediction),
        "probability": round(probability * 100, 2),
        "explanation": factors
    }

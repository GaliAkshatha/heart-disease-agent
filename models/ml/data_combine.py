# ---------------------------------------------
# 🩺 HEART DISEASE DATASET COMBINATION SCRIPT (Final Fixed Version for Pandas 2.2+)
# ---------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1️⃣ LOAD YOUR DATASETS (update file paths if needed)
sulianova = pd.read_csv("cardio_train_sulianova.csv", sep=';')       # Sulianova dataset
uci = pd.read_csv("HeartDiseaseTrain-Test_ketangangal.csv", sep=';') # Ketan Gangal (UCI)
redwan = pd.read_csv("heart_disease_uci_redwankarimsony.csv", sep=';') # Redwankarimsony dataset
india = pd.read_csv("heart_attack_prediction_india_ankushpanday.csv", sep=';') # Ankush Panday (India)

# 2️⃣ CLEAN & RENAME COLUMNS TO A COMMON STRUCTURE
print("Sulianova Columns:", sulianova.columns.tolist())

## Sulianova Dataset
sulianova = sulianova.rename(columns={
    'age': 'Age',
    'gender': 'Sex',
    'height': 'Height',
    'weight': 'Weight',
    'ap_hi': 'RestingBP',
    'ap_lo': 'DiastolicBP',
    'cholesterol': 'Cholesterol',
    'gluc': 'Glucose',
    'smoke': 'Smoking',
    'alco': 'AlcoholIntake',
    'active': 'PhysicalActivity',
    'cardio': 'HeartDisease'
})

# Convert age from days → years
if 'Age' in sulianova.columns:
    sulianova['Age'] = (sulianova['Age'] / 365).round(1)

## UCI Dataset
uci = uci.rename(columns={
    'age': 'Age',
    'sex': 'Sex',
    'trestbps': 'RestingBP',
    'chol': 'Cholesterol',
    'thalach': 'MaxHR',
    'exang': 'ExerciseAngina',
    'target': 'HeartDisease'
})
uci['PhysicalActivity'] = None
uci['AlcoholIntake'] = None
uci['Smoking'] = None
uci['Glucose'] = None
uci['Height'] = None
uci['Weight'] = None
uci['DiastolicBP'] = None

## Redwankarimsony Dataset
redwan = redwan.rename(columns={
    'age': 'Age',
    'sex': 'Sex',
    'trestbps': 'RestingBP',
    'chol': 'Cholesterol',
    'thalach': 'MaxHR',
    'exang': 'ExerciseAngina',
    'target': 'HeartDisease'
})
redwan['PhysicalActivity'] = None
redwan['AlcoholIntake'] = None
redwan['Smoking'] = None
redwan['Glucose'] = None
redwan['Height'] = None
redwan['Weight'] = None
redwan['DiastolicBP'] = None

## India Dataset
india = india.rename(columns={
    'Age': 'Age',
    'Sex': 'Sex',
    'BP': 'RestingBP',
    'Cholesterol': 'Cholesterol',
    'Diabetes': 'Glucose',
    'Heart Attack Risk': 'HeartDisease'
})
india['PhysicalActivity'] = india.get('Physical Activity', None)
india['AlcoholIntake'] = None
india['Smoking'] = india.get('Smoking', None)
india['Height'] = None
india['Weight'] = None
india['DiastolicBP'] = None
india['MaxHR'] = None
india['ExerciseAngina'] = None

# 3️⃣ UNIFY COLUMN ORDER (handle missing ones safely)
common_columns = [
    'Age', 'Sex', 'Height', 'Weight', 'RestingBP', 'DiastolicBP',
    'Cholesterol', 'Glucose', 'Smoking', 'AlcoholIntake', 
    'PhysicalActivity', 'MaxHR', 'ExerciseAngina', 'HeartDisease'
]

def safe_select(df, common_columns):
    # keep only columns that exist in this dataset
    cols = [c for c in common_columns if c in df.columns]
    return df[cols]

sulianova = safe_select(sulianova, common_columns)
uci = safe_select(uci, common_columns)
redwan = safe_select(redwan, common_columns)
india = safe_select(india, common_columns)


# 4️⃣ COMBINE DATASETS
combined = pd.concat([sulianova, uci, redwan, india], ignore_index=True)

# -------------------------------------------
# 🧹 Data Cleaning and Preprocessing
# -------------------------------------------

# Replace any text placeholders with NaN
combined.replace(['Unknown', 'unknown', 'N/A', 'na', 'NA', '?', '', 'None'], pd.NA, inplace=True)

# Identify numeric columns
numeric_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Convert numeric-like columns to numeric (force errors to NaN)
combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Fill missing numeric values with median
for col in numeric_cols:
    median_value = combined[col].median()
    combined[col].fillna(median_value, inplace=True)

# Check if any column still contains strings (optional debug)
for col in combined.columns:
    if combined[col].dtype == 'object':
        print(f"⚠️ Column '{col}' still contains non-numeric values: {combined[col].unique()[:5]}")

# Standardize numeric data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
combined[numeric_cols] = scaler.fit_transform(combined[numeric_cols])

print("✅ Data successfully cleaned and scaled!")


# 5️⃣ HANDLE MISSING VALUES (fill with mean/mode safely)
for col in combined.columns:
    if combined[col].dtype == 'O':
        if not combined[col].mode().empty:
            mode_value = combined[col].mode().iloc[0]
            combined[col] = combined[col].fillna(mode_value)
        else:
            combined[col] = combined[col].fillna("Unknown")
    else:
        mean_value = combined[col].mean()
        combined[col] = combined[col].fillna(mean_value)

# 6️⃣ NORMALIZE NUMERICAL FEATURES
numeric_cols = [c for c in ['Age', 'Height', 'Weight', 'RestingBP', 'DiastolicBP', 'Cholesterol', 'Glucose', 'MaxHR']
                if c in combined.columns]
if numeric_cols:
    scaler = StandardScaler()
    combined[numeric_cols] = scaler.fit_transform(combined[numeric_cols])

# 7️⃣ CLEAN SEX COLUMN
if 'Sex' in combined.columns:
    combined['Sex'] = combined['Sex'].replace({'M': 1, 'F': 0, 'male': 1, 'female': 0})
    combined['Sex'] = combined['Sex'].fillna(0).astype(int)

# 8️⃣ FINAL OUTPUT
print("✅ Combined dataset shape:", combined.shape)
print(combined.head())

# 💾 SAVE FILE
combined.to_csv("combined_heart_disease_dataset.csv", index=False)
print("💾 Saved as combined_heart_disease_dataset.csv")

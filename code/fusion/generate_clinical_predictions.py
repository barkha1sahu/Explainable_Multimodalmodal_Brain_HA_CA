import os
import joblib
import pandas as pd

BASE = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"

DATA_PATH = os.path.join(
    BASE,
    "clinical_stage",
    "clinical_stage",
    "data",
    "clinical_cleaned.csv"
)

MODEL_PATH = os.path.join(
    BASE,
    "clinical_stage",
    "clinical_stage",
    "models",
    "clinical_rf_pipeline.joblib"
)

OUTPUT_PATH = os.path.join(
    BASE,
    "code",
    "fusion",
    "clinical_predictions_all_subjects.csv"
)

print("Loading clinical data...")
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)

# EXACT columns expected by the saved pipeline
FEATURES = [
    "M/F",
    "Age",
    "Educ",
    "SES",
    "MMSE",
    "eTIV",
    "nWBV",
    "ASF",
    "Delay"
]

X = df[FEATURES].copy()

print("\nMissing values:")
print(X.isna().sum())

print("\nLoading saved clinical model...")
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# IMPORTANT:
# Do NOT remove missing rows.
# The saved pipeline contains its own imputer.

print("\nGenerating predictions...")
probabilities = model.predict_proba(X)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

# CDR-based binary target
true_labels = (df["CDR"] > 0).astype(int)

output = pd.DataFrame({
    "Subject_ID": df["ID"].astype(str),
    "True_Label": true_labels,
    "Clinical_Probability": probabilities,
    "Clinical_Prediction": predictions
})

output.to_csv(OUTPUT_PATH, index=False)

print("\n====================================")
print("SUCCESS")
print("====================================")
print("Subjects:", len(output))
print("Saved to:")
print(OUTPUT_PATH)

print("\nFirst 10 rows:")
print(output.head(10))

print("\nClinical prediction distribution:")
print(output["Clinical_Prediction"].value_counts())

print("\nDone.")
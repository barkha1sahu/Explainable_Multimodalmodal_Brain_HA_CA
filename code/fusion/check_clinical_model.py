import os
import joblib

BASE = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"

MODEL_PATH = os.path.join(
    BASE,
    "clinical_stage",
    "clinical_stage",
    "models",
    "clinical_rf_pipeline.joblib"
)

model = joblib.load(MODEL_PATH)

print("MODEL TYPE:")
print(type(model))

print("\nMODEL STEPS:")
if hasattr(model, "named_steps"):
    for name, step in model.named_steps.items():
        print(name, "->", type(step))

print("\nMODEL DETAILS:")
print(model)

print("\nDone.")
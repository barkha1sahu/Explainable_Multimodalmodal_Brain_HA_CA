import os
import pandas as pd

BASE = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"

MRI_FILE = os.path.join(
    BASE,
    "OASIS_CNN_Dataset_1_12",
    "model_results",
    "finetuned_subject_results_optimized_threshold.csv"
)

CLINICAL_FILE = os.path.join(
    BASE,
    "code",
    "fusion",
    "clinical_predictions_all_subjects.csv"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "code",
    "fusion",
    "common_subjects.csv"
)

print("Loading MRI results...")
mri = pd.read_csv(MRI_FILE)

print("MRI columns:")
print(mri.columns.tolist())

print("\nLoading clinical results...")
clinical = pd.read_csv(CLINICAL_FILE)

print("Clinical columns:")
print(clinical.columns.tolist())

# Make IDs strings
mri["Subject_ID"] = mri["Subject_ID"].astype(str)
clinical["Subject_ID"] = clinical["Subject_ID"].astype(str)

# Remove accidental duplicates
mri = mri.drop_duplicates(subset="Subject_ID")
clinical = clinical.drop_duplicates(subset="Subject_ID")

print("\nMRI subjects:", len(mri))
print("Clinical subjects:", len(clinical))

# Find common subjects
common = pd.merge(
    mri,
    clinical,
    on="Subject_ID",
    how="inner",
    suffixes=("_MRI", "_Clinical")
)

print("\n===================================")
print("SUBJECT OVERLAP")
print("===================================")

print("Common subjects:", len(common))
print("MRI-only subjects:", len(set(mri["Subject_ID"]) - set(clinical["Subject_ID"])))
print("Clinical-only subjects:", len(set(clinical["Subject_ID"]) - set(mri["Subject_ID"])))

# Check label agreement
if "True_Label_MRI" in common.columns and "True_Label_Clinical" in common.columns:

    common["Label_Match"] = (
        common["True_Label_MRI"] ==
        common["True_Label_Clinical"]
    )

    print("\nLabel mismatches:", (~common["Label_Match"]).sum())

common.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nFirst 10 common subjects:")
print(common.head(10))

print("\nDone.")
import pandas as pd
import os

# =========================================================
# 1. PATHS
# =========================================================

METADATA_FILE = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
    r"\image_metadata_cleaned.csv"
)

# IMPORTANT:
# Yahan apni Excel file ka exact path daalna hai
EXCEL_FILE = r"C:\Users\rog\Downloads\oasis_cross-sectional.xlsx"

OUTPUT_FILE = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
    r"\image_metadata_labelled.csv"
)

# =========================================================
# 2. CHECK FILES
# =========================================================

if not os.path.exists(METADATA_FILE):
    print("ERROR: Metadata file nahi mili:")
    print(METADATA_FILE)
    raise SystemExit

if not os.path.exists(EXCEL_FILE):
    print("ERROR: Excel file nahi mili:")
    print(EXCEL_FILE)
    print("\nEXCEL_FILE mein apni Excel file ka exact path daalo.")
    raise SystemExit

# =========================================================
# 3. LOAD FILES
# =========================================================

print("Loading MRI metadata...")
mri_df = pd.read_csv(METADATA_FILE)

print("Loading Excel...")
excel_df = pd.read_excel(EXCEL_FILE)

print("\nMRI metadata rows:", len(mri_df))
print("Excel rows:", len(excel_df))

# =========================================================
# 4. CLEAN COLUMN NAMES
# =========================================================

mri_df.columns = mri_df.columns.str.strip()
excel_df.columns = excel_df.columns.str.strip()

print("\nMRI columns:")
print(list(mri_df.columns))

print("\nExcel columns:")
print(list(excel_df.columns))

# =========================================================
# 5. CHECK REQUIRED COLUMNS
# =========================================================

if "Subject_ID" not in mri_df.columns:
    print("\nERROR: MRI metadata mein Subject_ID column nahi mila.")
    raise SystemExit

if "ID" not in excel_df.columns:
    print("\nERROR: Excel mein ID column nahi mila.")
    raise SystemExit

if "CDR" not in excel_df.columns:
    print("\nERROR: Excel mein CDR column nahi mila.")
    raise SystemExit

# =========================================================
# 6. CLEAN IDs
# =========================================================

mri_df["Subject_ID"] = (
    mri_df["Subject_ID"]
    .astype(str)
    .str.strip()
)

excel_df["ID"] = (
    excel_df["ID"]
    .astype(str)
    .str.strip()
)

# =========================================================
# 7. CHECK DUPLICATE IDs IN EXCEL
# =========================================================

duplicate_ids = excel_df[excel_df["ID"].duplicated(keep=False)]

print("\nDuplicate IDs in Excel:", len(duplicate_ids))

if len(duplicate_ids) > 0:
    print("\nDuplicate IDs:")
    print(duplicate_ids[["ID", "CDR"]].to_string(index=False))

# =========================================================
# 8. KEEP REQUIRED CLINICAL COLUMNS
# =========================================================

clinical_columns = [
    "ID",
    "M/F",
    "Age",
    "Educ",
    "SES",
    "MMSE",
    "CDR",
    "eTIV",
    "nWBV",
    "ASF",
    "Delay"
]

available_columns = [
    col for col in clinical_columns
    if col in excel_df.columns
]

clinical_df = excel_df[available_columns].copy()

# =========================================================
# 9. MERGE MRI + CLINICAL DATA
# =========================================================

print("\nMerging MRI metadata with Excel...")

labelled_df = mri_df.merge(
    clinical_df,
    left_on="Subject_ID",
    right_on="ID",
    how="left"
)

# =========================================================
# 10. CREATE BINARY LABEL
# =========================================================

def create_label(cdr):
    if pd.isna(cdr):
        return pd.NA

    if cdr == 0:
        return 0

    elif cdr > 0:
        return 1

    return pd.NA


labelled_df["Label"] = labelled_df["CDR"].apply(create_label)

# =========================================================
# 11. MATCHING REPORT
# =========================================================

matched = labelled_df["CDR"].notna().sum()
unmatched = labelled_df["CDR"].isna().sum()

print("\n==============================")
print("MATCHING REPORT")
print("==============================")

print("Total MRI images:", len(labelled_df))
print("Matched images:", matched)
print("Unmatched images:", unmatched)

# =========================================================
# 12. SUBJECT-LEVEL REPORT
# =========================================================

total_subjects = labelled_df["Subject_ID"].nunique()

matched_subjects = labelled_df.loc[
    labelled_df["CDR"].notna(),
    "Subject_ID"
].nunique()

unmatched_subjects = labelled_df.loc[
    labelled_df["CDR"].isna(),
    "Subject_ID"
].nunique()

print("\nTotal MRI subjects:", total_subjects)
print("Matched subjects:", matched_subjects)
print("Unmatched subjects:", unmatched_subjects)

# =========================================================
# 13. CDR DISTRIBUTION
# =========================================================

print("\n==============================")
print("CDR DISTRIBUTION")
print("==============================")

print(labelled_df["CDR"].value_counts(dropna=False).sort_index())

# =========================================================
# 14. LABEL DISTRIBUTION
# =========================================================

print("\n==============================")
print("BINARY LABEL DISTRIBUTION")
print("==============================")

print(labelled_df["Label"].value_counts(dropna=False).sort_index())

# =========================================================
# 15. SAVE UNMATCHED SUBJECTS
# =========================================================

unmatched_df = labelled_df[
    labelled_df["CDR"].isna()
][["Subject_ID"]].drop_duplicates()

UNMATCHED_FILE = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
    r"\unmatched_subjects.csv"
)

unmatched_df.to_csv(
    UNMATCHED_FILE,
    index=False
)

# =========================================================
# 16. SAVE LABELLED DATASET
# =========================================================

labelled_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================================================
# 17. FINAL OUTPUT
# =========================================================

print("\n==============================")
print("DONE")
print("==============================")

print("Labelled metadata saved at:")
print(OUTPUT_FILE)

print("\nUnmatched subjects saved at:")
print(UNMATCHED_FILE)

print("\nFinal columns:")
print(list(labelled_df.columns))

print("\nFirst 10 rows:")
print(labelled_df.head(10).to_string(index=False))
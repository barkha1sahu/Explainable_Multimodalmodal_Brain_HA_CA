import pandas as pd
import os
from sklearn.model_selection import train_test_split

# =========================================================
# PATH
# =========================================================

INPUT_FILE = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
    r"\image_metadata_labelled.csv"
)

OUTPUT_DIR = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
)

# =========================================================
# LOAD DATA
# =========================================================

print("Loading labelled dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total images:", len(df))
print("Total subjects:", df["Subject_ID"].nunique())

# =========================================================
# REMOVE SUBJECTS WITH MISSING LABEL
# =========================================================

print("\nRemoving subjects with missing labels...")

df_labeled = df.dropna(subset=["Label"]).copy()

print("Images after removing missing labels:", len(df_labeled))
print("Subjects after removing missing labels:",
      df_labeled["Subject_ID"].nunique())

# =========================================================
# CREATE SUBJECT-LEVEL TABLE
# =========================================================

subject_df = (
    df_labeled[
        ["Subject_ID", "Label"]
    ]
    .drop_duplicates()
    .copy()
)

print("\nSubject-level label distribution:")
print(subject_df["Label"].value_counts())

# =========================================================
# CHECK ONE LABEL PER SUBJECT
# =========================================================

label_check = (
    subject_df
    .groupby("Subject_ID")["Label"]
    .nunique()
)

if (label_check > 1).any():
    print("\nERROR: Some subjects have multiple labels!")
    print(label_check[label_check > 1])
    raise SystemExit

print("\nLabel consistency check: PASSED")

# =========================================================
# TRAIN / TEMP SPLIT
# 70% TRAIN
# 30% TEMP
# =========================================================

train_subjects, temp_subjects = train_test_split(
    subject_df,
    test_size=0.30,
    random_state=42,
    stratify=subject_df["Label"]
)

# =========================================================
# VALIDATION / TEST SPLIT
# 15% VALIDATION
# 15% TEST
# =========================================================

val_subjects, test_subjects = train_test_split(
    temp_subjects,
    test_size=0.50,
    random_state=42,
    stratify=temp_subjects["Label"]
)

# =========================================================
# GET SUBJECT IDS
# =========================================================

train_ids = set(train_subjects["Subject_ID"])
val_ids = set(val_subjects["Subject_ID"])
test_ids = set(test_subjects["Subject_ID"])

# =========================================================
# CREATE IMAGE-LEVEL DATASETS
# =========================================================

train_df = df_labeled[
    df_labeled["Subject_ID"].isin(train_ids)
].copy()

val_df = df_labeled[
    df_labeled["Subject_ID"].isin(val_ids)
].copy()

test_df = df_labeled[
    df_labeled["Subject_ID"].isin(test_ids)
].copy()

# =========================================================
# SAVE CSV FILES
# =========================================================

train_file = os.path.join(
    OUTPUT_DIR,
    "train_metadata.csv"
)

val_file = os.path.join(
    OUTPUT_DIR,
    "validation_metadata.csv"
)

test_file = os.path.join(
    OUTPUT_DIR,
    "test_metadata.csv"
)

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

# =========================================================
# REPORT
# =========================================================

print("\n===================================")
print("SPLIT COMPLETE")
print("===================================")

print("\nSUBJECT COUNT")
print("----------------")
print("Train subjects:", len(train_ids))
print("Validation subjects:", len(val_ids))
print("Test subjects:", len(test_ids))

print("\nIMAGE COUNT")
print("----------------")
print("Train images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

print("\nTRAIN LABEL DISTRIBUTION")
print("----------------")
print(train_df["Label"].value_counts().sort_index())

print("\nVALIDATION LABEL DISTRIBUTION")
print("----------------")
print(val_df["Label"].value_counts().sort_index())

print("\nTEST LABEL DISTRIBUTION")
print("----------------")
print(test_df["Label"].value_counts().sort_index())

# =========================================================
# CHECK SUBJECT LEAKAGE
# =========================================================

train_val_overlap = train_ids.intersection(val_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = val_ids.intersection(test_ids)

print("\n===================================")
print("LEAKAGE CHECK")
print("===================================")

print("Train-Val overlap:", len(train_val_overlap))
print("Train-Test overlap:", len(train_test_overlap))
print("Val-Test overlap:", len(val_test_overlap))

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\nNO SUBJECT LEAKAGE - PASSED")
else:
    print("\nWARNING: SUBJECT LEAKAGE FOUND!")

print("\nFiles saved:")
print(train_file)
print(val_file)
print(test_file)
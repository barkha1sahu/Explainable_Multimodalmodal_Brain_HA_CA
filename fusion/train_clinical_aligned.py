import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"

CLINICAL_FILE = os.path.join(
    BASE_DIR,
    "clinical_stage",
    "clinical_stage",
    "data",
    "clinical_cleaned.csv"
)

MRI_DIR = os.path.join(
    BASE_DIR,
    "OASIS_CNN_Dataset_1_12"
)

TRAIN_MRI = os.path.join(MRI_DIR, "train_metadata.csv")
VAL_MRI = os.path.join(MRI_DIR, "validation_metadata.csv")
TEST_MRI = os.path.join(MRI_DIR, "test_metadata.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "code", "fusion")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CLINICAL DATA
# ============================================================

print("\nLoading clinical data...")

clinical = pd.read_csv(CLINICAL_FILE)

print("Clinical shape:", clinical.shape)

# Subject ID
clinical["Subject_ID"] = clinical["ID"].astype(str)

# Binary target
clinical["True_Label"] = (
    clinical["CDR"] > 0
).astype(int)

print("Clinical subjects:", len(clinical))


# ============================================================
# LOAD MRI SUBJECT SPLITS
# ============================================================

print("\nLoading MRI subject splits...")

train_mri = pd.read_csv(TRAIN_MRI)
val_mri = pd.read_csv(VAL_MRI)
test_mri = pd.read_csv(TEST_MRI)

train_subjects = set(train_mri["Subject_ID"].unique())
val_subjects = set(val_mri["Subject_ID"].unique())
test_subjects = set(test_mri["Subject_ID"].unique())

print("MRI train subjects:", len(train_subjects))
print("MRI validation subjects:", len(val_subjects))
print("MRI test subjects:", len(test_subjects))

# Check that there is no overlap
assert train_subjects.isdisjoint(val_subjects)
assert train_subjects.isdisjoint(test_subjects)
assert val_subjects.isdisjoint(test_subjects)

print("No subject overlap between MRI splits.")


# ============================================================
# CHECK CLINICAL ↔ MRI SUBJECT MATCH
# ============================================================

clinical_subjects = set(clinical["Subject_ID"])

all_mri_subjects = (
    train_subjects |
    val_subjects |
    test_subjects
)

print("\nClinical subjects:", len(clinical_subjects))
print("MRI labelled subjects:", len(all_mri_subjects))

missing_from_clinical = all_mri_subjects - clinical_subjects
missing_from_mri = clinical_subjects - all_mri_subjects

print("MRI subjects missing from clinical:", len(missing_from_clinical))
print("Clinical subjects missing from MRI:", len(missing_from_mri))

if missing_from_clinical:
    print("Missing MRI subjects:", sorted(missing_from_clinical))

if missing_from_mri:
    print("Missing clinical subjects:", sorted(missing_from_mri))


# ============================================================
# CREATE ALIGNED CLINICAL SPLITS
# ============================================================

train_df = clinical[
    clinical["Subject_ID"].isin(train_subjects)
].copy()

val_df = clinical[
    clinical["Subject_ID"].isin(val_subjects)
].copy()

test_df = clinical[
    clinical["Subject_ID"].isin(test_subjects)
].copy()

print("\nAligned clinical split sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# VERIFY SPLITS
# ============================================================

assert len(train_df) == len(train_subjects)
assert len(val_df) == len(val_subjects)
assert len(test_df) == len(test_subjects)

assert set(train_df["Subject_ID"]).isdisjoint(
    val_df["Subject_ID"]
)

assert set(train_df["Subject_ID"]).isdisjoint(
    test_df["Subject_ID"]
)

assert set(val_df["Subject_ID"]).isdisjoint(
    test_df["Subject_ID"]
)

print("Clinical split verification passed.")


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "Age",
    "Educ",
    "SES",
    "eTIV",
    "nWBV",
    "ASF"
]

CATEGORICAL_FEATURES = [
    "M/F"
]

MMSE_FEATURE = [
    "MMSE"
]


# ============================================================
# PREPROCESSOR FUNCTION
# ============================================================

def create_preprocessor(include_mmse):

    numeric_features = NUMERIC_FEATURES.copy()

    if include_mmse:
        numeric_features += MMSE_FEATURE

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_clinical_model(include_mmse, model_name):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    features = NUMERIC_FEATURES.copy()

    if include_mmse:
        features += MMSE_FEATURE

    features += CATEGORICAL_FEATURES

    X_train = train_df[features]
    y_train = train_df["True_Label"]

    X_val = val_df[features]
    y_val = val_df["True_Label"]

    X_test = test_df[features]
    y_test = test_df["True_Label"]

    # Same RF methodology as your original clinical_analysis.py
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor(include_mmse)
        ),
        (
            "classifier",
            rf
        )
    ])

    print("Features:", features)
    print("Training...")

    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_prob = pipeline.predict_proba(X_train)[:, 1]
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    train_pred = (train_prob >= 0.5).astype(int)
    val_pred = (val_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def calculate_metrics(y_true, prob, pred):

        return {
            "Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(
                y_true, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_true, pred, zero_division=0
            ),
            "F1": f1_score(
                y_true, pred, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(
                y_true, prob
            )
        }

    train_metrics = calculate_metrics(
        y_train, train_prob, train_pred
    )

    val_metrics = calculate_metrics(
        y_val, val_prob, val_pred
    )

    test_metrics = calculate_metrics(
        y_test, test_prob, test_pred
    )

    print("\nTRAIN:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nVALIDATION:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nTEST:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = os.path.join(
        OUTPUT_DIR,
        f"clinical_rf_{model_name}_aligned.joblib"
    )

    joblib.dump(
        pipeline,
        model_file
    )

    print("\nModel saved:")
    print(model_file)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    train_predictions = pd.DataFrame({
        "Subject_ID": train_df["Subject_ID"],
        "True_Label": y_train,
        "Clinical_Probability": train_prob,
        "Clinical_Prediction": train_pred,
        "Split": "train"
    })

    val_predictions = pd.DataFrame({
        "Subject_ID": val_df["Subject_ID"],
        "True_Label": y_val,
        "Clinical_Probability": val_prob,
        "Clinical_Prediction": val_pred,
        "Split": "validation"
    })

    test_predictions = pd.DataFrame({
        "Subject_ID": test_df["Subject_ID"],
        "True_Label": y_test,
        "Clinical_Probability": test_prob,
        "Clinical_Prediction": test_pred,
        "Split": "test"
    })

    predictions = pd.concat([
        train_predictions,
        val_predictions,
        test_predictions
    ], ignore_index=True)

    prediction_file = os.path.join(
        OUTPUT_DIR,
        f"clinical_predictions_{model_name}_aligned.csv"
    )

    predictions.to_csv(
        prediction_file,
        index=False
    )

    print("Predictions saved:")
    print(prediction_file)

    # --------------------------------------------------------
    # Metrics dataframe
    # --------------------------------------------------------

    metrics_rows = []

    for split_name, metrics in [
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics)
    ]:

        row = {
            "Model": model_name,
            "Split": split_name
        }

        row.update(metrics)

        metrics_rows.append(row)

    return metrics_rows


# ============================================================
# TRAIN BOTH VERSIONS
# ============================================================

all_metrics = []

# WITHOUT MMSE
metrics_without_mmse = train_clinical_model(
    include_mmse=False,
    model_name="without_MMSE"
)

all_metrics.extend(metrics_without_mmse)


# WITH MMSE
metrics_with_mmse = train_clinical_model(
    include_mmse=True,
    model_name="with_MMSE"
)

all_metrics.extend(metrics_with_mmse)


# ============================================================
# SAVE ALL METRICS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "clinical_aligned_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(metrics_df.to_string(index=False))

print("\nMetrics saved:")
print(metrics_file)

print("\nDONE.")
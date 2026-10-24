"""
==========================================================================
OASIS-1 CROSS-SECTIONAL DATASET
STAGE 1: CLINICAL / COGNITIVE DATA ANALYSIS AND PREDICTION
==========================================================================
This script performs a complete, reproducible analysis of the OASIS-1
clinical/cognitive table and trains baseline models to predict
CDR-based cognitive status (NOT a direct "Alzheimer's diagnosis").

Run from the `clinical_stage` folder with:
    python clinical_analysis.py

Expected folder layout (created automatically if missing):
    clinical_stage/
        data/            -> input + cleaned data
        clinical_eda/    -> EDA plots
        models/          -> trained pipeline (.joblib)
        outputs/         -> metrics, tables, confusion matrices
==========================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, classification_report
)
import joblib

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
RANDOM_STATE = 42

# --------------------------------------------------------------------
# Paths (Windows-compatible relative paths, run script from clinical_stage/)
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EDA_DIR = os.path.join(BASE_DIR, "clinical_eda")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
for d in [DATA_DIR, EDA_DIR, MODEL_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "oasis_cross-sectional.xlsx")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================================
# STEP 1: LOAD AND INSPECT THE DATA
# ==========================================================================
section("STEP 1: LOAD AND INSPECT THE DATA")

df_raw = pd.read_excel(INPUT_FILE)   # original file is never modified
df = df_raw.copy()                   # all further work happens on a copy

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nColumn names:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nDescriptive statistics (numeric):\n", df.describe())
print("\nCategorical unique values:")
for c in ["M/F", "Hand"]:
    print(f"  {c}: {df[c].unique().tolist()}")
print("\nCDR distribution (including missing):\n",
      df["CDR"].value_counts(dropna=False).sort_index())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate IDs:", df["ID"].duplicated().sum())

with open(os.path.join(OUT_DIR, "step1_inspection_summary.txt"), "w") as f:
    f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")
    f.write("Data types:\n" + df.dtypes.to_string() + "\n\n")
    f.write("Descriptive statistics:\n" + df.describe().to_string() + "\n\n")
    f.write("Missing values:\n" + df.isnull().sum().to_string() + "\n\n")
    f.write(f"Duplicate rows: {df.duplicated().sum()}\n")
    f.write(f"Duplicate IDs: {df['ID'].duplicated().sum()}\n\n")
    f.write("CDR distribution:\n" + df["CDR"].value_counts(dropna=False).sort_index().to_string())

# ==========================================================================
# STEP 2: VARIABLE ROLES (documented here, used for feature selection below)
# ==========================================================================
# ID     - subject identifier                -> not a feature
# M/F    - demographic (sex)                  -> candidate input
# Hand   - demographic (handedness)            -> constant column, see Step 3
# Age    - demographic                         -> candidate input
# Educ   - demographic (years of education)    -> candidate input
# SES    - demographic (socio-economic status) -> candidate input
# MMSE   - cognitive test score                -> candidate input (leakage caution)
# CDR    - clinical rating of dementia severity -> TARGET
# eTIV   - MRI-derived structural volume        -> candidate input
# nWBV   - MRI-derived structural volume        -> candidate input
# ASF    - MRI-derived scaling factor           -> candidate input
# Delay  - scan/exam timing gap                 -> mostly missing, see Step 3

# ==========================================================================
# STEP 3: DATA QUALITY CHECK
# ==========================================================================
section("STEP 3: DATA QUALITY CHECK")

# Handedness is constant ('R' for all subjects) -> carries no information
print("Hand unique values:", df["Hand"].unique().tolist(), "-> constant column, will be dropped")

# Delay is missing for ~95% of rows -> not usable as a feature
delay_missing_pct = df["Delay"].isnull().mean() * 100
print(f"Delay missing: {delay_missing_pct:.1f}% -> too sparse to use as a model feature")

# Invalid value checks
print("\nAge range:", df["Age"].min(), "-", df["Age"].max())
print("MMSE range:", df["MMSE"].min(), "-", df["MMSE"].max(), "(valid MMSE range is 0-30)")
print("CDR unique (non-null):", sorted(df["CDR"].dropna().unique().tolist()))
print("nWBV range:", df["nWBV"].min(), "-", df["nWBV"].max(), "(expected proportion 0-1)")

# CDR missingness pattern: is it random, or tied to age?
cdr_present_age = df.loc[df["CDR"].notna(), "Age"]
cdr_missing_age = df.loc[df["CDR"].isna(), "Age"]
print(f"\nAge range where CDR IS recorded:  {cdr_present_age.min()}-{cdr_present_age.max()}")
print(f"Age range where CDR is MISSING:   {cdr_missing_age.min()}-{cdr_missing_age.max()}")
print("-> CDR is missing only for the younger subjects (<=58). This means these")
print("   subjects were simply never given a clinical dementia assessment; it is")
print("   NOT random missingness, so we cannot impute CDR - those rows must be")
print("   excluded from supervised modeling (Step 4).")

df = df.drop(columns=["Hand"])  # constant, no information

with open(os.path.join(OUT_DIR, "step3_data_quality.txt"), "w") as f:
    f.write("Hand: constant ('R' only) -> dropped\n")
    f.write(f"Delay: {delay_missing_pct:.1f}% missing -> dropped as a feature\n")
    f.write(f"CDR missing rows correspond to subjects aged {cdr_missing_age.min()}-{cdr_missing_age.max()} "
            f"(never clinically assessed) -> excluded from modeling, not imputed\n")
    f.write(f"Duplicate rows: {df.duplicated().sum()}, Duplicate IDs: {df['ID'].duplicated().sum()}\n")

# ==========================================================================
# STEP 4: DEFINE THE TARGET
# ==========================================================================
section("STEP 4: DEFINE THE TARGET (CDR-based cognitive status)")

cdr_counts = df["CDR"].value_counts(dropna=False).sort_index()
print("Raw CDR distribution (including missing / not clinically assessed):")
print(cdr_counts)

# Keep only rows where a clinical CDR rating actually exists
df_model = df[df["CDR"].notna()].copy()
print(f"\nUsable rows for supervised modeling (CDR present): {len(df_model)}")
print(df_model["CDR"].value_counts().sort_index())

# The CDR=2.0 class has only 2 subjects. A stratified 80/20 split, and any
# meaningful multiclass model evaluation, is not reliable with that few
# samples in a class. We therefore use a BINARY formulation:
#   CDR = 0.0            -> "Nondemented" (0)
#   CDR = 0.5, 1.0, 2.0   -> "Cognitively impaired / Demented spectrum" (1)
# This mirrors the common OASIS convention and is explicitly reported here
# as a transformation, not an arbitrary hidden step.
df_model["cognitive_status"] = (df_model["CDR"] > 0).astype(int)
print("\nBinary target distribution (0 = Nondemented, 1 = CDR>0 spectrum):")
print(df_model["cognitive_status"].value_counts())
print("Class balance: {:.1f}% / {:.1f}%".format(
    *(df_model["cognitive_status"].value_counts(normalize=True) * 100)))

cdr_counts.to_csv(os.path.join(OUT_DIR, "step4_cdr_raw_distribution.csv"))
df_model["cognitive_status"].value_counts().to_csv(
    os.path.join(OUT_DIR, "step4_binary_target_distribution.csv"))

# ==========================================================================
# STEP 5: FEATURE SELECTION
# ==========================================================================
section("STEP 5: FEATURE SELECTION")

# Candidate inputs: Age, M/F, Educ, SES, MMSE, eTIV, nWBV, ASF
# Excluded:
#   ID     - identifier, not predictive
#   CDR    - this IS the target, must never be a feature
#   Delay  - >95% missing, dropped in Step 3
#   Hand   - constant, dropped in Step 3
#
# MMSE leakage note: MMSE (Mini-Mental State Examination) is a cognitive
# screening test and CDR (Clinical Dementia Rating) is a separate clinical
# rating, but both are collected as part of the same diagnostic workup and
# are strongly correlated by construction (impaired cognition drives both
# scores down). Using MMSE as an input is standard in OASIS-based studies,
# but it can make the prediction look unrealistically strong because MMSE
# is close to "a cheaper version of the label." To be transparent about
# this, we train and report models BOTH WITH and WITHOUT MMSE.

NUMERIC_FEATURES = ["Age", "Educ", "SES", "eTIV", "nWBV", "ASF"]
CATEGORICAL_FEATURES = ["M/F"]
MMSE_FEATURE = ["MMSE"]

print("Base numeric features:", NUMERIC_FEATURES)
print("Categorical features:", CATEGORICAL_FEATURES)
print("Optional (leakage-risk) feature:", MMSE_FEATURE)
print("Excluded: ID (identifier), CDR (target), Delay (95%+ missing), Hand (constant)")

# ==========================================================================
# STEP 6: EXPLORATORY DATA ANALYSIS
# ==========================================================================
section("STEP 6: EXPLORATORY DATA ANALYSIS")

# 1. CDR class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="CDR", data=df_model, order=sorted(df_model["CDR"].unique()))
plt.title("CDR Class Distribution (subjects with clinical rating)")
plt.xlabel("CDR score")
plt.ylabel("Number of subjects")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "01_cdr_distribution.png"), dpi=150)
plt.close()

# 2. Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(df_model["Age"], bins=15, kde=True)
plt.title("Age Distribution (clinically assessed subjects)")
plt.xlabel("Age (years)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "02_age_distribution.png"), dpi=150)
plt.close()

# 3. MMSE distribution
plt.figure(figsize=(6, 4))
sns.histplot(df_model["MMSE"], bins=15, kde=True)
plt.title("MMSE Score Distribution")
plt.xlabel("MMSE score (0-30)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "03_mmse_distribution.png"), dpi=150)
plt.close()

# 4. CDR vs MMSE
plt.figure(figsize=(6, 4))
sns.boxplot(x="CDR", y="MMSE", data=df_model, order=sorted(df_model["CDR"].unique()))
plt.title("MMSE Score by CDR Category")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "04_cdr_vs_mmse.png"), dpi=150)
plt.close()

# 5. CDR vs Age
plt.figure(figsize=(6, 4))
sns.boxplot(x="CDR", y="Age", data=df_model, order=sorted(df_model["CDR"].unique()))
plt.title("Age by CDR Category")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "05_cdr_vs_age.png"), dpi=150)
plt.close()

# 6. Correlation heatmap
corr_cols = ["Age", "Educ", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
plt.figure(figsize=(8, 6))
sns.heatmap(df_model[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (numerical clinical variables)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "06_correlation_heatmap.png"), dpi=150)
plt.close()

print("6 EDA plots saved to:", EDA_DIR)

# ==========================================================================
# STEP 7: STATISTICAL ANALYSIS
# ==========================================================================
section("STEP 7: STATISTICAL ANALYSIS")

stat_rows = []

# Age vs MMSE (both continuous, check normality -> use Spearman to be safe)
r, p = stats.spearmanr(df_model["Age"], df_model["MMSE"])
stat_rows.append(["Age", "MMSE", "Spearman correlation", round(r, 3), round(p, 4),
                   "Weak/monotonic relationship" if abs(r) < 0.3 else "Moderate relationship"])

# Age vs CDR (CDR is ordinal) -> Spearman
r, p = stats.spearmanr(df_model["Age"], df_model["CDR"])
stat_rows.append(["Age", "CDR", "Spearman correlation", round(r, 3), round(p, 4),
                   "Weak/monotonic relationship" if abs(r) < 0.3 else "Moderate relationship"])

# MMSE vs CDR (both ordinal/scores) -> Spearman
r, p = stats.spearmanr(df_model["MMSE"], df_model["CDR"])
stat_rows.append(["MMSE", "CDR", "Spearman correlation", round(r, 3), round(p, 4),
                   "Strong negative relationship" if abs(r) > 0.5 else "Moderate relationship"])

# nWBV vs CDR (continuous vs ordinal) -> Spearman
r, p = stats.spearmanr(df_model["nWBV"], df_model["CDR"])
stat_rows.append(["nWBV", "CDR", "Spearman correlation", round(r, 3), round(p, 4),
                   "Moderate negative relationship" if abs(r) > 0.3 else "Weak relationship"])

stat_table = pd.DataFrame(stat_rows, columns=["Variable 1", "Variable 2", "Test used",
                                               "Statistic (rho)", "p-value", "Interpretation"])
print(stat_table.to_string(index=False))
print("\nNote: Spearman correlation is used because CDR is an ordinal variable")
print("(not a continuous measurement), so a rank-based correlation is more")
print("appropriate than Pearson. All results describe ASSOCIATION, not causation.")

stat_table.to_csv(os.path.join(OUT_DIR, "step7_statistical_analysis.csv"), index=False)

# ==========================================================================
# STEP 8 & 9: PREPROCESSING PIPELINE + TRAIN/TEST SPLIT
# ==========================================================================
section("STEP 8-9: PREPROCESSING PIPELINE AND TRAIN/TEST SPLIT")

def build_pipeline(numeric_features, categorical_features, model):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

# Two feature sets, as decided in Step 5
FEATURE_SETS = {
    "with_MMSE": NUMERIC_FEATURES + MMSE_FEATURE,
    "without_MMSE": NUMERIC_FEATURES,
}

y = df_model["cognitive_status"]

splits = {}
for name, feats in FEATURE_SETS.items():
    X = df_model[feats + CATEGORICAL_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    splits[name] = (X_train, X_test, y_train, y_test, feats)
    print(f"[{name}] Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows "
          f"(stratified 80/20, random_state={RANDOM_STATE})")

print("\nPreprocessing note: the imputer and scaler/encoder are fit ONLY on the")
print("training split inside the pipeline, then applied unchanged to the test")
print("split. This prevents information leakage from test to train.")

# ==========================================================================
# STEP 10 & 11: BASELINE MODELS + EVALUATION
# ==========================================================================
section("STEP 10-11: BASELINE MODELS AND EVALUATION")

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}

for fset_name, (X_train, X_test, y_train, y_test, feats) in splits.items():
    for model_name, model in MODELS.items():
        # clone() gives each pipeline its own fresh, unfitted copy of the
        # estimator so that fitting one pipeline never overwrites another
        pipe = build_pipeline(feats, CATEGORICAL_FEATURES, clone(model))
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Feature set": fset_name,
            "Model": model_name,
            "Accuracy": round(acc, 3),
            "Precision (weighted)": round(prec, 3),
            "Recall (weighted)": round(rec, 3),
            "F1 (weighted)": round(f1, 3),
            "ROC-AUC": round(auc, 3),
        })

        key = f"{fset_name}__{model_name.replace(' ', '_')}"
        fitted_pipelines[key] = pipe

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=["Nondemented", "CDR>0"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name} ({fset_name})")
        plt.tight_layout()
        fname = f"confusion_matrix_{fset_name}_{model_name.replace(' ', '_')}.png"
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close()

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUT_DIR, "step11_model_comparison.csv"), index=False)

# ==========================================================================
# STEP 12: FEATURE IMPORTANCE (Random Forest, with_MMSE feature set)
# ==========================================================================
section("STEP 12: FEATURE IMPORTANCE (Random Forest)")

best_key = "with_MMSE__Random_Forest"
best_pipe = fitted_pipelines[best_key]
ohe_names = best_pipe.named_steps["preprocessor"].named_transformers_["cat"] \
    .named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
feature_names = FEATURE_SETS["with_MMSE"] + list(ohe_names)
importances = best_pipe.named_steps["classifier"].feature_importances_

fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}) \
    .sort_values("Importance", ascending=False)
print(fi_df.to_string(index=False))

plt.figure(figsize=(7, 5))
sns.barplot(x="Importance", y="Feature", data=fi_df, color="steelblue")
plt.title("Random Forest Feature Importance (with_MMSE feature set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance_random_forest.png"), dpi=150)
plt.close()

fi_df.to_csv(os.path.join(OUT_DIR, "step12_feature_importance.csv"), index=False)
print("\nNote: feature importance reflects predictive contribution within this")
print("model only. It does NOT establish that a feature causes cognitive decline.")

# ==========================================================================
# STEP 13: SAVE OUTPUTS
# ==========================================================================
section("STEP 13: SAVE OUTPUTS")

# Save cleaned dataset (never overwrites the original Excel file)
df_model.to_csv(os.path.join(DATA_DIR, "clinical_cleaned.csv"), index=False)
print("Cleaned dataset saved to:", os.path.join(DATA_DIR, "clinical_cleaned.csv"))

# Save the best-performing pipeline (with_MMSE Random Forest, as an example)
joblib.dump(best_pipe, os.path.join(MODEL_DIR, "clinical_rf_pipeline.joblib"))
print("Trained pipeline saved to:",
      os.path.join(MODEL_DIR, "clinical_rf_pipeline.joblib"))

# ==========================================================================
# STEP 14: FINAL REPORT SUMMARY
# ==========================================================================
section("STEP 14: FINAL REPORT SUMMARY")

summary_lines = [
    f"Total rows in original Excel file: {df_raw.shape[0]}",
    f"Rows usable for supervised modeling (CDR present): {len(df_model)}",
    f"Rows excluded (CDR not clinically assessed, age <= {int(cdr_missing_age.max())}): "
    f"{df_raw.shape[0] - len(df_model)}",
    "",
    "Missing values (original file): " + ", ".join(
        f"{c}={v}" for c, v in df_raw.isnull().sum().items() if v > 0),
    "",
    "Target: CDR, binarized as 0 (Nondemented) vs 1 (CDR>0, cognitively impaired spectrum)",
    f"Target distribution: {df_model['cognitive_status'].value_counts().to_dict()}",
    "",
    f"Features used (with MMSE): {FEATURE_SETS['with_MMSE'] + CATEGORICAL_FEATURES}",
    f"Features used (without MMSE, leakage-sensitivity check): "
    f"{FEATURE_SETS['without_MMSE'] + CATEGORICAL_FEATURES}",
    "Excluded: ID, Hand (constant), Delay (>95% missing), CDR (target itself)",
    "",
    "Preprocessing: median imputation + standard scaling for numeric features; "
    "most-frequent imputation + one-hot encoding for M/F; fit on training data only.",
    "",
    "Models trained: Logistic Regression, Random Forest (baseline, minimal tuning)",
    "",
    "Evaluation metrics (see outputs/step11_model_comparison.csv for exact numbers):",
]
print("\n".join(summary_lines))
print(results_df.to_string(index=False))

with open(os.path.join(OUT_DIR, "step14_final_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines) + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\nLimitations:\n")
    f.write("- Sample size is small (235 clinically-assessed subjects) for machine learning.\n")
    f.write("- CDR=2 class has only 2 subjects; binary formulation was used instead of multiclass.\n")
    f.write("- MMSE is highly related to CDR by construction; models with/without MMSE are reported "
            "to make this transparent.\n")
    f.write("- Subjects without a CDR rating are structurally younger; the model does not generalize "
            "to that unassessed population.\n")
    f.write("- This model predicts CDR-based cognitive status, not a clinical Alzheimer's diagnosis.\n\n")
    f.write("Multimodal connection (future work, NOT implemented here):\n")
    f.write("MRI -> CNN/Transfer Learning -> MRI feature vector\n")
    f.write("Clinical data -> Clinical model -> Clinical feature vector\n")
    f.write("[MRI feature vector + Clinical feature vector] -> Feature Fusion -> Final Multimodal Model\n")

print("\nSTAGE 1 COMPLETE")


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# PATHS
# Script location: HA_DATASET/code/fusion_with_MMSE.py
# --------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
FUSION_DIR = CODE_DIR / "fusion"

MRI_VALIDATION = FUSION_DIR / "mri_validation_subject_predictions.csv"
CLINICAL_NO_MMSE = FUSION_DIR / "clinical_predictions_without_MMSE_aligned.csv"
CLINICAL_WITH_MMSE = FUSION_DIR / "clinical_predictions_with_MMSE_aligned.csv"
EXISTING_TEST = FUSION_DIR / "constrained_multimodal_test_results.csv"

OUTPUT_VALIDATION = FUSION_DIR / "mmse_comparison_validation.csv"
OUTPUT_METRICS = FUSION_DIR / "mmse_comparison_metrics.csv"


def read_csv_checked(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return pd.read_csv(path)


def calculate_metrics(y_true, probability, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)

    results = {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "Confusion_Matrix": str(confusion_matrix(y_true, prediction).tolist()),
    }

    results["ROC_AUC"] = (
        roc_auc_score(y_true, probability)
        if len(np.unique(y_true)) == 2
        else np.nan
    )
    return results


def main():
    print("=" * 60)
    print("OASIS MRI + CLINICAL FUSION COMPARISON")
    print("=" * 60)
    print("Project folder:", BASE_DIR)
    print("Fusion folder:", FUSION_DIR)

    mri = read_csv_checked(MRI_VALIDATION)
    clinical_without = read_csv_checked(CLINICAL_NO_MMSE)
    clinical_with = read_csv_checked(CLINICAL_WITH_MMSE)

    required_mri = {"Subject_ID", "True_Label", "Ensemble_Probability"}
    required_clinical = {
        "Subject_ID", "True_Label", "Clinical_Probability", "Split"
    }

    for name, df, required in [
        ("MRI validation", mri, required_mri),
        ("Clinical without MMSE", clinical_without, required_clinical),
        ("Clinical with MMSE", clinical_with, required_clinical),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    # Only validation subjects are used for comparison.
    no_mmse_val = clinical_without[
        clinical_without["Split"].str.lower() == "validation"
    ].copy()

    with_mmse_val = clinical_with[
        clinical_with["Split"].str.lower() == "validation"
    ].copy()

    mri["Subject_ID"] = mri["Subject_ID"].astype(str).str.strip()
    no_mmse_val["Subject_ID"] = no_mmse_val["Subject_ID"].astype(str).str.strip()
    with_mmse_val["Subject_ID"] = with_mmse_val["Subject_ID"].astype(str).str.strip()

    def merge_branch(clinical, branch_name):
        merged = mri.merge(
            clinical[
                ["Subject_ID", "True_Label", "Clinical_Probability"]
            ],
            on="Subject_ID",
            how="inner",
            suffixes=("_MRI", "_Clinical"),
            validate="one_to_one",
        )

        if merged.empty:
            raise ValueError(
                f"No matched validation subjects for {branch_name}. "
                "Check subject IDs and split labels."
            )

        if not (
            merged["True_Label_MRI"].astype(int)
            == merged["True_Label_Clinical"].astype(int)
        ).all():
            raise ValueError(f"Labels disagree in {branch_name}.")

        merged["True_Label"] = merged["True_Label_MRI"].astype(int)
        merged["MRI_Probability"] = merged["Ensemble_Probability"].astype(float)
        merged["Clinical_Probability"] = merged[
            "Clinical_Probability"
        ].astype(float)
        merged["Branch"] = branch_name
        return merged

    branches = [
        merge_branch(no_mmse_val, "Clinical_without_MMSE"),
        merge_branch(with_mmse_val, "Clinical_with_MMSE"),
    ]

    all_results = []
    all_metrics = []

    # Keep MRI contribution >= 40%; tune on validation only.
    for df in branches:
        best = None

        for mri_weight in np.round(np.arange(0.40, 0.91, 0.05), 2):
            clinical_weight = round(1.0 - mri_weight, 2)

            probability = (
                mri_weight * df["MRI_Probability"].to_numpy()
                + clinical_weight * df["Clinical_Probability"].to_numpy()
            )

            for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
                metrics = calculate_metrics(
                    df["True_Label"], probability, float(threshold)
                )

                # Maximize F1; use accuracy as tie-breaker.
                score = (metrics["F1"], metrics["Accuracy"])

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "weight": float(mri_weight),
                        "clinical_weight": clinical_weight,
                        "threshold": float(threshold),
                        "probability": probability.copy(),
                        "metrics": metrics,
                    }

        result_df = df[
            ["Subject_ID", "True_Label", "MRI_Probability",
             "Clinical_Probability", "Branch"]
        ].copy()

        result_df["MRI_Weight"] = best["weight"]
        result_df["Clinical_Weight"] = best["clinical_weight"]
        result_df["Threshold"] = best["threshold"]
        result_df["Fusion_Probability"] = best["probability"]
        result_df["Fusion_Prediction"] = (
            result_df["Fusion_Probability"] >= best["threshold"]
        ).astype(int)

        all_results.append(result_df)

        row = {
            "Branch": df["Branch"].iloc[0],
            "MRI_Weight": best["weight"],
            "Clinical_Weight": best["clinical_weight"],
            **best["metrics"],
            "Validation_Subjects": len(df),
        }
        all_metrics.append(row)

        print(f"\n{row['Branch']}")
        print(f"Matched validation subjects: {len(df)}")
        print(f"MRI weight: {best['weight']:.2f}")
        print(f"Clinical weight: {best['clinical_weight']:.2f}")
        print(f"Threshold: {best['threshold']:.2f}")
        print(f"Validation accuracy: {row['Accuracy']:.4f}")
        print(f"Validation precision: {row['Precision']:.4f}")
        print(f"Validation recall: {row['Recall']:.4f}")
        print(f"Validation F1: {row['F1']:.4f}")
        print(f"Validation ROC-AUC: {row['ROC_AUC']:.4f}")
        print("Confusion matrix:", row["Confusion_Matrix"])

    pd.concat(all_results, ignore_index=True).to_csv(
        OUTPUT_VALIDATION, index=False
    )
    pd.DataFrame(all_metrics).to_csv(OUTPUT_METRICS, index=False)

    print("\nSaved validation comparison:")
    print(OUTPUT_VALIDATION)
    print(OUTPUT_METRICS)

    # Report the existing held-out test result; do not retune on test data.
    if EXISTING_TEST.exists():
        test = pd.read_csv(EXISTING_TEST)
        print("\nExisting held-out test result:")
        print("Subjects:", len(test))
        print("Accuracy:", accuracy_score(
            test["True_Label"].astype(int),
            test["Fusion_Prediction"].astype(int),
        ))
        print("This is the previously saved test result, not a new MMSE test.")
    else:
        print("\nExisting test CSV not found:", EXISTING_TEST)

    print("\nDONE")


if __name__ == "__main__":
    main()

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# =========================================================
# PATHS
# =========================================================

RESULTS_DIR = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
    r"\model_results"
)

INDIVIDUAL_FILE = os.path.join(
    RESULTS_DIR,
    "individual_model_results.csv"
)

ENSEMBLE_FILE = os.path.join(
    RESULTS_DIR,
    "ensemble_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "test_predictions_all_models.csv"
)

SUBJECT_FILE = os.path.join(
    RESULTS_DIR,
    "subject_level_ensemble_results.csv"
)

# =========================================================
# 1. LOAD RESULTS
# =========================================================

individual = pd.read_csv(INDIVIDUAL_FILE)
ensemble = pd.read_csv(ENSEMBLE_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)
subject = pd.read_csv(SUBJECT_FILE)

print("\n===================================")
print("INDIVIDUAL MODEL RESULTS")
print("===================================")

print(individual.to_string(index=False))

print("\n===================================")
print("ENSEMBLE RESULTS")
print("===================================")

print(ensemble.to_string(index=False))

# =========================================================
# 2. MODEL COMPARISON GRAPH
# =========================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

individual_plot = individual.set_index("Model")[metrics]

individual_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Individual CNN Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

comparison_file = os.path.join(
    RESULTS_DIR,
    "model_comparison.png"
)

plt.savefig(comparison_file, dpi=300)
plt.show()

# =========================================================
# 3. CONFUSION MATRIX - ENSEMBLE
# =========================================================

cm = confusion_matrix(
    predictions["Label"],
    predictions["Ensemble_Prediction"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Class 0", "Class 1"]
)

disp.plot()

plt.title("Ensemble Confusion Matrix")
plt.tight_layout()

cm_file = os.path.join(
    RESULTS_DIR,
    "ensemble_confusion_matrix.png"
)

plt.savefig(cm_file, dpi=300)
plt.show()

# =========================================================
# 4. ROC CURVES - ALL 4 MODELS + ENSEMBLE
# =========================================================

y_true = predictions["Label"]

model_probability_columns = {
    "ResNet50": "ResNet50_Probability",
    "DenseNet121": "DenseNet121_Probability",
    "EfficientNetB0": "EfficientNetB0_Probability",
    "EfficientNetV2B0": "EfficientNetV2B0_Probability",
    "Ensemble": "Ensemble_Probability"
}

plt.figure(figsize=(10, 7))

for model_name, column in model_probability_columns.items():

    fpr, tpr, _ = roc_curve(
        y_true,
        predictions[column]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - CNN Models and Ensemble")
plt.legend()
plt.tight_layout()

roc_file = os.path.join(
    RESULTS_DIR,
    "roc_curves.png"
)

plt.savefig(
    roc_file,
    dpi=300
)

plt.show()

# =========================================================
# 5. SUBJECT-LEVEL CONFUSION MATRIX
# =========================================================

subject_cm = confusion_matrix(
    subject["True_Label"],
    subject["Ensemble_Prediction"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=subject_cm,
    display_labels=["Class 0", "Class 1"]
)

disp.plot()

plt.title("Subject-Level Ensemble Confusion Matrix")
plt.tight_layout()

subject_cm_file = os.path.join(
    RESULTS_DIR,
    "subject_level_confusion_matrix.png"
)

plt.savefig(
    subject_cm_file,
    dpi=300
)

plt.show()

# =========================================================
# 6. FINAL SUMMARY
# =========================================================

print("\n===================================")
print("ANALYSIS COMPLETE")
print("===================================")

print("\nGenerated files:")

print("\n1. Model comparison:")
print(comparison_file)

print("\n2. Ensemble confusion matrix:")
print(cm_file)

print("\n3. ROC curves:")
print(roc_file)

print("\n4. Subject-level confusion matrix:")
print(subject_cm_file)
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================================================
# PATHS
# =========================================================

DATASET_ROOT = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
)

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "images"
)

VALIDATION_FILE = os.path.join(
    DATASET_ROOT,
    "validation_metadata.csv"
)

TEST_FILE = os.path.join(
    DATASET_ROOT,
    "test_metadata.csv"
)

RESULTS_DIR = os.path.join(
    DATASET_ROOT,
    "model_results"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

MODEL_FILES = {
    "ResNet50":
        "ResNet50_finetuned_best.keras",

    "DenseNet121":
        "DenseNet121_finetuned_best.keras",

    "EfficientNetB0":
        "EfficientNetB0_finetuned_best.keras",

    "EfficientNetV2B0":
        "EfficientNetV2B0_finetuned_best.keras"
}


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# =========================================================
# PREDICT
# =========================================================

def predict_model(model_name, model_file, df):

    model_path = os.path.join(
        RESULTS_DIR,
        model_file
    )

    print("\nLoading:", model_name)

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    probabilities = []

    for start in range(
        0,
        len(df),
        BATCH_SIZE
    ):

        batch_df = df.iloc[
            start:start + BATCH_SIZE
        ]

        batch_images = []

        for path in batch_df["Image_Path"]:

            image = load_image(path)

            batch_images.append(image)

        batch_images = tf.stack(
            batch_images
        )

        batch_probabilities = model.predict(
            batch_images,
            verbose=0
        ).flatten()

        probabilities.extend(
            batch_probabilities
        )

    return np.array(
        probabilities
    )


# =========================================================
# FUNCTION: GET SUBJECT-LEVEL PREDICTIONS
# =========================================================

def get_subject_predictions(df):

    df = df.copy()

    df["Label"] = df["Label"].astype(int)

    df["Image_Path"] = df["Image_Name"].apply(
        lambda x: os.path.join(
            IMAGE_DIR,
            str(x)
        )
    )

    # Check images
    missing = [
        path
        for path in df["Image_Path"]
        if not os.path.exists(path)
    ]

    if len(missing) > 0:

        print(
            "First missing image:",
            missing[0]
        )

        raise FileNotFoundError(
            "Missing image found."
        )

    prediction_df = df[
        [
            "Subject_ID",
            "Image_Name",
            "Label"
        ]
    ].copy()

    # Predict using all four models

    for model_name, model_file in MODEL_FILES.items():

        probabilities = predict_model(
            model_name,
            model_file,
            df
        )

        prediction_df[
            model_name + "_Probability"
        ] = probabilities

    # Average four models at image level

    model_columns = [
        "ResNet50_Probability",
        "DenseNet121_Probability",
        "EfficientNetB0_Probability",
        "EfficientNetV2B0_Probability"
    ]

    prediction_df[
        "Ensemble_Probability"
    ] = prediction_df[
        model_columns
    ].mean(axis=1)

    # Convert image predictions to subject predictions

    subject_df = prediction_df.groupby(
        "Subject_ID"
    ).agg(
        True_Label=("Label", "first"),

        ResNet50_Probability=(
            "ResNet50_Probability",
            "mean"
        ),

        DenseNet121_Probability=(
            "DenseNet121_Probability",
            "mean"
        ),

        EfficientNetB0_Probability=(
            "EfficientNetB0_Probability",
            "mean"
        ),

        EfficientNetV2B0_Probability=(
            "EfficientNetV2B0_Probability",
            "mean"
        ),

        Ensemble_Probability=(
            "Ensemble_Probability",
            "mean"
        )
    ).reset_index()

    return subject_df


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

print("\n===================================")
print("LOADING VALIDATION DATA")
print("===================================")

validation_df = pd.read_csv(
    VALIDATION_FILE
)

validation_subject_df = get_subject_predictions(
    validation_df
)

print(
    "\nValidation subjects:",
    len(validation_subject_df)
)


# =========================================================
# THRESHOLD SEARCH
# =========================================================

y_true_val = validation_subject_df[
    "True_Label"
].values

y_prob_val = validation_subject_df[
    "Ensemble_Probability"
].values


thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

threshold_results = []


for threshold in thresholds:

    y_pred = (
        y_prob_val >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true_val,
        y_pred
    )

    precision = precision_score(
        y_true_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true_val,
        y_pred,
        zero_division=0
    )

    threshold_results.append({

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1
    })


threshold_df = pd.DataFrame(
    threshold_results
)


# =========================================================
# SELECT BEST THRESHOLD
# =========================================================
# We select threshold using validation F1.
# TEST DATA IS NOT USED HERE.

best_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(
    best_row["Threshold"]
)


print("\n===================================")
print("BEST VALIDATION THRESHOLD")
print("===================================")

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_row['Accuracy']:.4f}"
)

print(
    f"Precision : {best_row['Precision']:.4f}"
)

print(
    f"Recall    : {best_row['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_row['F1']:.4f}"
)


# =========================================================
# VALIDATION ROC-AUC
# =========================================================

validation_auc = roc_auc_score(
    y_true_val,
    y_prob_val
)

print(
    f"ROC-AUC   : {validation_auc:.4f}"
)


# =========================================================
# SAVE THRESHOLD RESULTS
# =========================================================

threshold_file = os.path.join(
    RESULTS_DIR,
    "validation_threshold_results.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)

print(
    "\nThreshold results saved:"
)

print(threshold_file)


# =========================================================
# LOAD TEST DATA
# =========================================================

print("\n===================================")
print("LOADING TEST DATA")
print("===================================")

test_df = pd.read_csv(
    TEST_FILE
)

test_subject_df = get_subject_predictions(
    test_df
)

print(
    "\nTest subjects:",
    len(test_subject_df)
)


# =========================================================
# APPLY VALIDATION THRESHOLD TO TEST
# =========================================================

y_true_test = test_subject_df[
    "True_Label"
].values

y_prob_test = test_subject_df[
    "Ensemble_Probability"
].values

y_pred_test = (
    y_prob_test >= best_threshold
).astype(int)


# =========================================================
# TEST METRICS
# =========================================================

test_accuracy = accuracy_score(
    y_true_test,
    y_pred_test
)

test_precision = precision_score(
    y_true_test,
    y_pred_test,
    zero_division=0
)

test_recall = recall_score(
    y_true_test,
    y_pred_test,
    zero_division=0
)

test_f1 = f1_score(
    y_true_test,
    y_pred_test,
    zero_division=0
)

test_auc = roc_auc_score(
    y_true_test,
    y_prob_test
)

test_cm = confusion_matrix(
    y_true_test,
    y_pred_test
)


# =========================================================
# PRINT FINAL TEST RESULTS
# =========================================================

print("\n===================================")
print("FINAL TEST RESULTS")
print("===================================")

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_auc:.4f}"
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(test_cm)


# =========================================================
# SAVE TEST SUBJECT RESULTS
# =========================================================

test_subject_df[
    "Prediction"
] = y_pred_test


test_subject_file = os.path.join(
    RESULTS_DIR,
    "finetuned_subject_results_optimized_threshold.csv"
)

test_subject_df.to_csv(
    test_subject_file,
    index=False
)


# =========================================================
# SAVE FINAL METRICS
# =========================================================

final_results = pd.DataFrame([{

    "Level": "Subject",

    "Threshold": best_threshold,

    "Accuracy": test_accuracy,

    "Precision": test_precision,

    "Recall": test_recall,

    "F1": test_f1,

    "ROC_AUC": test_auc
}])


final_metrics_file = os.path.join(
    RESULTS_DIR,
    "finetuned_subject_metrics_optimized_threshold.csv"
)

final_results.to_csv(
    final_metrics_file,
    index=False
)


print("\n===================================")
print("COMPLETE")
print("===================================")

print(
    "\nTest subject results:"
)

print(test_subject_file)

print(
    "\nFinal metrics:"
)

print(final_metrics_file)
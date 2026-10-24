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

TEST_FILE = os.path.join(
    DATASET_ROOT,
    "test_metadata.csv"
)

RESULTS_DIR = os.path.join(
    DATASET_ROOT,
    "model_results"
)


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# =========================================================
# MODELS
# =========================================================

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
# LOAD TEST METADATA
# =========================================================

print("\nLoading test metadata...")

test_df = pd.read_csv(
    TEST_FILE
)

test_df["Label"] = test_df["Label"].astype(int)

test_df["Image_Path"] = test_df[
    "Image_Name"
].apply(
    lambda x: os.path.join(
        IMAGE_DIR,
        str(x)
    )
)


print(
    "\nTest images:",
    len(test_df)
)

print(
    "Test subjects:",
    test_df["Subject_ID"].nunique()
)


# =========================================================
# CHECK FILES
# =========================================================

missing = [
    path
    for path in test_df["Image_Path"]
    if not os.path.exists(path)
]

print(
    "Missing images:",
    len(missing)
)

if len(missing) > 0:

    print(
        "First missing:",
        missing[0]
    )

    raise FileNotFoundError(
        "Missing test image."
    )


# =========================================================
# IMAGE LOADING
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
# PREDICT WITH ONE MODEL
# =========================================================

def predict_model(
    model_name,
    model_file
):

    print("\n===================================")
    print("Loading:", model_name)
    print("===================================")

    model_path = os.path.join(
        RESULTS_DIR,
        model_file
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    probabilities = []

    # Process images in batches

    for start in range(
        0,
        len(test_df),
        BATCH_SIZE
    ):

        batch_df = test_df.iloc[
            start:start + BATCH_SIZE
        ]

        batch_images = []

        for path in batch_df[
            "Image_Path"
        ]:

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

    probabilities = np.array(
        probabilities
    )

    return probabilities


# =========================================================
# PREDICT ALL MODELS
# =========================================================

prediction_df = test_df[
    [
        "Subject_ID",
        "Image_Name",
        "Label"
    ]
].copy()


for model_name, model_file in MODEL_FILES.items():

    probabilities = predict_model(
        model_name,
        model_file
    )

    prediction_df[
        model_name + "_Probability"
    ] = probabilities


# =========================================================
# ENSEMBLE
# =========================================================

model_probability_columns = [

    "ResNet50_Probability",

    "DenseNet121_Probability",

    "EfficientNetB0_Probability",

    "EfficientNetV2B0_Probability"
]


prediction_df[
    "Ensemble_Probability"
] = prediction_df[
    model_probability_columns
].mean(
    axis=1
)


# =========================================================
# IMAGE-LEVEL PREDICTION
# =========================================================

prediction_df[
    "Ensemble_Prediction"
] = (
    prediction_df[
        "Ensemble_Probability"
    ] >= 0.5
).astype(int)


# =========================================================
# SUBJECT-LEVEL AGGREGATION
# =========================================================

subject_df = prediction_df.groupby(
    "Subject_ID"
).agg(

    True_Label=(
        "Label",
        "first"
    ),

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


# =========================================================
# SUBJECT PREDICTION
# =========================================================

subject_df[
    "Ensemble_Prediction"
] = (
    subject_df[
        "Ensemble_Probability"
    ] >= 0.5
).astype(int)


# =========================================================
# SUBJECT METRICS
# =========================================================

y_true = subject_df[
    "True_Label"
]

y_pred = subject_df[
    "Ensemble_Prediction"
]

y_prob = subject_df[
    "Ensemble_Probability"
]


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n===================================")
print("SUBJECT-LEVEL RESULTS")
print("===================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)


print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

prediction_file = os.path.join(
    RESULTS_DIR,
    "finetuned_test_predictions_all_models.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)


subject_file = os.path.join(
    RESULTS_DIR,
    "finetuned_subject_level_results.csv"
)

subject_df.to_csv(
    subject_file,
    index=False
)


# =========================================================
# SAVE METRICS
# =========================================================

results = pd.DataFrame(
    [
        {
            "Level": "Subject",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": auc
        }
    ]
)


results_file = os.path.join(
    RESULTS_DIR,
    "finetuned_subject_level_metrics.csv"
)

results.to_csv(
    results_file,
    index=False
)


# =========================================================
# COMPLETE
# =========================================================

print("\n===================================")
print("SUBJECT-LEVEL EVALUATION COMPLETE")
print("===================================")

print("\nPrediction file:")
print(prediction_file)

print("\nSubject-level file:")
print(subject_file)

print("\nMetrics file:")
print(results_file)
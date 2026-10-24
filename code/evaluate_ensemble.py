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

RESULTS_DIR = os.path.join(
    DATASET_ROOT,
    "model_results"
)

TEST_FILE = os.path.join(
    DATASET_ROOT,
    "test_metadata.csv"
)

# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

MODELS = [
    "ResNet50",
    "DenseNet121",
    "EfficientNetB0",
    "EfficientNetV2B0"
]

# =========================================================
# LOAD TEST DATA
# =========================================================

print("Loading test metadata...")

test_df = pd.read_csv(TEST_FILE)

test_df["Label"] = test_df["Label"].astype(int)

test_df["Image_Path"] = test_df["Image_Name"].apply(
    lambda x: os.path.join(
        IMAGE_DIR,
        str(x)
    )
)

print("Test images:", len(test_df))
print("Test subjects:", test_df["Subject_ID"].nunique())

# =========================================================
# LOAD IMAGES
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


def create_image_dataset(df):

    paths = df["Image_Path"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        paths
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


test_ds = create_image_dataset(test_df)

# =========================================================
# METRIC FUNCTION
# =========================================================

def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    }


# =========================================================
# PREDICT WITH EACH MODEL
# =========================================================

all_predictions = {}

individual_results = []

y_true = test_df["Label"].values

print("\n===================================")
print("INDIVIDUAL MODEL EVALUATION")
print("===================================")

for model_name in MODELS:

    model_path = os.path.join(
        RESULTS_DIR,
        model_name + "_best.keras"
    )

    print("\nLoading:", model_name)

    if not os.path.exists(model_path):

        print("ERROR: Model not found:")
        print(model_path)
        continue

    model = tf.keras.models.load_model(
        model_path
    )

    print("Generating predictions...")

    probabilities = model.predict(
        test_ds,
        verbose=1
    ).flatten()

    all_predictions[model_name] = probabilities

    metrics = calculate_metrics(
        y_true,
        probabilities
    )

    metrics["Model"] = model_name

    individual_results.append(
        metrics
    )

    print("\n", model_name)

    print(
        "Accuracy:",
        round(metrics["Accuracy"], 4)
    )

    print(
        "Precision:",
        round(metrics["Precision"], 4)
    )

    print(
        "Recall:",
        round(metrics["Recall"], 4)
    )

    print(
        "F1:",
        round(metrics["F1"], 4)
    )

    print(
        "ROC-AUC:",
        round(metrics["ROC_AUC"], 4)
    )

    del model


# =========================================================
# INDIVIDUAL RESULTS TABLE
# =========================================================

individual_results_df = pd.DataFrame(
    individual_results
)

individual_results_df = individual_results_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]
]

print("\n===================================")
print("INDIVIDUAL RESULTS")
print("===================================")

print(
    individual_results_df.to_string(
        index=False
    )
)

individual_file = os.path.join(
    RESULTS_DIR,
    "individual_model_results.csv"
)

individual_results_df.to_csv(
    individual_file,
    index=False
)

# =========================================================
# CREATE PREDICTION DATAFRAME
# =========================================================

prediction_df = test_df[
    [
        "Subject_ID",
        "Image_Name",
        "Label"
    ]
].copy()

for model_name in MODELS:

    prediction_df[
        model_name + "_Probability"
    ] = all_predictions[model_name]

# =========================================================
# SOFT VOTING / PROBABILITY AVERAGING
# =========================================================

prediction_columns = [
    model + "_Probability"
    for model in MODELS
]

prediction_df[
    "Ensemble_Probability"
] = prediction_df[
    prediction_columns
].mean(axis=1)

prediction_df[
    "Ensemble_Prediction"
] = (
    prediction_df["Ensemble_Probability"] >= 0.5
).astype(int)

# =========================================================
# IMAGE-LEVEL ENSEMBLE
# =========================================================

ensemble_metrics = calculate_metrics(
    prediction_df["Label"].values,
    prediction_df["Ensemble_Probability"].values
)

print("\n===================================")
print("IMAGE-LEVEL ENSEMBLE")
print("===================================")

for key, value in ensemble_metrics.items():

    print(
        key + ":",
        round(value, 4)
    )

# =========================================================
# SUBJECT-LEVEL AGGREGATION
# =========================================================

print("\n===================================")
print("SUBJECT-LEVEL ENSEMBLE")
print("===================================")

subject_df = (
    prediction_df
    .groupby("Subject_ID")
    .agg(
        True_Label=("Label", "first"),
        Ensemble_Probability=(
            "Ensemble_Probability",
            "mean"
        )
    )
    .reset_index()
)

subject_df[
    "Ensemble_Prediction"
] = (
    subject_df["Ensemble_Probability"] >= 0.5
).astype(int)

# =========================================================
# SUBJECT-LEVEL METRICS
# =========================================================

subject_metrics = calculate_metrics(
    subject_df["True_Label"].values,
    subject_df["Ensemble_Probability"].values
)

print("\nSubjects:", len(subject_df))

for key, value in subject_metrics.items():

    print(
        key + ":",
        round(value, 4)
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    subject_df["True_Label"],
    subject_df["Ensemble_Prediction"]
)

print("\nSubject-level confusion matrix:")
print(cm)

# =========================================================
# SAVE FILES
# =========================================================

prediction_file = os.path.join(
    RESULTS_DIR,
    "test_predictions_all_models.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

subject_file = os.path.join(
    RESULTS_DIR,
    "subject_level_ensemble_results.csv"
)

subject_df.to_csv(
    subject_file,
    index=False
)

ensemble_results_df = pd.DataFrame(
    [
        {
            "Level": "Image",
            **ensemble_metrics
        },
        {
            "Level": "Subject",
            **subject_metrics
        }
    ]
)

ensemble_file = os.path.join(
    RESULTS_DIR,
    "ensemble_results.csv"
)

ensemble_results_df.to_csv(
    ensemble_file,
    index=False
)

# =========================================================
# FINAL
# =========================================================

print("\n===================================")
print("ENSEMBLE EVALUATION COMPLETE")
print("===================================")

print("\nIndividual results:")
print(individual_file)

print("\nAll model predictions:")
print(prediction_file)

print("\nSubject-level results:")
print(subject_file)

print("\nEnsemble results:")
print(ensemble_file)
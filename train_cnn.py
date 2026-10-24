import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50,
    DenseNet121,
    EfficientNetB0,
    EfficientNetV2B0
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================================================
# SETTINGS
# =========================================================

DATASET_ROOT = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
)

IMAGE_DIR = os.path.join(DATASET_ROOT, "images")

TRAIN_FILE = os.path.join(
    DATASET_ROOT, "train_metadata.csv"
)

VAL_FILE = os.path.join(
    DATASET_ROOT, "validation_metadata.csv"
)

TEST_FILE = os.path.join(
    DATASET_ROOT, "test_metadata.csv"
)

RESULTS_DIR = os.path.join(
    DATASET_ROOT, "model_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

# =========================================================
# CHOOSE MODEL
# =========================================================

MODEL_NAME = "EfficientNetV2B0"

# Options:
# ResNet50
# DenseNet121
# EfficientNetB0
# EfficientNetV2B0

# =========================================================
# LOAD CSV
# =========================================================

print("Loading metadata...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

# Convert labels to integer
train_df["Label"] = train_df["Label"].astype(int)
val_df["Label"] = val_df["Label"].astype(int)
test_df["Label"] = test_df["Label"].astype(int)

print("\nTrain images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

# =========================================================
# IMAGE PATHS
# =========================================================

def add_image_path(df):
    df = df.copy()

    df["Image_Path"] = df["Image_Name"].apply(
        lambda x: os.path.join(IMAGE_DIR, str(x))
    )

    return df


train_df = add_image_path(train_df)
val_df = add_image_path(val_df)
test_df = add_image_path(test_df)

# =========================================================
# CHECK IMAGE FILES
# =========================================================

def check_files(df, name):

    missing = [
        path for path in df["Image_Path"]
        if not os.path.exists(path)
    ]

    print(f"\n{name} missing images:", len(missing))

    if len(missing) > 0:
        print("First missing file:")
        print(missing[0])
        raise FileNotFoundError("Missing image detected.")


check_files(train_df, "Train")
check_files(val_df, "Validation")
check_files(test_df, "Test")

# =========================================================
# TF.DATA DATASET
# =========================================================

def load_image(path, label):

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

    return image, tf.cast(label, tf.float32)


def make_dataset(df, shuffle=False):

    paths = df["Image_Path"].values
    labels = df["Label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = make_dataset(
    train_df,
    shuffle=True
)

val_ds = make_dataset(
    val_df
)

test_ds = make_dataset(
    test_df
)

# =========================================================
# MODEL BUILDER
# =========================================================

def build_model(model_name):

    if model_name == "ResNet50":

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    elif model_name == "DenseNet121":

        base_model = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    elif model_name == "EfficientNetB0":

        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    elif model_name == "EfficientNetV2B0":

        base_model = EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    else:
        raise ValueError("Unknown model name.")

    # Freeze pretrained layers
    base_model.trainable = False

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    return model


model = build_model(MODEL_NAME)

# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# =========================================================
# MODEL SUMMARY
# =========================================================

print("\n===================================")
print("MODEL:", MODEL_NAME)
print("===================================")

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

model_file = os.path.join(
    RESULTS_DIR,
    MODEL_NAME + "_best.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        model_file,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )
]

# =========================================================
# TRAIN
# =========================================================

print("\nStarting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# TEST EVALUATION
# =========================================================

print("\n===================================")
print("TEST RESULTS")
print("===================================")

test_results = model.evaluate(
    test_ds,
    return_dict=True
)

for metric, value in test_results.items():

    print(
        f"{metric}: {value:.4f}"
    )

# =========================================================
# SAVE HISTORY
# =========================================================

history_df = pd.DataFrame(
    history.history
)

history_file = os.path.join(
    RESULTS_DIR,
    MODEL_NAME + "_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)

# =========================================================
# SAVE TEST RESULTS
# =========================================================

results_df = pd.DataFrame(
    [test_results]
)

results_file = os.path.join(
    RESULTS_DIR,
    MODEL_NAME + "_test_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")

print("Best model:")
print(model_file)

print("\nTraining history:")
print(history_file)

print("\nTest results:")
print(results_file)
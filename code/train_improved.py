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

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# SETTINGS
# =========================================================

DATASET_ROOT = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
)

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "images"
)

TRAIN_FILE = os.path.join(
    DATASET_ROOT,
    "train_metadata.csv"
)

VAL_FILE = os.path.join(
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

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


IMG_SIZE = (224, 224)

BATCH_SIZE = 16

SEED = 42


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "ResNet50"

# Options:
#
# ResNet50
# DenseNet121
# EfficientNetB0
# EfficientNetV2B0


# =========================================================
# TRAINING SETTINGS
# =========================================================

# Stage 1
STAGE1_EPOCHS = 5

# Stage 2
STAGE2_EPOCHS = 20

# Learning rates
STAGE1_LR = 1e-4
STAGE2_LR = 5e-6


# =========================================================
# RANDOM SEED
# =========================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading metadata...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df["Label"] = train_df["Label"].astype(int)
val_df["Label"] = val_df["Label"].astype(int)
test_df["Label"] = test_df["Label"].astype(int)


print("\n===================================")
print("DATASET")
print("===================================")

print("Train images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))


# =========================================================
# IMAGE PATHS
# =========================================================

def add_image_path(df):

    df = df.copy()

    df["Image_Path"] = df["Image_Name"].apply(
        lambda x: os.path.join(
            IMAGE_DIR,
            str(x)
        )
    )

    return df


train_df = add_image_path(train_df)
val_df = add_image_path(val_df)
test_df = add_image_path(test_df)


# =========================================================
# CHECK FILES
# =========================================================

def check_files(df, name):

    missing = [
        path
        for path in df["Image_Path"]
        if not os.path.exists(path)
    ]

    print(
        f"{name} missing images:",
        len(missing)
    )

    if len(missing) > 0:

        print(
            "First missing:",
            missing[0]
        )

        raise FileNotFoundError(
            "Missing image detected."
        )


check_files(train_df, "Train")
check_files(val_df, "Validation")
check_files(test_df, "Test")


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n===================================")
print("CLASS DISTRIBUTION")
print("===================================")

print(
    train_df["Label"].value_counts()
    .sort_index()
)


# =========================================================
# CLASS WEIGHTS
# =========================================================

classes = np.unique(
    train_df["Label"]
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["Label"]
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass weights:")

for key, value in class_weights.items():

    print(
        f"Class {key}: {value:.4f}"
    )


# =========================================================
# IMAGE LOADING
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

    return image, tf.cast(
        label,
        tf.float32
    )


# =========================================================
# DATASET CREATION
# =========================================================

def make_dataset(
    df,
    shuffle=False
):

    paths = df["Image_Path"].values

    labels = df["Label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
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

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

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
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential(
    [

        layers.RandomRotation(
            0.03
        ),

        layers.RandomZoom(
            0.08
        ),

        layers.RandomTranslation(
            height_factor=0.03,
            width_factor=0.03
        )

    ],
    name="data_augmentation"
)


# =========================================================
# MODEL BUILDER
# =========================================================

def get_base_model(model_name):

    if model_name == "ResNet50":

        return ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    elif model_name == "DenseNet121":

        return DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    elif model_name == "EfficientNetB0":

        return EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    elif model_name == "EfficientNetV2B0":

        return EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    else:

        raise ValueError(
            "Unknown model name."
        )


base_model = get_base_model(
    MODEL_NAME
)


# =========================================================
# STAGE 1
# FREEZE BACKBONE
# =========================================================

base_model.trainable = False


inputs = layers.Input(
    shape=(224, 224, 3)
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(
    0.4
)(x)


outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# =========================================================
# COMPILE STAGE 1
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=STAGE1_LR
    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )

    ]
)


# =========================================================
# FILES
# =========================================================

model_file = os.path.join(
    RESULTS_DIR,
    MODEL_NAME +
    "_improved_best.keras"
)

history_file = os.path.join(
    RESULTS_DIR,
    MODEL_NAME +
    "_improved_history.csv"
)

results_file = os.path.join(
    RESULTS_DIR,
    MODEL_NAME +
    "_improved_test_results.csv"
)


# =========================================================
# CALLBACKS
# =========================================================

callbacks_stage1 = [

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
        model_file,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )

]


# =========================================================
# STAGE 1 TRAINING
# =========================================================

print("\n===================================")
print("STAGE 1 TRAINING")
print("MODEL:", MODEL_NAME)
print("===================================")


history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE1_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_stage1
)


# =========================================================
# STAGE 2
# FINE-TUNE LAST 50 LAYERS
# =========================================================

print("\n===================================")
print("STAGE 2 FINE-TUNING")
print("===================================")


base_model.trainable = True


# Freeze all except last 50 layers

for layer in base_model.layers[:-50]:

    layer.trainable = False


# Keep BatchNormalization frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# =========================================================
# COMPILE STAGE 2
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=STAGE2_LR
    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )

    ]
)


# =========================================================
# STAGE 2 CALLBACKS
# =========================================================

callbacks_stage2 = [

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
        model_file,
        monitor="val_auc",
        mode="max",
        save_best_only=True
    )

]


# =========================================================
# STAGE 2 TRAINING
# =========================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE2_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_stage2
)


# =========================================================
# TEST
# =========================================================

print("\n===================================")
print("FINAL TEST RESULTS")
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

history1_df = pd.DataFrame(
    history1.history
)

history2_df = pd.DataFrame(
    history2.history
)


history1_df["stage"] = 1
history2_df["stage"] = 2


combined_history = pd.concat(
    [
        history1_df,
        history2_df
    ],
    ignore_index=True
)


combined_history.to_csv(
    history_file,
    index=False
)


# =========================================================
# SAVE TEST RESULTS
# =========================================================

results_df = pd.DataFrame(
    [test_results]
)

results_df.to_csv(
    results_file,
    index=False
)


# =========================================================
# COMPLETE 
# =========================================================

print("\n===================================")
print("IMPROVED TRAINING COMPLETE")
print("===================================")

print("\nModel:")
print(MODEL_NAME)

print("\nBest model:")
print(model_file)

print("\nHistory:")
print(history_file)

print("\nTest results:")
print(results_file)
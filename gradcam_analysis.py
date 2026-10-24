import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


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

GRADCAM_DIR = os.path.join(
    RESULTS_DIR,
    "gradcam"
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)

IMG_SIZE = (224, 224)


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
# LOAD TEST DATA
# =========================================================

test_df = pd.read_csv(
    TEST_FILE
)

test_df["Label"] = test_df[
    "Label"
].astype(int)

test_df["Image_Path"] = test_df[
    "Image_Name"
].apply(
    lambda x: os.path.join(
        IMAGE_DIR,
        str(x)
    )
)


# =========================================================
# IMAGE LOADER
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
# FIND BACKBONE
# =========================================================

def find_backbone(model):

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            print(
                "Found backbone:",
                layer.name
            )

            return layer

    raise ValueError(
        "Could not find CNN backbone inside model."
    )


# =========================================================
# FIND LAST CONVOLUTIONAL LAYER
# =========================================================

def find_last_conv_layer(backbone):

    for layer in reversed(
        backbone.layers
    ):

        try:

            output_shape = layer.output.shape

            if (
                len(output_shape) == 4
                and
                output_shape[-1] is not None
            ):

                print(
                    "Using convolutional layer:",
                    layer.name
                )

                return layer

        except Exception:
            continue

    raise ValueError(
        "Could not find convolutional layer."
    )


# =========================================================
# BUILD GRAD-CAM MODEL
# =========================================================

def build_gradcam_model(
    model,
    backbone,
    conv_layer
):

    # -----------------------------------------------------
    # We create a new graph:
    #
    # Input
    #   ↓
    # Backbone
    #   ↓
    # Selected convolutional layer
    #   ↓
    # Global Average Pooling
    #   ↓
    # Dropout
    #   ↓
    # Dense classifier
    #
    # This keeps the gradient connection intact.
    # -----------------------------------------------------

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    # Get convolutional feature map
    conv_output_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=conv_layer.output
    )

    conv_outputs = conv_output_model(
        inputs
    )

    # -----------------------------------------------------
    # Find classifier layers after backbone
    # -----------------------------------------------------

    x = conv_outputs

    # GlobalAveragePooling
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="gradcam_gap"
    )(x)

    # -----------------------------------------------------
    # Get original classifier weights
    # -----------------------------------------------------

    dense_layer = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Dense
        ):

            if layer.units == 1:

                dense_layer = layer

    if dense_layer is None:

        raise ValueError(
            "Could not find final Dense(1) classifier."
        )

    # -----------------------------------------------------
    # Apply Dropout only if needed
    #
    # Dropout is disabled during inference anyway.
    # -----------------------------------------------------

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Dropout
        ):

            x = layer(
                x,
                training=False
            )

    # -----------------------------------------------------
    # Apply original classifier
    # -----------------------------------------------------

    predictions = dense_layer(
        x
    )

    gradcam_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            conv_outputs,
            predictions
        ]
    )

    return gradcam_model


# =========================================================
# MAKE GRAD-CAM HEATMAP
# =========================================================

def make_gradcam_heatmap(
    image,
    gradcam_model
):

    # -----------------------------------------------------
    # GradientTape
    # -----------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            gradcam_model(
                image,
                training=False
            )
        )

        # Positive-class score
        class_score = predictions[:, 0]

    # -----------------------------------------------------
    # Calculate gradients
    # -----------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients are None. "
            "Grad-CAM graph is not connected."
        )

    # -----------------------------------------------------
    # Global average pooling of gradients
    # -----------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # -----------------------------------------------------
    # Remove batch dimension
    # -----------------------------------------------------

    conv_outputs = conv_outputs[0]

    # -----------------------------------------------------
    # Weighted feature maps
    # -----------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs *
        pooled_gradients,
        axis=-1
    )

    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        tf.zeros_like(heatmap)
    )

    return heatmap.numpy()


# =========================================================
# SAVE GRAD-CAM IMAGE
# =========================================================

def save_gradcam(
    original_image,
    heatmap,
    save_path,
    title
):

    original_image = (
        original_image.numpy()
        .astype(np.uint8)
    )

    # Resize heatmap
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE
    ).numpy().squeeze()

    plt.figure(
        figsize=(6, 6)
    )

    # Original MRI
    plt.imshow(
        original_image
    )

    # Grad-CAM overlay
    plt.imshow(
        heatmap_resized,
        alpha=0.45,
        cmap="jet"
    )

    plt.axis(
        "off"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# SELECT CLASS 0 IMAGE
# =========================================================

class_0_rows = test_df[
    test_df["Label"] == 0
]

if len(class_0_rows) == 0:

    raise ValueError(
        "No class-0 image found in test dataset."
    )

class_0_image = class_0_rows.iloc[0]


# =========================================================
# SELECT CLASS 1 IMAGE
# =========================================================

class_1_rows = test_df[
    test_df["Label"] == 1
]

if len(class_1_rows) == 0:

    raise ValueError(
        "No class-1 image found in test dataset."
    )

class_1_image = class_1_rows.iloc[0]


selected_images = [
    class_0_image,
    class_1_image
]


# =========================================================
# RUN GRAD-CAM
# =========================================================

for model_name, model_file in MODEL_FILES.items():

    print("\n===================================")
    print(
        "MODEL:",
        model_name
    )
    print("===================================")

    model_path = os.path.join(
        RESULTS_DIR,
        model_file
    )

    if not os.path.exists(
        model_path
    ):

        print(
            "Model not found:",
            model_path
        )

        continue

    print(
        "Loading model..."
    )

    model = tf.keras.models.load_model(
        model_path
    )

    # -----------------------------------------------------
    # Find backbone
    # -----------------------------------------------------

    backbone = find_backbone(
        model
    )

    # -----------------------------------------------------
    # Find last convolutional layer
    # -----------------------------------------------------

    conv_layer = find_last_conv_layer(
        backbone
    )

    print(
        "Backbone:",
        backbone.name
    )

    print(
        "Grad-CAM layer:",
        conv_layer.name
    )

    # -----------------------------------------------------
    # Build Grad-CAM model
    # -----------------------------------------------------

    print(
        "Building Grad-CAM model..."
    )

    gradcam_model = build_gradcam_model(
        model,
        backbone,
        conv_layer
    )

    print(
        "Grad-CAM model ready."
    )

    # -----------------------------------------------------
    # Output directory
    # -----------------------------------------------------

    model_output_dir = os.path.join(
        GRADCAM_DIR,
        model_name
    )

    os.makedirs(
        model_output_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Generate Grad-CAM
    # -----------------------------------------------------

    for row in selected_images:

        image_path = row[
            "Image_Path"
        ]

        image_name = row[
            "Image_Name"
        ]

        true_label = int(
            row["Label"]
        )

        print(
            "\nProcessing:",
            image_name
        )

        # -------------------------------------------------
        # Load image
        # -------------------------------------------------

        original_image = load_image(
            image_path
        )

        input_image = tf.expand_dims(
            original_image,
            axis=0
        )

        # -------------------------------------------------
        # Prediction using original model
        # -------------------------------------------------

        prediction = model.predict(
            input_image,
            verbose=0
        )[0][0]

        predicted_label = int(
            prediction >= 0.5
        )

        print(
            "True label:",
            true_label
        )

        print(
            "Prediction:",
            predicted_label
        )

        print(
            "Probability:",
            round(
                float(prediction),
                4
            )
        )

        # -------------------------------------------------
        # Grad-CAM
        # -------------------------------------------------

        heatmap = make_gradcam_heatmap(
            input_image,
            gradcam_model
        )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        safe_name = os.path.splitext(
            image_name
        )[0]

        save_path = os.path.join(
            model_output_dir,
            safe_name +
            "_gradcam.png"
        )

        title = (
            f"{model_name} | "
            f"True={true_label} | "
            f"Pred={predicted_label} | "
            f"Prob={prediction:.3f}"
        )

        save_gradcam(
            original_image,
            heatmap,
            save_path,
            title
        )

        print(
            "Saved:",
            save_path
        )


# =========================================================
# COMPLETE
# =========================================================

print("\n===================================")
print("GRAD-CAM COMPLETE")
print("===================================")

print(
    "\nResults saved in:"
)

print(
    GRADCAM_DIR
)
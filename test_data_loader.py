import pandas as pd
import os
from PIL import Image
import numpy as np

# =========================================================
# PATHS
# =========================================================

DATASET_ROOT = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_1_12"
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

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "images"
)

# =========================================================
# LOAD CSV FILES
# =========================================================

print("Loading metadata...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTrain images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

# =========================================================
# IMAGE PATH FUNCTION
# =========================================================

def get_image_path(row):

    image_name = str(row["Image_Name"])

    path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    return path

# =========================================================
# TEST ONE IMAGE
# =========================================================

sample = train_df.iloc[0]

image_path = get_image_path(sample)

print("\nTesting image:")
print(image_path)

if not os.path.exists(image_path):

    print("\nERROR: Image not found!")
    print("Check image path.")

else:

    print("\nImage found!")

    image = Image.open(image_path)

    print("Original image size:", image.size)
    print("Image mode:", image.mode)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((224, 224))

    image_array = np.array(image)

    print("After resize:", image_array.shape)

    # =====================================================
    # CHECK LABEL
    # =====================================================

    print("\nSubject ID:", sample["Subject_ID"])
    print("CDR:", sample["CDR"])
    print("Label:", sample["Label"])

    print("\nDATA LOADER TEST PASSED!")
import os
import shutil
import csv


# ============================================================
# DATASET 1–4
# ============================================================

DATASET_1_4_ROOT = r"C:\Users\rog\Downloads\images"

DATASET_1_4_IMAGES = os.path.join(
    DATASET_1_4_ROOT,
    "images"
)

DATASET_1_4_METADATA = os.path.join(
    DATASET_1_4_ROOT,
    "image_metadata.csv"
)


# ============================================================
# DATASETS 5–12
# ============================================================

BASE_DIR = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"

DATASET_5_6 = os.path.join(
    BASE_DIR,
    "OASIS_CNN_Dataset_5_6"
)

DATASET_7_9 = os.path.join(
    BASE_DIR,
    "OASIS_CNN_Dataset_7_9"
)

DATASET_10_12 = os.path.join(
    BASE_DIR,
    "OASIS_CNN_Dataset_10_12"
)


# ============================================================
# FINAL MERGED DATASET
# ============================================================

MERGED_ROOT = os.path.join(
    BASE_DIR,
    "OASIS_CNN_Dataset_1_12"
)

MERGED_IMAGE_ROOT = os.path.join(
    MERGED_ROOT,
    "images"
)

MERGED_METADATA = os.path.join(
    MERGED_ROOT,
    "image_metadata.csv"
)


# ============================================================
# CREATE MERGED FOLDER
# ============================================================

os.makedirs(
    MERGED_IMAGE_ROOT,
    exist_ok=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

DATASETS = [

    {
        "name": "Disc 1–4",
        "images": DATASET_1_4_IMAGES,
        "metadata": DATASET_1_4_METADATA
    },

    {
        "name": "Disc 5–6",
        "images": os.path.join(
            DATASET_5_6,
            "images"
        ),
        "metadata": os.path.join(
            DATASET_5_6,
            "image_metadata.csv"
        )
    },

    {
        "name": "Disc 7–9",
        "images": os.path.join(
            DATASET_7_9,
            "images"
        ),
        "metadata": os.path.join(
            DATASET_7_9,
            "image_metadata.csv"
        )
    },

    {
        "name": "Disc 10–12",
        "images": os.path.join(
            DATASET_10_12,
            "images"
        ),
        "metadata": os.path.join(
            DATASET_10_12,
            "image_metadata.csv"
        )
    }
]


# ============================================================
# HEADER
# ============================================================

metadata_header = [
    "Subject_ID",
    "Original_MRI",
    "Image_Path",
    "Image_Name",
    "Orientation",
    "Width",
    "Height"
]


all_metadata = []

total_images = 0
total_datasets = 0


print("\n========================================")
print("MERGING OASIS MRI DATASETS 1–12")
print("========================================")


# ============================================================
# PROCESS EACH DATASET
# ============================================================

for dataset in DATASETS:

    dataset_name = dataset["name"]
    image_folder = dataset["images"]
    metadata_file = dataset["metadata"]


    print("\n----------------------------------------")
    print(f"Checking {dataset_name}")
    print("----------------------------------------")


    # --------------------------------------------------------
    # CHECK IMAGE FOLDER
    # --------------------------------------------------------

    if not os.path.exists(image_folder):

        print(
            f"ERROR: Images folder not found:\n"
            f"{image_folder}"
        )

        continue


    # --------------------------------------------------------
    # CHECK METADATA
    # --------------------------------------------------------

    if not os.path.exists(metadata_file):

        print(
            f"ERROR: Metadata file not found:\n"
            f"{metadata_file}"
        )

        continue


    # --------------------------------------------------------
    # COPY IMAGES
    # --------------------------------------------------------

    image_count = 0


    for filename in os.listdir(image_folder):

        source_image = os.path.join(
            image_folder,
            filename
        )


        if not os.path.isfile(source_image):
            continue


        destination_image = os.path.join(
            MERGED_IMAGE_ROOT,
            filename
        )


        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        if os.path.exists(destination_image):

            print(
                f"WARNING: Duplicate skipped: "
                f"{filename}"
            )

            continue


        shutil.copy2(
            source_image,
            destination_image
        )

        image_count += 1


    print(
        f"Images copied from {dataset_name}: "
        f"{image_count}"
    )


    # --------------------------------------------------------
    # READ METADATA
    # --------------------------------------------------------

    with open(
        metadata_file,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)


        for row in reader:

            image_name = row["Image_Name"]


            # New path inside merged dataset
            new_image_path = os.path.join(
                MERGED_IMAGE_ROOT,
                image_name
            )


            all_metadata.append([

                row["Subject_ID"],

                row["Original_MRI"],

                new_image_path,

                image_name,

                row["Orientation"],

                row["Width"],

                row["Height"]

            ])


    total_images += image_count

    total_datasets += 1


# ============================================================
# SAVE MERGED METADATA
# ============================================================

with open(
    MERGED_METADATA,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)


    writer.writerow(
        metadata_header
    )


    writer.writerows(
        all_metadata
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("MERGING COMPLETE")
print("========================================")


print(
    f"Datasets successfully merged: "
    f"{total_datasets}"
)


print(
    f"Total images copied: "
    f"{total_images}"
)


print(
    f"Total metadata rows: "
    f"{len(all_metadata)}"
)


print(
    f"\nMerged dataset location:"
)


print(
    MERGED_ROOT
)


print(
    f"\nMerged images location:"
)


print(
    MERGED_IMAGE_ROOT
)


print(
    f"\nMerged metadata:"
)


print(
    MERGED_METADATA
)


print("\n========================================")
import os
import csv
import re


# ============================================================
# DISC 1–4 PROCESSED IMAGES
# ============================================================

IMAGE_ROOT = r"C:\Users\rog\Downloads\images\images"


# ============================================================
# METADATA OUTPUT
# ============================================================

OUTPUT_CSV = r"C:\Users\rog\Downloads\images\image_metadata.csv"


IMAGE_SIZE = (224, 224)


# ============================================================
# CHECK FOLDER
# ============================================================

if not os.path.exists(IMAGE_ROOT):

    print("ERROR: Image folder not found:")
    print(IMAGE_ROOT)
    exit()


# ============================================================
# FIND PNG FILES
# ============================================================

png_files = [
    f for f in os.listdir(IMAGE_ROOT)
    if f.lower().endswith(".png")
]


print("\n========================================")
print("CREATING METADATA FOR OASIS DISC 1–4")
print("========================================")

print(
    f"PNG files found: {len(png_files)}"
)


# ============================================================
# FILE NAME PATTERN
# Example:
# OAS1_0063_MR1_sagittal_00.png
# OAS1_0064_MR1_axial_03.png
# ============================================================

pattern = re.compile(
    r"^(OAS1_\d+_MR\d+)_(sagittal|coronal|axial)_(\d+)\.png$",
    re.IGNORECASE
)


metadata = []

skipped = 0


# ============================================================
# CREATE METADATA
# ============================================================

for filename in sorted(png_files):

    match = pattern.match(filename)


    if not match:

        print(
            f"WARNING: Filename pattern not recognized: "
            f"{filename}"
        )

        skipped += 1

        continue


    subject_id = match.group(1)

    orientation = match.group(2).lower()


    image_path = os.path.join(
        IMAGE_ROOT,
        filename
    )


    metadata.append([
        subject_id,
        "",
        image_path,
        filename,
        orientation,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    ])


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)


    writer.writerow([
        "Subject_ID",
        "Original_MRI",
        "Image_Path",
        "Image_Name",
        "Orientation",
        "Width",
        "Height"
    ])


    writer.writerows(metadata)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("METADATA CREATION COMPLETE")
print("========================================")

print(
    f"PNG files found: {len(png_files)}"
)

print(
    f"Metadata rows created: {len(metadata)}"
)

print(
    f"Files skipped: {skipped}"
)

print(
    f"\nMetadata saved at:"
)

print(
    OUTPUT_CSV
)

print("========================================")
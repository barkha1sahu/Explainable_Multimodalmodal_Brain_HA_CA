import pandas as pd
import os


# ============================================================
# INPUT
# ============================================================

INPUT_CSV = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET\OASIS_CNN_Dataset_1_12\image_metadata.csv"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CSV = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET\OASIS_CNN_Dataset_1_12\image_metadata_cleaned.csv"


# ============================================================
# LOAD CSV
# ============================================================

print("\n========================================")
print("CLEANING MRI METADATA")
print("========================================")

df = pd.read_csv(INPUT_CSV)

print(f"Original rows: {len(df)}")


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicates = df[df.duplicated(
    subset=["Image_Name"],
    keep=False
)]


print(
    f"Duplicate image-name rows found: "
    f"{len(duplicates)}"
)


if len(duplicates) > 0:

    print("\nDuplicate entries:")

    print(
        duplicates[
            [
                "Subject_ID",
                "Image_Name",
                "Orientation"
            ]
        ].to_string(index=False)
    )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

df_clean = df.drop_duplicates(
    subset=["Image_Name"],
    keep="first"
).copy()


# ============================================================
# SAVE CLEANED CSV
# ============================================================

df_clean.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n========================================")
print("CLEANING COMPLETE")
print("========================================")

print(
    f"Original metadata rows: "
    f"{len(df)}"
)

print(
    f"Cleaned metadata rows: "
    f"{len(df_clean)}"
)

print(
    f"Rows removed: "
    f"{len(df) - len(df_clean)}"
)

print(
    f"\nCleaned CSV saved at:"
)

print(OUTPUT_CSV)

print("========================================")
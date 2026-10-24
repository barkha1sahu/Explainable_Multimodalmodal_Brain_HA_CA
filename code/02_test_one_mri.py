import os
import numpy as np
import nibabel as nib
from PIL import Image


# ============================================================
# 1. ONE MRI FILE FOR TESTING
# ============================================================

MRI_PATH = r"C:\Users\rog\Downloads\oasis_cross-sectional_disc7\disc7\OAS1_0232_MR1\RAW\OAS1_0232_MR1_mpr-1_anon.img"


# ============================================================
# 2. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET\test_output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 3. CHECK FILE
# ============================================================

print("=" * 60)
print("TESTING ONE OASIS MRI")
print("=" * 60)

print()
print("MRI file:")
print(MRI_PATH)

if not os.path.exists(MRI_PATH):

    print()
    print("ERROR: MRI file was not found.")
    exit()

print()
print("MRI file found successfully.")


# ============================================================
# 4. CHECK HDR FILE
# ============================================================

HDR_PATH = os.path.splitext(MRI_PATH)[0] + ".hdr"

print()
print("HDR file:")
print(HDR_PATH)

if os.path.exists(HDR_PATH):

    print("HDR file found successfully.")

else:

    print("ERROR: HDR file was not found.")
    exit()


# ============================================================
# 5. LOAD MRI
# ============================================================

print()
print("Loading MRI...")

mri = nib.load(MRI_PATH)

print("MRI loaded successfully.")


# ============================================================
# 6. SHOW MRI INFORMATION
# ============================================================

print()
print("MRI shape:")
print(mri.shape)

print()
print("Voxel size:")
print(mri.header.get_zooms())

print()
print("Data type:")
print(mri.get_data_dtype())


# ============================================================
# 7. READ MRI DATA
# ============================================================

print()
print("Reading MRI data...")

volume = np.asarray(mri.get_fdata())

print("Original shape:")
print(volume.shape)


# ============================================================
# 8. REMOVE EXTRA DIMENSIONS
# ============================================================

volume = np.squeeze(volume)

print()
print("Shape after squeeze:")
print(volume.shape)


# ============================================================
# 9. CHECK 3D
# ============================================================

if volume.ndim != 3:

    print()
    print("ERROR: MRI is not a 3D volume.")
    exit()

print()
print("SUCCESS: MRI is a 3D volume.")


# ============================================================
# 10. BASIC INFORMATION
# ============================================================

print()
print("Minimum value:", np.min(volume))
print("Maximum value:", np.max(volume))
print("Mean value:", np.mean(volume))


# ============================================================
# 11. TAKE ONE CENTRAL AXIAL SLICE
# ============================================================

x, y, z = volume.shape

middle_slice = volume[:, :, z // 2]


# ============================================================
# 12. NORMALIZE SLICE
# ============================================================

middle_slice = np.nan_to_num(middle_slice)

low = np.percentile(middle_slice, 1)
high = np.percentile(middle_slice, 99)

middle_slice = np.clip(
    middle_slice,
    low,
    high
)

middle_slice = (
    (middle_slice - low)
    / (high - low)
    * 255
)

middle_slice = middle_slice.astype(np.uint8)


# ============================================================
# 13. CONVERT TO IMAGE
# ============================================================

image = Image.fromarray(middle_slice)

image = image.resize(
    (224, 224),
    Image.Resampling.BILINEAR
)


# ============================================================
# 14. SAVE IMAGE
# ============================================================

output_path = os.path.join(
    OUTPUT_FOLDER,
    "OAS1_0232_MR1_test.png"
)

image.save(output_path)


# ============================================================
# 15. FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)

print()
print("PNG saved at:")

print(output_path)
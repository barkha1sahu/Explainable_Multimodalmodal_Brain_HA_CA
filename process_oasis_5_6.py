import os
import csv
import numpy as np
from PIL import Image
import nibabel as nib


# ============================================================
# PATHS — OASIS DISC 5 & 6
# ============================================================

DATASET_ROOTS = [
    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc5\disc5",
    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc6\disc6"
]

OUTPUT_ROOT = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_5_6"
)

IMAGE_ROOT = os.path.join(OUTPUT_ROOT, "images")

IMAGE_SIZE = (224, 224)

MRI_SUFFIX = "_mpr-1_anon.img"


# ============================================================
# NORMALIZE SLICE
# ============================================================

def normalize_slice(slice_data):

    slice_data = np.nan_to_num(slice_data)

    min_val = np.min(slice_data)
    max_val = np.max(slice_data)

    if max_val == min_val:
        return np.zeros_like(slice_data, dtype=np.uint8)

    normalized = (
        (slice_data - min_val)
        / (max_val - min_val)
    )

    normalized = (
        normalized * 255
    ).astype(np.uint8)

    return normalized


# ============================================================
# GET SUBJECT ID
# ============================================================

def get_subject_id(mri_path):

    for part in mri_path.split(os.sep):

        if part.startswith("OAS1_"):
            return part

    return os.path.basename(
        mri_path
    ).split("_MR")[0]


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(slice_data, output_path):

    slice_data = normalize_slice(slice_data)

    image = Image.fromarray(slice_data)

    image = image.resize(IMAGE_SIZE)

    image.save(output_path)


# ============================================================
# PROCESS ONE MRI
# ============================================================

def process_mri(mri_path, output_dir):

    try:

        # ----------------------------------------------------
        # LOAD MRI
        # ----------------------------------------------------

        img = nib.load(mri_path)

        volume = img.get_fdata()


        # ----------------------------------------------------
        # HANDLE 4D MRI
        # ----------------------------------------------------

        if volume.ndim == 4:

            volume = volume[:, :, :, 0]


        # Make sure volume is 3D
        if volume.ndim != 3:

            raise ValueError(
                f"Unexpected MRI dimensions: "
                f"{volume.shape}"
            )


        # ----------------------------------------------------
        # SUBJECT ID
        # ----------------------------------------------------

        subject_id = get_subject_id(mri_path)


        # ----------------------------------------------------
        # BASE FILE NAME
        # ----------------------------------------------------

        base_name = os.path.basename(
            mri_path
        ).replace(
            "_mpr-1_anon.img",
            ""
        )


        metadata = []


        # ----------------------------------------------------
        # VOLUME DIMENSIONS
        # ----------------------------------------------------

        x, y, z = volume.shape


        # ====================================================
        # 7 SAGITTAL SLICES
        # ====================================================

        sagittal_indices = np.linspace(
            0,
            x - 1,
            7,
            dtype=int
        )

        for i, idx in enumerate(
            sagittal_indices
        ):

            slice_data = volume[idx, :, :]

            image_name = (
                f"{base_name}"
                f"_sagittal_{i+1}.png"
            )

            image_path = os.path.join(
                output_dir,
                image_name
            )

            save_image(
                slice_data,
                image_path
            )

            metadata.append([
                subject_id,
                mri_path,
                image_path,
                image_name,
                "sagittal",
                IMAGE_SIZE[0],
                IMAGE_SIZE[1]
            ])


        # ====================================================
        # 7 CORONAL SLICES
        # ====================================================

        coronal_indices = np.linspace(
            0,
            y - 1,
            7,
            dtype=int
        )

        for i, idx in enumerate(
            coronal_indices
        ):

            slice_data = volume[:, idx, :]

            image_name = (
                f"{base_name}"
                f"_coronal_{i+1}.png"
            )

            image_path = os.path.join(
                output_dir,
                image_name
            )

            save_image(
                slice_data,
                image_path
            )

            metadata.append([
                subject_id,
                mri_path,
                image_path,
                image_name,
                "coronal",
                IMAGE_SIZE[0],
                IMAGE_SIZE[1]
            ])


        # ====================================================
        # 7 AXIAL SLICES
        # ====================================================

        axial_indices = np.linspace(
            0,
            z - 1,
            7,
            dtype=int
        )

        for i, idx in enumerate(
            axial_indices
        ):

            slice_data = volume[:, :, idx]

            image_name = (
                f"{base_name}"
                f"_axial_{i+1}.png"
            )

            image_path = os.path.join(
                output_dir,
                image_name
            )

            save_image(
                slice_data,
                image_path
            )

            metadata.append([
                subject_id,
                mri_path,
                image_path,
                image_name,
                "axial",
                IMAGE_SIZE[0],
                IMAGE_SIZE[1]
            ])


        return metadata


    except Exception as e:

        print(
            f"ERROR processing {mri_path}"
        )

        print(e)

        return []


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs(
    IMAGE_ROOT,
    exist_ok=True
)


# ============================================================
# FIND MRI FILES
# ============================================================

mri_files = []


for root in DATASET_ROOTS:

    if not os.path.exists(root):

        print(
            "WARNING: Folder not found:"
        )

        print(root)

        continue


    for dirpath, dirnames, filenames in os.walk(
        root
    ):

        for filename in filenames:

            if filename.endswith(
                MRI_SUFFIX
            ):

                img_path = os.path.join(
                    dirpath,
                    filename
                )

                # Matching HDR file
                hdr_path = img_path.replace(
                    ".img",
                    ".hdr"
                )


                if os.path.exists(
                    hdr_path
                ):

                    mri_files.append(
                        img_path
                    )


# ============================================================
# PROCESS ALL MRI FILES
# ============================================================

all_metadata = []

successful = 0
failed = 0


print("\n========================================")
print("OASIS DISC 5 & 6 MRI PROCESSING")
print("========================================")

print(
    f"Total MRI files found: "
    f"{len(mri_files)}"
)

print()


for i, mri_path in enumerate(
    mri_files,
    start=1
):

    print(
        f"Processing {i}/{len(mri_files)}: "
        f"{mri_path}"
    )


    metadata = process_mri(
        mri_path,
        IMAGE_ROOT
    )


    if metadata:

        all_metadata.extend(
            metadata
        )

        successful += 1

    else:

        failed += 1


# ============================================================
# SAVE METADATA CSV
# ============================================================

metadata_csv = os.path.join(
    OUTPUT_ROOT,
    "image_metadata.csv"
)


with open(
    metadata_csv,
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


    writer.writerows(
        all_metadata
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("PROCESSING COMPLETE")
print("========================================")


print(
    f"Total MRI files processed: "
    f"{len(mri_files)}"
)


print(
    f"MRI files successfully processed: "
    f"{successful}"
)


print(
    f"MRI files failed: "
    f"{failed}"
)


print(
    f"Total PNG images generated: "
    f"{len(all_metadata)}"
)


print(
    f"\nMetadata saved at:\n"
    f"{metadata_csv}"
)


print(
    f"\nImages saved at:\n"
    f"{IMAGE_ROOT}"
)


print("========================================")

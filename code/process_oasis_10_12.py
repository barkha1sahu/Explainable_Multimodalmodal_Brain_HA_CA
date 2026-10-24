import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from tqdm import tqdm


# ============================================================
# 1. OASIS DISCS 10, 11, 12
# ============================================================

DATASET_ROOTS = [

    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc10\disc10",

    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc11\disc11",

    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc12\disc12"

]


# ============================================================
# 2. FINAL OUTPUT FOLDER
# ============================================================

OUTPUT_ROOT = (
    r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"
    r"\OASIS_CNN_Dataset_10_12"
)

IMAGE_ROOT = os.path.join(
    OUTPUT_ROOT,
    "images"
)

os.makedirs(
    IMAGE_ROOT,
    exist_ok=True
)


# ============================================================
# 3. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

# Use the first MRI acquisition only
MRI_SUFFIX = "_mpr-1_anon.img"


# ============================================================
# 4. NORMALIZE MRI SLICE
# ============================================================

def normalize_slice(slice_data):

    slice_data = np.asarray(slice_data)

    # Remove NaN / infinite values
    slice_data = np.nan_to_num(
        slice_data
    )

    minimum = np.min(
        slice_data
    )

    maximum = np.max(
        slice_data
    )

    if maximum <= minimum:
        return None

    # Remove extreme intensity values
    low = np.percentile(
        slice_data,
        1
    )

    high = np.percentile(
        slice_data,
        99
    )

    if high <= low:
        return None

    slice_data = np.clip(
        slice_data,
        low,
        high
    )

    # Convert to 0-255
    slice_data = (
        (slice_data - low)
        / (high - low)
        * 255
    )

    return slice_data.astype(
        np.uint8
    )


# ============================================================
# 5. GET SUBJECT ID
# ============================================================

def get_subject_id(filename):

    match = re.search(
        r"(OAS1_\d{4}_MR\d+)",
        filename
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# 6. SAVE IMAGE
# ============================================================

def save_image(
    slice_data,
    subject_id,
    orientation,
    index
):

    slice_data = normalize_slice(
        slice_data
    )

    if slice_data is None:
        return None

    image = Image.fromarray(
        slice_data
    )

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    filename = (
        f"{subject_id}_"
        f"{orientation}_"
        f"{index:02d}.png"
    )

    output_path = os.path.join(
        IMAGE_ROOT,
        filename
    )

    image.save(
        output_path
    )

    return output_path


# ============================================================
# 7. PROCESS ONE MRI
# ============================================================

def process_mri(img_path):

    filename = os.path.basename(
        img_path
    )

    print("\n" + "-" * 70)
    print("Processing:", filename)
    print("-" * 70)


    # --------------------------------------------------------
    # SUBJECT ID
    # --------------------------------------------------------

    subject_id = get_subject_id(
        filename
    )

    print(
        "Subject ID:",
        subject_id
    )

    if subject_id is None:

        raise ValueError(
            f"Could not extract subject ID from {filename}"
        )


    # --------------------------------------------------------
    # LOAD MRI
    # --------------------------------------------------------

    print("Loading MRI...")

    mri = nib.load(
        img_path
    )

    print(
        "Nibabel shape:",
        mri.shape
    )

    print(
        "Data type:",
        mri.get_data_dtype()
    )

    print(
        "Voxel size:",
        mri.header.get_zooms()
    )


    # --------------------------------------------------------
    # READ MRI DATA
    # --------------------------------------------------------

    print("Reading MRI data...")

    volume = np.asarray(
        mri.get_fdata()
    )

    print(
        "Original NumPy shape:",
        volume.shape
    )


    # Handle 4D volume with one channel
    if (
        volume.ndim == 4
        and volume.shape[-1] == 1
    ):

        volume = volume[..., 0]


    # Remove dimensions of size 1
    volume = np.squeeze(
        volume
    )

    print(
        "Shape after squeeze:",
        volume.shape
    )


    # --------------------------------------------------------
    # CHECK 3D
    # --------------------------------------------------------

    if volume.ndim != 3:

        raise ValueError(
            f"Expected 3D MRI volume, got {volume.shape}"
        )


    # --------------------------------------------------------
    # CHECK EMPTY MRI
    # --------------------------------------------------------

    if np.all(
        volume == 0
    ):

        raise ValueError(
            "MRI contains only zeros"
        )


    print(
        "MRI is valid 3D data."
    )

    print(
        "Minimum:",
        np.min(volume)
    )

    print(
        "Maximum:",
        np.max(volume)
    )

    print(
        "Mean:",
        np.mean(volume)
    )


    # ========================================================
    # 8. MRI DIMENSIONS
    # ========================================================

    x, y, z = volume.shape

    print(
        "MRI dimensions:",
        x,
        y,
        z
    )


    # ========================================================
    # 9. SELECT 7 CENTRAL SLICES
    # ========================================================

    x_indices = np.linspace(
        int(x * 0.35),
        int(x * 0.65),
        7
    ).astype(int)

    y_indices = np.linspace(
        int(y * 0.35),
        int(y * 0.65),
        7
    ).astype(int)

    z_indices = np.linspace(
        int(z * 0.35),
        int(z * 0.65),
        7
    ).astype(int)


    saved_images = []


    # ========================================================
    # 10. SAGITTAL
    # ========================================================

    for i, index in enumerate(
        x_indices
    ):

        path = save_image(
            volume[index, :, :],
            subject_id,
            "sagittal",
            i
        )

        if path:
            saved_images.append(
                path
            )


    # ========================================================
    # 11. CORONAL
    # ========================================================

    for i, index in enumerate(
        y_indices
    ):

        path = save_image(
            volume[:, index, :],
            subject_id,
            "coronal",
            i
        )

        if path:
            saved_images.append(
                path
            )


    # ========================================================
    # 12. AXIAL
    # ========================================================

    for i, index in enumerate(
        z_indices
    ):

        path = save_image(
            volume[:, :, index],
            subject_id,
            "axial",
            i
        )

        if path:
            saved_images.append(
                path
            )


    print(
        "PNG images generated:",
        len(saved_images)
    )

    return saved_images


# ============================================================
# 13. SEARCH DISCS 10, 11, 12
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING OASIS DISCS 10, 11, 12")
print("=" * 70)


mri_files = []


for dataset_root in DATASET_ROOTS:

    print("\nSearching:")
    print(dataset_root)


    if not os.path.exists(
        dataset_root
    ):

        print(
            "WARNING: Folder does not exist!"
        )

        continue


    for root, dirs, files in os.walk(
        dataset_root
    ):

        for file in files:

            # Only mpr-1
            if file.lower().endswith(
                MRI_SUFFIX
            ):

                img_path = os.path.join(
                    root,
                    file
                )


                # Corresponding HDR
                hdr_path = (
                    os.path.splitext(
                        img_path
                    )[0]
                    + ".hdr"
                )


                # Require both IMG and HDR
                if os.path.exists(
                    hdr_path
                ):

                    mri_files.append(
                        img_path
                    )


# Remove accidental duplicate paths
mri_files = list(
    dict.fromkeys(
        mri_files
    )
)


print(
    "\nTotal MRI files found:",
    len(mri_files)
)


# ============================================================
# 14. PROCESS ALL MRI FILES
# ============================================================

metadata = []

successful = 0
failed = 0


for img_path in tqdm(
    mri_files,
    desc="Processing all MRI files"
):

    try:

        saved_images = process_mri(
            img_path
        )


        subject_id = get_subject_id(
            os.path.basename(
                img_path
            )
        )


        # ----------------------------------------------------
        # STORE METADATA
        # ----------------------------------------------------

        for image_path in saved_images:

            image_name = os.path.basename(
                image_path
            )


            if "sagittal" in image_name:

                orientation = "sagittal"

            elif "coronal" in image_name:

                orientation = "coronal"

            else:

                orientation = "axial"


            metadata.append({

                "Subject_ID":
                    subject_id,

                "Original_MRI":
                    img_path,

                "Image_Path":
                    image_path,

                "Image_Name":
                    image_name,

                "Orientation":
                    orientation,

                "Width":
                    IMAGE_SIZE[0],

                "Height":
                    IMAGE_SIZE[1]

            })


        successful += 1


    except Exception as e:

        failed += 1


        print(
            "\n" + "!" * 60
        )

        print(
            "FAILED FILE"
        )

        print(
            "!" * 60
        )

        print(
            "File:",
            img_path
        )

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        print(
            "!" * 60
        )


# ============================================================
# 15. CREATE METADATA
# ============================================================

metadata_df = pd.DataFrame(
    metadata
)


# ============================================================
# 16. SAVE METADATA
# ============================================================

metadata_path = os.path.join(
    OUTPUT_ROOT,
    "image_metadata.csv"
)


metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL OASIS CNN DATASET 10-12 CREATED")
print("=" * 70)


print(
    "Disc 10:",
    DATASET_ROOTS[0]
)

print(
    "Disc 11:",
    DATASET_ROOTS[1]
)

print(
    "Disc 12:",
    DATASET_ROOTS[2]
)


print(
    "\nTotal MRI files found:",
    len(mri_files)
)

print(
    "MRI files successfully processed:",
    successful
)

print(
    "MRI files failed:",
    failed
)

print(
    "Total PNG images generated:",
    len(metadata_df)
)


print(
    "\nImages per successful MRI:"
)

print(
    "7 sagittal + 7 coronal + 7 axial = 21"
)


print(
    "\nFinal images folder:"
)

print(
    IMAGE_ROOT
)


print(
    "\nMetadata file:"
)

print(
    metadata_path
)


print(
    "\n" + "=" * 70
)

print(
    "DATASET PREPARATION COMPLETED"
)

print(
    "=" * 70
)
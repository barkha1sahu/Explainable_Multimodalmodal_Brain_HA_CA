import os

DATASET_ROOTS = [

    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc7\disc7",

    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc8\disc8",

    r"C:\Users\rog\Downloads\oasis_cross-sectional_disc9\disc9"

]

MRI_SUFFIX = "_mpr-1_anon.img"

mri_files = []

for dataset_root in DATASET_ROOTS:

    print()
    print("Searching:", dataset_root)

    for root, dirs, files in os.walk(dataset_root):

        for file in files:

            if file.lower().endswith(MRI_SUFFIX):

                img_path = os.path.join(root, file)

                hdr_path = (
                    os.path.splitext(img_path)[0]
                    + ".hdr"
                )

                if os.path.exists(hdr_path):

                    mri_files.append(img_path)


mri_files = list(dict.fromkeys(mri_files))

print()
print("=" * 60)
print("SEARCH COMPLETED")
print("=" * 60)

print("Total MRI files found:", len(mri_files))

print()
print("First 10 MRI files:")

for file in mri_files[:10]:

    print(file)
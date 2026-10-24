import os

BASE = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET"

print("\nSearching for clinical prediction/result files...\n")

for root, dirs, files in os.walk(BASE):
    for file in files:
        name = file.lower()

        if (
            "clinical" in name
            or "prediction" in name
            or "result" in name
            or "prob" in name
            or "random" in name
            or "forest" in name
        ):
            print(os.path.join(root, file))

print("\nSearch complete.")
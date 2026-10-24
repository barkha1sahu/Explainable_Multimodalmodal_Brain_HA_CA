import pandas as pd

FILE = r"C:\Users\rog\OneDrive\Desktop\HA_DATASET\OASIS_CNN_Dataset_1_12\image_metadata_labelled.csv"

df = pd.read_csv(FILE)

print("Total images:", len(df))
print("Total subjects:", df["Subject_ID"].nunique())

print("\nCDR distribution:")
print(df["CDR"].value_counts(dropna=False).sort_index())

print("\nLabel distribution:")
print(df["Label"].value_counts(dropna=False).sort_index())

print("\nMissing labels:")
print(df["Label"].isna().sum())

print("\nSample data:")
print(df[["Subject_ID", "Image_Name", "CDR", "Label"]].head(20).to_string(index=False))
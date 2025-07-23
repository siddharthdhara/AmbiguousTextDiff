import os
import random
import shutil
import pandas as pd
from pathlib import Path

# Configuration
DATA_ROOT = "data/LIDC-IDRI-slices"  # Path to your LIDC-IDRI dataset
OUTPUT_ROOT = "data"  # Output directory for train/test split
CSV_PATH = "lidc_annotations_text_augmented.csv"  # Path to annotations CSV
TEST_RATIO = 0.2  # 20% for test set
RANDOM_SEED = 42  # For reproducibility

# Create output directories
train_dir = os.path.join(OUTPUT_ROOT, "training")
test_dir = os.path.join(OUTPUT_ROOT, "testing")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all patient folders
patient_folders = [f for f in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, f))]
patient_folders.sort()  # Sort to ensure reproducibility

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

# Shuffle and split
random.shuffle(patient_folders)
split_idx = int(len(patient_folders) * (1 - TEST_RATIO))
train_patients = patient_folders[:split_idx]
test_patients = patient_folders[split_idx:]

print(f"Total patients: {len(patient_folders)}")
print(f"Training set: {len(train_patients)} patients")
print(f"Test set: {len(test_patients)} patients")

# For training set
with open(os.path.join(train_dir, "train_list.txt"), "w") as f:
    for patient in train_patients:
        f.write(f"{patient}\n")
        # Actually copy the data
        src = os.path.join(DATA_ROOT, patient)
        dst = os.path.join(train_dir, patient)
        print(f"Copying {patient} to training set...")
        shutil.copytree(src, dst, dirs_exist_ok=True)

# For test set
with open(os.path.join(test_dir, "test_list.txt"), "w") as f:
    for patient in test_patients:
        f.write(f"{patient}\n")
        # Actually copy the data
        src = os.path.join(DATA_ROOT, patient)
        dst = os.path.join(test_dir, patient)
        print(f"Copying {patient} to test set...")
        shutil.copytree(src, dst, dirs_exist_ok=True)

# Split the annotations CSV file as well
df = pd.read_csv(CSV_PATH)
train_cases = [patient for patient in train_patients]
test_cases = [patient for patient in test_patients]

# Filter based on the file_name column
train_df = df[df['file_name'].isin(train_cases)]
test_df = df[df['file_name'].isin(test_cases)]

# Save the split annotation files
train_df.to_csv(os.path.join(train_dir, "lidc_annotations_text_augmented_train.csv"), index=False)
test_df.to_csv(os.path.join(test_dir, "lidc_annotations_text_augmented_test.csv"), index=False)

print("Dataset splitting complete!")
print(f"Training annotations: {len(train_df)} entries")
print(f"Test annotations: {len(test_df)} entries")
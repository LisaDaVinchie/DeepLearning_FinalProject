from zip_utils import unzip
from pathlib import Path
import sys

if len(sys.argv) < 3:
    print("Usage: python unzip_datasets.py <train_dataset_folder> <test_dataset_folder>")
    exit()
    
train_dataset_folder = Path(sys.argv[1])
test_dataset_folder = Path(sys.argv[2])

if not train_dataset_folder.exists() or not test_dataset_folder.exists():
    print("One or both of the specified folders do not exist.")
    exit()

train_files = list(train_dataset_folder.glob("*.zip"))
test_files = list(test_dataset_folder.glob("*.zip"))

if len(train_files) == 0:
    print(f"No train files found in {train_dataset_folder}.")
    exit()

if len(test_files) == 0:
    print(f"No test files found in {test_dataset_folder}.")
    exit()

print(f"Found {len(train_files)} train files and {len(test_files)} test files.")

print("Unzipping datasets...")

for file in train_files:
    unzip(file, train_dataset_folder)

for file in test_files:
    unzip(file, test_dataset_folder)
    
print("Datasets unzipped.")
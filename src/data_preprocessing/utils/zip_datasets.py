from zip_utils import zip_to_file
from pathlib import Path
import sys

if len(sys.argv) < 3:
    sys.exit("Usage: python zip_datasets.py <train_dataset_folder> <test_dataset_folder>")
    
train_dataset_folder = Path(sys.argv[1])
test_dataset_folder = Path(sys.argv[2])

if not train_dataset_folder.exists():
    print(f"Train dataset folder {train_dataset_folder} does not exist.")
    exit()
if not test_dataset_folder.exists():
    print(f"Test dataset folder {test_dataset_folder} does not exist.")
    exit()

train_files = list(train_dataset_folder.glob("*.pth"))
test_files = list(test_dataset_folder.glob("*.pth"))

if len(train_files) <= 0:
    print("Train dataset folder is empty.")
    exit()
if len(test_files) <= 0:
    print("Test dataset folder is empty.")
    exit()

print(f"Found {len(train_files)} train files and {len(test_files)} test files.")

print("Zipping datasets...")
for file in train_files:
    zip_to_file([file], file.with_suffix(".zip"))

for file in test_files:
    zip_to_file([file], file.with_suffix(".zip"))
    
print("Finished zipping datasets.")
import ConvertLandData as cld
from concurrent.futures import ThreadPoolExecutor
from utils.ZipFileUtilities import extract
import sys
from pathlib import Path
import os

if len(sys.argv) < 3:
    print("Usage: python run.py <download_folder> <converted_folder> <zip_folder>")
    exit()

download_folder = Path(sys.argv[1])
converted_files_folder = Path(sys.argv[2])
zip_files_folder = Path(sys.argv[3])
unzipped_files_folder = download_folder.parent / 'unzipped'

if not download_folder.exists():
    print(f"Folder {download_folder} does not exist.")
    exit()

unzipped_files_folder.mkdir(parents=True, exist_ok=True)
converted_files_folder.mkdir(parents=True, exist_ok=True)
    
zip_files = list(download_folder.glob('*.zip'))

if not zip_files:
    print(f"No zip files found in {download_folder}.")
    exit()

cld_util = cld.ConvertLandData(converted_files_folder)
overwrite = False

def process_zip_file(zip_file: Path):
    file_name = zip_file.stem
    print(f"Processing {file_name}...")
    
    unzipped_file_path = unzipped_files_folder / (file_name + '.csv')
    destination_path = converted_files_folder /  (file_name + '.csv')
    
    if destination_path.exists():
        print(f"File {file_name} already exists in {converted_files_folder}.")
        if overwrite:
            destination_path.unlink()
        else:
            print(f"Skipping {file_name}.")
            return
    
    extract(zip_file, unzipped_files_folder)
    
    cld_util.transform(unzipped_file_path)
    print("Done.")
    
    for ext in ['tiff', 'xml', 'csv']:
        (unzipped_files_folder / (file_name + '.' + ext)).unlink(missing_ok=True)
        
    print (f"Processed {file_name}.\n")

print(f"Processing {len(zip_files)} files...")

max_w = min(os.cpu_count(), len(zip_files))

with ThreadPoolExecutor(max_workers=max_w) as executor:
    executor.map(process_zip_file, zip_files)

if not any(unzipped_files_folder.iterdir()):
    unzipped_files_folder.rmdir()
    print(f"Deleted empty folder {unzipped_files_folder}.")
else:
    print(f"Folder {unzipped_files_folder} is not empty.")
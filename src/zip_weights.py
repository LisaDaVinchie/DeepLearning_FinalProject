from pathlib import Path
import argparse
import zipfile


parser = argparse.ArgumentParser()
parser.add_argument("--weights_folder", type=Path, required=True, help="Path to the folder containing the files to zip")

args = parser.parse_args()
    
weights_folder = Path(args.weights_folder)

if not weights_folder.exists():
    print(f"Path {weights_folder} does not exist")
    exit()
    
result_files = list(weights_folder.glob("*.pth"))

for file in result_files:
    print(f"Zipping {file}")
    with zipfile.ZipFile(file.with_suffix(".zip"), 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file, file.name)
    print(f"Zipped {file}")
    print("\n")
    

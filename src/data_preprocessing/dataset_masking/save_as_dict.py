import numpy as np
from pathlib import Path
import torch as th
import sys
from time import time

if len(sys.argv) < 3:
    raise ValueError("Usage: python save_as_dict.py <images_folder> <output_file>")

full_images_folder = Path(sys.argv[1])
masks_folder = Path(sys.argv[2])
dict_filename = Path(sys.argv[3])

# Check if the data folders exist
assert full_images_folder.exists(), "Images folder does not exist"
assert masks_folder.exists(), "Images folder does not exist"

# Get the names of the files in the folder
masks_paths = list(masks_folder.glob("*[0-9].npy"))

# Check if the files exist
assert len(masks_paths) > 0, "No image files found"

full_images_files_paths = [full_images_folder / f"{f.stem}.npy" for f in masks_paths]

for path in full_images_files_paths:
    assert path.exists(), f"File {path} does not exist"

# Create the dictionary
start_time = time()

dataset = {}

dataset['masks'] = [th.tensor(np.load(f)) for f in masks_paths]

names = [None] * len(dataset['masks'])
coords = [None] * len(dataset['masks'])
images = [None] * len(dataset['masks'])
masked_images = [None] * len(dataset['masks'])

for i in range(len(dataset['masks'])):
    data = np.load(full_images_files_paths[i])
    names[i] = full_images_files_paths[i].stem
    coords[i] = th.tensor(data[:2, :, :]).float()
    images[i] = th.tensor(data[2, :, :]).float()
    mask = np.load(masks_paths[i])
    masked_image = data[2, :, :].copy()
    masked_image[mask] = 10000
    
    masked_images[i] = th.tensor(masked_image).float()
    
dataset['names'] = names
dataset['coords'] = coords
dataset['images'] = images
dataset['masked_images'] = masked_images
    
# Create the folder to save the dictionary, if it does not exist
dict_folder = dict_filename.parent
dict_folder.mkdir(exist_ok=True, parents=True)

# Save the dictionary
th.save(dataset, dict_filename)

print(f"Dictionary saved to {dict_filename} in {time() - start_time:.2f} seconds")

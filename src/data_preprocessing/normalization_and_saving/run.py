import numpy as np
from pathlib import Path
import torch as th
import sys

if len(sys.argv) < 3:
    raise ValueError("Usage: python save_as_dict.py <images_folder> <masks_folder> <output_file>")

images_folder = Path(sys.argv[1])
masks_folder = Path(sys.argv[2])
output_file = Path(sys.argv[3])

# Check if the data folders exist
assert images_folder.exists(), "Images folder does not exist"
assert masks_folder.exists(), "Masks folder does not exist"

print("Folders exist")

# Get the names of the files in the folder
images_paths = list(images_folder.glob("*.npy"))
assert len(images_paths) > 0, "No image files found"

masks_paths = list(masks_folder.glob("*.npy"))
assert len(masks_paths) > 0, "No mask files found"

# Normalize the images

def normalize_image(image: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return (image - min_value) / (max_value - min_value)


print("Finding min and max values")
min_value = np.inf
max_value = -np.inf

for path in images_paths:
    data = np.load(path)
    depths = data[2, :, :].ravel()
    min_value = min(min_value, np.min(depths))
    max_value = max(max_value, np.max(depths))
    
print(f"Min value: {min_value}, Max value: {max_value}")
    

print("Normalizing images and saving as dictionary")
dataset = {}

dataset['masks'] = [th.tensor(np.load(f)).bool() for f in masks_paths]

names = [None] * len(dataset['masks'])
coords = [None] * len(dataset['masks'])
images = [None] * len(dataset['masks'])
masked_images = [None] * len(dataset['masks'])

for i in range(len(dataset['masks'])):
    data = np.load(images_paths[i])
    depths = data[2, :, :]
    names[i] = images_paths[i].stem
    coords[i] = th.tensor(data[:2, :, :]).float()
    norm_image = normalize_image(depths, min_value, max_value)
    
    images[i] = th.tensor(norm_image).float()
    masked_image = norm_image.copy()
    masked_image[dataset['masks'][i]] = 2.0
    masked_images[i] = th.tensor(masked_image).float()

dataset['names'] = names
dataset['coords'] = coords
dataset['images'] = images
dataset['masked_images'] = masked_images

# Create the folder to save the dictionary, if it does not exist
dict_folder = output_file.parent
dict_folder.mkdir(exist_ok=True, parents=True)

print(f"Saving dictionary to {output_file}")
# Save the dictionary
th.save(dataset, output_file)
print("Done")

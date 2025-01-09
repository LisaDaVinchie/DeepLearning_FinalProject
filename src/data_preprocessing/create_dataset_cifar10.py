import pickle
import numpy as np
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=Path, required=True, help="Path to the paths config file")
parser.add_argument("--params", type=Path, required=True, help="Path to the parameters config file")

args = parser.parse_args()

paths_file_path = args.paths
params_file_path = args.params

with open (paths_file_path, "r") as f:
    paths = json.load(f)

with open (params_file_path, "r") as f:
    params = json.load(f)

useful_keys = ["raw_data_folder", "dataset_path"]
    
for key in useful_keys:
    if key not in paths:
        exit(f"The key {key} was not found in the paths config file.")
    
raw_data_folder = Path(paths["raw_data_folder"])
destination_folder = Path(paths["dataset_path"])

if not raw_data_folder.exists():
    exit("The raw data folder does not exist.")
    

useful_keys = ["n_train", "n_test", "mask_percentage", "image_width", "image_height"]

for key in useful_keys:
    if key not in params:
        exit(f"The key {key} was not found in the parameters config file.")
    
n_train = int(params["n_train"])
n_test = int(params["n_test"])
mask_percentage = int(params["mask_percentage"])
image_width = int(params["image_width"])
image_height = int(params["image_height"])
n_pixels = int((mask_percentage / 100) * image_width * image_height)

# Get a list of the train images class folders

data_batches = list(raw_data_folder.glob("data_batch_*"))

if len(data_batches) <= 0:
    exit("The train images folder has no data batches.")
    
data_dict = unpickle(data_batches[0])

print(data_dict.keys())

print(data_dict[b'data'].shape)

plt.imshow(data_dict[b'data'][0].reshape(3, 32, 32).transpose(1, 2, 0))
plt.show()




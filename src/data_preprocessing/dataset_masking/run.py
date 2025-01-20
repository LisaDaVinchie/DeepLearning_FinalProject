import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion
import sys
from time import time

def mask_near_coastline(bathymetry_data: np.ndarray, distance: int) -> np.ndarray:
    """Create a nan mask near the coastline of the bathymetry data

    Args:
        bathymetry_data (np.ndarray): bathymetry data, with nans where there is land
        distance (int): distance from the coastline to mask

    Returns:
        np.ndarray: boolean mask of the same shape as the depths, with True where the mask is
    """
    nan_mask = np.isnan(bathymetry_data[2, :, :])
    
    nan_mask |= bathymetry_data[2, :, :] > 0

    border_mask = (binary_dilation(~nan_mask) ^ binary_erosion(~nan_mask)) & nan_mask

    structuring_element = np.ones((distance * 2 + 1, distance * 2 + 1), dtype=bool)
    expanded_mask = binary_dilation(border_mask, structure=structuring_element) & nan_mask
    
    return expanded_mask

DISTANCE = 5

if len(sys.argv) < 4:
    raise ValueError("Usage: python create_mask_near_coastline.py <bathymetry_data_folder> <interpolated_data_folder> <mask_folder>")

# Get the paths to the data folders

bath_data_folder = Path(sys.argv[1])
interpolated_data_folder = Path(sys.argv[2])
mask_folder = Path(sys.argv[3])

# Check if the data folders exist
assert bath_data_folder.exists(), "Bathymetry data folder does not exist"
assert interpolated_data_folder.exists(), "Interpolated data folder does not exist"

# Get a list of the interoplated data files
interpolated_data_file_paths = list(interpolated_data_folder.glob("*.npy"))

if len(interpolated_data_file_paths) == 0:
    raise ValueError("No interpolated data files found")

bathymetry_data_paths = [bath_data_folder / file_path.name for file_path in interpolated_data_file_paths]

for path in bathymetry_data_paths:
    assert path.exists(), f"File {path} does not exist"

print("Files exist")

mask_folder.mkdir(exist_ok=True, parents=True)
print("Destination folder created")

start_time = time()
print("Starting processing")
for bathymetry_data_path in bathymetry_data_paths:
    bathymetry_data = np.load(bathymetry_data_path)
    mask = mask_near_coastline(bathymetry_data, DISTANCE)
    
    if not np.any(mask):
        print(f"No mask for {bathymetry_data_path.stem}")
        continue
    
    np.save(mask_folder / bathymetry_data_path.name, mask)

print(f"Processing took {time() - start_time} seconds for {len(interpolated_data_file_paths)} files")
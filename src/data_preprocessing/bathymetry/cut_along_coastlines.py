import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from multiprocessing import shared_memory, Pool
from pathlib import Path
import sys
import json
from utils.get_workers_number import get_available_cpus


# Worker function for slicing images
def transform_using_shared_memory(args):
    shm_name, shape, dtype, point, half_x, half_y, features_to_keep = args
    
    # Access the shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    dataset = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    # Perform slicing
    start_row = point[0] - half_x
    end_row = point[0] + half_x
    start_col = point[1] - half_y
    end_col = point[1] + half_y
    cut_image = dataset[features_to_keep, start_row:end_row, start_col:end_col]
    
    return cut_image


# Main function
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python cut_along_coastlines.py <merged_file> <destination_folder> <config_file>")
        sys.exit(1)
    
    merged_file_path = Path(sys.argv[1])
    assert merged_file_path.exists(), "The merged file does not exist"

    destination_folder = Path(sys.argv[2])

    config_file = Path(sys.argv[3])
    assert config_file.exists(), "The config file does not exist"

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    assert 'image_width' in config, "The config file must contain the 'image_width' key"
    assert 'image_height' in config, "The config file must contain the 'image_height' key"
    assert 'n_images' in config, "The config file must contain the 'n_images' key"

    image_width = int(config['image_width'])
    image_height = int(config['image_height'])
    n_images = int(config['n_images'])
    
    n_cpus = get_available_cpus()
    print("Available CPUs:", n_cpus)

    half_x = image_width // 2
    half_y = image_height // 2
    print("Hyperparameters loaded")

    # Load the dataset
    # merged_dataset = np.load(merged_file_path)
    merged_dataset = np.memmap(merged_file_path, dtype='float32', mode='r', shape=(3, 18008, 18968))
    assert merged_dataset.shape[0] == 3, "The dataset must have 3 layers"
    print("Dataset loaded")

    # Store the dataset in shared memory
    shm = shared_memory.SharedMemory(create=True, size=merged_dataset.nbytes)
    shared_dataset = np.ndarray(merged_dataset.shape, dtype=merged_dataset.dtype, buffer=shm.buf)
    np.copyto(shared_dataset, merged_dataset)
    print("Dataset loaded into shared memory")

    # Find the coastline
    nan_mask = np.isnan(merged_dataset[2, :, :])
    edges = binary_dilation(nan_mask) ^ binary_erosion(nan_mask)
    edges_idxs = np.argwhere(edges)
    print("Coastline found")

    # Select random points
    filtered_edges = np.array([point for point in edges_idxs if point[0] > half_x
                                and point[0] < merged_dataset.shape[1] - half_x
                                and point[1] > half_y
                                and point[1] < merged_dataset.shape[2] - half_y],
                               dtype=int)

    points_idxs = np.random.choice(filtered_edges.shape[0], n_images, replace=False)
    random_points = filtered_edges[points_idxs]
    print("Random points selected")

    # Prepare arguments for the worker processes
    args = [
        (shm.name, merged_dataset.shape, merged_dataset.dtype, point, half_x, half_y, [0, 1, 2])
        for point in random_points
    ]

    # Create destination folder
    destination_folder.mkdir(parents=True, exist_ok=True)

    # Cut and save the images
    print("Starting the cutting process")
    with Pool(processes=n_cpus) as pool:
        results = pool.map(transform_using_shared_memory, args)
        for i, result in enumerate(results):
            np.save(destination_folder / f"image_{i:04d}.npy", result)
            print(f"Image {i + 1}/{n_images} saved")

    # Clean up shared memory
    shm.close()
    shm.unlink()
    print("Shared memory cleaned up")

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor
from RegridLandData import RegridLandData
#load_all_land_data, find_land_coord_boundaries
from utils.get_workers_number import get_available_cpus

def process_file(file_path: Path, regridder: RegridLandData, filled_data_folder: Path, unprocessable_files_file: Path) -> None:
    """Helper function to process a single file and save the interpolated data

    Args:
        file_path (Path): path to the bathymetry file to interpolate
        regridder (RegridLandData): the regridder object to use for interpolation
        filled_data_folder (Path): the folder to save the interpolated data to

    Raises:
        ValueError: returns an error if the file could not be processed
    """
    print("Processing file: ", file_path.name, flush=True)
    regridder.reset()
    bath_file = np.load(file_path)
    try:
        new_data = regridder.transform(bath_file)
    except Exception as e:
        with open(unprocessable_files_file, "a") as f:
            f.write(str(file_path.name) + "\n")
        
        raise ValueError(f"Error processing file {file_path.name}: {e}")
    
    new_file_path = filled_data_folder / file_path.name
    np.save(new_file_path, new_data)
    print(f"File saved to {new_file_path}\n", flush=True)


def check_for_already_processed_files(filled_data_folder: Path, bath_file_paths: list, unprocessable_files_file: Path) -> list:
    """Check for already processed files and return a list of files to process. Exit if no files to process are found.

    Args:
        bath_data_folder (Path): folder containing the bathymetry data
        filled_data_folder (Path): folder containing the interpolated data
        bath_file_paths (list): paths to all bathymetry data files
        unprocessable_files_file (Path): _description_

    Returns:
        list: list with the names of the files to process
    """
    already_processed_files = list(filled_data_folder.glob("*.npy"))
    if unprocessable_files_file.exists():
        with open(unprocessable_files_file, "r") as f:
            unprocessable_files = [Path(line.strip()) for line in f.readlines()]
        already_processed_files.extend(unprocessable_files)
    
    already_processed_files = [file_path.name for file_path in already_processed_files]
    
    print(f"Found {len(already_processed_files)} already processed files")
    
    files_to_process = [file_path for file_path in bath_file_paths if Path(file_path).name not in already_processed_files]
    
    if len(files_to_process) == 0:
        print("No files to process")
        sys.exit(0)
    return files_to_process

if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 6:
        raise ValueError("Usage: python run.py <bath_data_folder> <land_data_folder> <land_coord_boundaries_file> <filled_data_folder> <config_file>")

    # Get the paths to the data folders
    bath_data_folder = Path(sys.argv[1])
    land_data_folder = Path(sys.argv[2])
    land_coord_boundaries_file_path = Path(sys.argv[3])
    filled_data_folder = Path(sys.argv[4])
    config_file = Path(sys.argv[5])
    
    # Check if the data folders exist
    assert bath_data_folder.exists(), "Bathymetry data folder does not exist"
    assert land_data_folder.exists(), "Land data folder does not exist"
    
    # Load the config file
    with open(config_file, "r") as f:
        config = json.load(f)
        
    assert "extra_space" in config, "Extra space not found in config file"
    EXTRA_SPACE = config["extra_space"]
    
    # Create a file to store the names of the unprocessable files
    unprocessable_files_filename = "unprocessable_files.txt"
    unprocessable_files_path = filled_data_folder / unprocessable_files_filename
    n_cpus = get_available_cpus()
    
    # Get a list of all the bathymetry and land data files
    print("Reading data files")
    bath_file_paths = list(bath_data_folder.glob("*.npy"))
    assert len(bath_file_paths) > 0, "No bathymetry files found"
    land_files_paths = list(land_data_folder.glob("*.csv"))
    assert len(land_files_paths) > 0, "No land files found"
    print("Data files read")
    
    print("Checking for already processed files")
    # Check for already processed files and get a list of files to process
    files_to_process = check_for_already_processed_files(filled_data_folder, bath_file_paths, unprocessable_files_path)
    
    print("Loading land data dictionary")
    # Load all land data files into a dict with the file name as the key
    # land_data_cache = load_all_land_data(land_files_paths)
    
    # Load the land coordinate boundaries if the file exists, otherwise calculate them
    if land_coord_boundaries_file_path.exists():
        land_coord_boundaries = pd.read_csv(land_coord_boundaries_file_path)
    else:
        print("Land coord boundaries not found")
        exit()
    #     land_coord_boundaries = find_land_coord_boundaries(land_data_cache)
    #     land_coord_boundaries.to_csv(land_coord_boundaries_file_path, index=False)

    # Create the filled data folder if it does not exist
    filled_data_folder.mkdir(parents=True, exist_ok=True)
    
    # Create the regridder object
    regridder = RegridLandData(land_files_paths, land_coord_boundaries, extra_space = EXTRA_SPACE)
    
    # Create a list of arguments for the process_file function
    args = [(file_path, regridder, filled_data_folder, unprocessable_files_path) for file_path in files_to_process]
    
    process_file(*args[0])
    
    # print(f"Starting processing of {len(args)} files")
    # Declare max number of workers
    # max_w = min(n_cpus, len(bath_file_paths))
    # with ProcessPoolExecutor(max_workers = max_w) as executor:
    #     futures = {executor.submit(process_file, *arg) for arg in args}
    #     for future in futures:
    #         print("", flush=True)
    #         try:
    #             result = future.result()
    #         except KeyboardInterrupt:
    #             print("Keyboard interrupt")
    #             print("Time taken: ", time.time() - start_time)
    #             sys.exit(0)
    #         except Exception as e:
    #             print("Error: ", e)
                
    print("Time taken: ", time.time() - start_time)
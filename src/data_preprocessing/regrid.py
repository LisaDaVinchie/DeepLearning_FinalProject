import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import time
import argparse
from utils.RegridLandData import RegridLandData
from concurrent.futures import ThreadPoolExecutor

parent_path = Path(__file__).resolve().parents[2]
print("Parent path: ", parent_path)
sys.path.append(str(parent_path))
from src.utils.get_workers_number import get_available_cpus


def process_file(file_path: Path, regridder: RegridLandData, filled_data_folder: Path, unprocessable_files_file: Path) -> None:
    """Helper function to process a single file and save the interpolated data

    Args:
        file_path (Path): path to the bathymetry file to interpolate
        regridder (RegridLandData): the regridder object to use for interpolation
        filled_data_folder (Path): the folder to save the interpolated data to

    Raises:
        ValueError: returns an error if the file could not be processed
    """
    print("Processing file: ", file_path.name, "\n", flush=True)
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
    print(f"File saved to {new_file_path}\n\n", flush=True)


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
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True, help="Path to the paths config file")
    parser.add_argument("--params", type=Path, required=True, help="Path to the parameters config file")

    args = parser.parse_args()
    
    paths_config_path = args.paths
    params_config_file = args.params
    
    with open(paths_config_path, "r") as f:
        paths = json.load(f)

    # Get the paths to the data folders
    bath_data_folder = Path(paths["bathymetry"]["base"])
    land_data_folder = Path(paths["land"]["base"])
    land_coord_boundaries_file_path = Path(paths["land"]["coord_boundaries"])
    filled_data_folder = Path(paths["interpolated"]["base"])
    
    if not bath_data_folder.exists():
        print(f"Bathymetry data folder {bath_data_folder} does not exist")
        exit()
    if not land_data_folder.exists():
        print(f"Land data folder {land_data_folder} does not exist")
        exit()
        
    print("Paths loaded\n")
    
    # Load the config file
    with open(params_config_file, "r") as f:
        config = json.load(f)
        
    EXTRA_SPACE = config["dataset_params"]["extra_space"]
    n_train = config["dataset_params"]["n_train"]
    n_test = config["dataset_params"]["n_test"]
    
    print("Config loaded\n")
    
    # Create a file to store the names of the unprocessable files
    unprocessable_files_filename = "unprocessable_files.txt"
    unprocessable_files_path = filled_data_folder / unprocessable_files_filename
    n_cpus = get_available_cpus()
    
    # Get a list of all the bathymetry and land data files
    print("Reading data files")
    bath_file_paths = list(bath_data_folder.glob("*.npy"))
    
    if len(bath_file_paths) == 0:
        raise ValueError("No bathymetry files found")
    
    land_files_paths = list(land_data_folder.glob("*.csv"))
    
    if len(land_files_paths) == 0:
        raise ValueError("No land files found")
    print("Data files read")
    
    print("Checking for already processed files")
    # Check for already processed files and get a list of files to process
    files_to_process = check_for_already_processed_files(filled_data_folder, bath_file_paths, unprocessable_files_path)
    
    print("Loading land data dictionary")
    
    # Load the land coordinate boundaries if the file exists, otherwise calculate them
    if land_coord_boundaries_file_path.exists():
        land_coord_boundaries = pd.read_csv(land_coord_boundaries_file_path)
    else:
        raise ValueError("Land coordinate boundaries file not found")

    # Create the filled data folder if it does not exist
    filled_data_folder.mkdir(parents=True, exist_ok=True)
    
    # Create the regridder object
    regridder = RegridLandData(land_files_paths, land_coord_boundaries, extra_space = EXTRA_SPACE)
    
    # Create a list of arguments for the process_file function
    args = [(file_path, regridder, filled_data_folder, unprocessable_files_path) for file_path in files_to_process]
    
    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        executor.map(lambda x: process_file(*x), args)
                
    print("Time taken: ", time.time() - start_time)
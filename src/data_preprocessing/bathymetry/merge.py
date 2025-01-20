from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import os

def fill_with_file(row: int, col: int, files_list: list, result_array: np.ndarray, merged_dataset: np.ndarray, data_rows: int, data_columns: int):
    
    file_key = result_array[row][col]
    file_path = next(file for file in files_list if file_key in file.stem)
    
    data = np.load(file_path)
    
    startrow = row * data_rows
    endrow = (row + 1) * data_rows
    startcol = col * data_columns
    endcol = (col + 1) * data_columns
    
    merged_dataset[:, startrow:endrow, startcol:endcol] = data

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python join_bathymetry_data.py <npy_folder> <output_file>")
        sys.exit(1)

    npy_folder = Path(sys.argv[1])
    assert npy_folder.exists(), "The folder does not exist"

    output_file_path = Path(sys.argv[2])
    assert output_file_path.parent.exists(), "The parent folder of the output file does not exist"

    files_list = list(npy_folder.glob('*.npy'))
    assert files_list, "No files found in the folder"

    # Sort files first by the letter and then by the number
    files_basenames = [file.stem.split("_")[0] for file in files_list]
    files_endings = [file.stem.split("_")[1] for file in files_list]
    files_sorted = sorted(files_basenames, key=lambda x: (x[0], int(x[1:])), reverse=True)

    # Group the files by letter and number
    grouped_files = []
    for _, group in groupby(files_sorted, key=lambda x: x[0]):
        # Sort each group by number
        grouped_files.append(sorted(list(group), key=lambda x: x[1]))

    result_array = np.array(grouped_files)

    # Create result file
    n_rows = result_array.shape[0]
    n_columns = result_array.shape[1]

    data = np.load(files_list[0])
    data_layers = data.shape[0]
    data_rows = data.shape[1]
    data_columns = data.shape[2]

    merged_dataset = np.ones((data_layers, data_rows * n_rows, data_columns * n_columns), dtype=np.float32)
        

    args = [(row, col, files_list, result_array, merged_dataset, data_rows, data_columns) for row in range(n_rows) for col in range(n_columns)]

    max_w = min(len(args), os.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_w) as executor:
        futures = [executor.submit(fill_with_file, *arg) for arg in args]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                executor.shutdown(wait=False, cancel_futures=True)
                raise e
            
    np.save(output_file_path, merged_dataset)
    
    plt.imshow(merged_dataset[2, :, :])
    plt.colorbar()
    plt.savefig(output_file_path.with_suffix(".png"))
    plt.close()
from pathlib import Path
import xarray as xr
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor

def netcdf_to_numpy(netcdf_files_paths: Path, npy_files_folder: Path, lon_column: str = 'lon', lat_column: str = 'lat', elevation_column: str = 'elevation'):
    file_name = netcdf_files_paths.stem
    
    try:
        dataset = xr.open_dataset(netcdf_files_paths)
    except Exception as e:
        raise RuntimeError("Error opening the netcdf file: ", e)
    
    file_columns = np.array([var_name for var_name in dataset.variables.keys()])
    
    required_columns = {lon_column, lat_column, elevation_column}
    if not required_columns.issubset(file_columns):
        missing_columns = required_columns - set(file_columns)
        raise ValueError(f"Missing required columns in the netcdf file: {missing_columns}")
    
    lons = dataset[lon_column].values
    lats = dataset[lat_column].values

    # Directly use broadcasting to create the meshgrid of coordinates
    new_lons, new_lats = np.meshgrid(lons, lats)
    
    shapes = new_lons.shape
    
    transformed_data = np.empty((3, shapes[0], shapes[1]), dtype=np.float32)
    transformed_data[0, :, :] = new_lons
    transformed_data[1, :, :] = new_lats
    transformed_data[2, :, :] = dataset[elevation_column].values
    
    npy_file_path = npy_files_folder / f"{file_name}.npy"
    
    np.save(npy_file_path, transformed_data)

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise ValueError("Usage is python netcdf_to_numpy.py <netcdf_files_folder> <npy_files_folder>")

    netcdf_files_fodler = Path(sys.argv[1])
    assert netcdf_files_fodler.exists(), "The netcdf folder does not exist"

    npy_files_folder = Path(sys.argv[2])

    if not npy_files_folder.exists():
        npy_files_folder.mkdir(parents=True, exist_ok=True)

    netcdf_files_list = list(netcdf_files_fodler.glob('*.nc'))
        
    args = [(netcdf_file, npy_files_folder) for netcdf_file in netcdf_files_list]

    with ProcessPoolExecutor() as executor:
        executor.map(netcdf_to_numpy, *zip(*args))
        


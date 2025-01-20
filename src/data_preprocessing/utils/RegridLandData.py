import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator

class RegridLandData:
    def __init__(self, land_files_paths: list, land_coord_boundaries: pd.DataFrame, extra_space):
        """Class ti interpolate missing bathymetry data using land data

        Args:
            land_data_cache (dict): dictionary containing the land data with the file name as the key
            land_coord_boundaries (pd.DataFrame): dataframe containing the boundaries of each land data file
            extra_space (float, optional): extra space to add to the bathymetry data boundaries, in degrees. Defaults to 0.5.
        """
        
        self.land_files_paths = land_files_paths
        self.land_coord_boundaries = land_coord_boundaries
        self.extra_space = np.float32(extra_space)
        self.land_data_cache = {}
        
    def reset(self):
        self.land_data_cache = {}
        
    def _get_land_data(self, file_path: Path):
        """Loads a land file if not already cached
        """
        if file_path.name not in self.land_data_cache:
            self.land_data_cache[file_path.name] = pd.read_csv(file_path)
            
        return self.land_data_cache[file_path.name].values
    
    def _is_land_data_inside_bath_boundaries(self, bath_boundaries: list, land_boundaries: list) -> bool:
        """Check if any part of the land data is inside the bathymetry data boundaries

        Args:
            bath_boundaries (list): the boundaries of the missing bathymetry data, as [min_lon, max_lon, min_lat, max_lat]
            land_boundaries (list): the boundaries of the land data, as [min_lon, max_lon, min_lat, max_lat]
            
        Returns:
            bool: True if any part of the land data is inside the bathymetry data, False otherwise
        """
        
        # Unpack the bath boundaries and add extra space to them
        min_lon_b, max_lon_b, min_lat_b, max_lat_b = bath_boundaries
        min_lon_l, max_lon_l, min_lat_l, max_lat_l = land_boundaries
        
        # Check if any part of the land data is inside the bathymetry data
        lon_condition_part = ((max_lon_l >= min_lon_b) & (max_lon_l <= max_lon_b)) | ((min_lon_l >= min_lon_b) & (min_lon_l <= max_lon_b))
        lat_condition_part = ((max_lat_l >= min_lat_b) & (max_lat_l <= max_lat_b)) | ((min_lat_l >= min_lat_b) & (min_lat_l <= max_lat_b))
        
        # Check if the bathymetry data is fully inside the land data
        lon_condition_full = (min_lon_l <= min_lon_b) & (max_lon_l >= max_lon_b)
        lat_condition_full = (min_lat_l <= min_lat_b) & (max_lat_l >= max_lat_b)
        
        # Combine the conditions for partial and full overlap
        lon_condition = lon_condition_part | lon_condition_full
        lat_condition = lat_condition_part | lat_condition_full
        
        # Check that both the conditions are satisfied
        return lon_condition & lat_condition
    
    def _get_useful_land_files(self, bath_boundaries: list) -> list:
        
        min_lon_b = bath_boundaries[0] - self.extra_space
        max_lon_b = bath_boundaries[1] + self.extra_space
        min_lat_b = bath_boundaries[2] - self.extra_space
        max_lat_b = bath_boundaries[3] + self.extra_space
        
        relevant_files = []
        
        for idx, item in self.land_coord_boundaries.iterrows():
            file_name = item["land_file"]
            land_bounds = item[["min_lon", "max_lon", "min_lat", "max_lat"]]
            if self._is_land_data_inside_bath_boundaries([min_lon_b, min_lat_b, max_lon_b, max_lat_b], land_bounds):
                relevant_files.append(file_name)
                
        return relevant_files
    
    def _filter_land_coords(self, land_data: np.ndarray, bath_boundaries: list) -> np.ndarray:
        """Filter the land data to only include the coordinates that are useful for interpolation

        Args:
            land_data (np.ndarray): the land data to filter, as a 2D array with the longitudes in the first column and the latitudes in the second column
            bath_boundaries (list): the boundaries of the missing bathymetry data, as [min_lon, max_lon, min_lat, max_lat]

        Returns:
            np.ndarray: the filtered land data, as a 2D array with the longitudes in the first column and the latitudes in the second column
        """
        
        # Unpack the bath boundaries and add extra space to them
        min_lon_b, max_lon_b, min_lat_b, max_lat_b = bath_boundaries
        
        # Filter the land data to only include the coordinates that are useful for interpolation
        lon_condition = (land_data[:, 0] >= min_lon_b) & (land_data[:, 0] <= max_lon_b)
        lat_condition = (land_data[:, 1] >= min_lat_b) & (land_data[:, 1] <= max_lat_b)
        
        useful_land_data = land_data[lon_condition & lat_condition]
        
        return useful_land_data[:, :2], useful_land_data[:, 2]
        
    
    def transform(self, bath_data: np.ndarray) -> np.ndarray:
        """Interpolate the missing values in the bathymetry data, using the land data.

        Args:
            bath_data (np.ndarray): the bathymetry data to interpolate, as a 3D array with [0, :, :] as the longitudes, [1, :, :] as the latitudes, and [2, :, :] as the depths.

        Raises:
            UserWarning: data useful for interpolation could not be found
            ValueError: interpolation failed
            ValueError: NaN values found in the interpolated depths

        Returns:
            np.ndarray: the bathymetry data with the missing values filled, as a 3d array with the same structure as the input.
        """
        
        # Copy the bathimetry data to avoid modifying the original
        self.bath_data = bath_data
        
        # Flatten the bath data and separate it into coordinates and depths
        bath_coords = np.stack([self.bath_data[0, :, :].ravel(), self.bath_data[1, :, :].ravel()], axis=1)
        depths = self.bath_data[2, :, :].ravel()
        
        # Find the missing values
        nan_mask  = np.isnan(depths)
        
        # Get the coordinates of the missing values
        self.missing_coords = bath_coords[nan_mask]
        min_lon, max_lon = np.min(self.missing_coords[:, 0]), np.max(self.missing_coords[:, 0])
        min_lat, max_lat = np.min(self.missing_coords[:, 1]), np.max(self.missing_coords[:, 1])
        bath_boundaries = [min_lon, max_lon, min_lat, max_lat]
        
        # Get the land files that are useful for interpolation
        useful_land_files_names = self._get_useful_land_files(bath_boundaries)
        useful_land_files_paths =  [file_path for file_path in self.land_files_paths if Path(file_path).name in useful_land_files_names]
        land_data = np.concatenate([self._get_land_data(file_path) for file_path in useful_land_files_paths])
        
        # Find the land coordinates and heights that are useful for interpolation
        useful_land_coords, useful_land_heights = self._filter_land_coords(land_data, bath_boundaries)
        
        # useful_land_coords, useful_land_heights = land_data[:, :2], land_data[:, 2]
        
        # Interpolate the missing valuess
        useful_coords = np.concatenate([bath_coords[~nan_mask], useful_land_coords])
        useful_heights = np.concatenate([depths[~nan_mask], useful_land_heights])
        
        try:
            interpolator = LinearNDInterpolator(useful_coords, useful_heights)
            new_depths = interpolator(self.missing_coords)
        except Exception as e:
            raise ValueError("Interpolation failed: ", e)
        
        if np.any(np.isnan(new_depths)):
            raise ValueError("NaN values found in the interpolated depths")
        
        # Replace the missing values with the interpolated values
        depths[nan_mask] = new_depths
        
        # Reshape the depths array to the bathymetry data shape
        depths = depths.reshape(self.bath_data[0].shape)
        
        # Return the bathymetry data with the missing values filled
        return np.array([self.bath_data[0], self.bath_data[1], depths])
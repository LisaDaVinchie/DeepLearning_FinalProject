import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import binary_erosion, binary_dilation

class InterpolateLandData:
    def __init__(self, land_data_cache: dict, land_coord_boundaries: pd.DataFrame, distance: int = 5):
        """Given a specific image and all the land files, interpolate the useful missing land cooordinates for inpainting

        Args:
            land_file_paths (list): List of paths to all the land data files
            land_coord_boundaries_file_path (Path): Path to the file containing the land coord boundaries for each file

        Raises:
            ValueError: _description_
        """
        
        self.distance = distance
        self.land_data_cache = land_data_cache
        self.land_coord_boundaries = land_coord_boundaries
    
    def _find_land_coord_boundaries(self, land_data_cache: dict) -> pd.DataFrame:
        """Find the boundaries of the land coordinates for each land data file

        Args:
            land_data_cache (dict): dictionary containing the land data with the file name as the key

        Returns:
            pd.DataFrame: a dataframe containing the boundaries of the land coordinates for each land data file
        """
        coord_bounds = [
            {
                "land_file": file_name,
                "min_lon": data["longitude"].min(),
                "max_lon": data["longitude"].max(),
                "min_lat": data["latitude"].min(),
                "max_lat": data["latitude"].max()
            }
            for file_name, data in land_data_cache.items()]
        
        return pd.DataFrame(coord_bounds)
    
    def _find_inpainting_mask(self, depths: np.ndarray) -> np.ndarray:
        """Find the edges of the coastline in the bathymetry data, i.e. the points where we pass from naa to not nan.

        Args:
            depths (np.ndarray): the bathymetry data as a 2d array
            
        Returns:
            np.ndarray: a boolean 2d array containing the edges of the coastline
        """

        # Find the points where the bathymetry data is nan
        # Gives a boolean array where True indicates a nan point
        nan_mask = np.isnan(depths)

        # Use binary erosion and dilation to identify boundary changes (included image borders)
        # Gives a boolean array where True indicates a boundary point
        edges_mask = (binary_dilation(~nan_mask) ^ binary_erosion(~nan_mask))
        
        structuring_element = np.ones((self.distance * 2 + 1, self.distance * 2 + 1), dtype=bool)
        expanded_mask = binary_dilation(edges_mask, structure=structuring_element)

        # Expand the mask of the edges of the coastline and keep only the missing values
        return expanded_mask & nan_mask
    
    def _find_useful_land_coords(self, inpainting_bath_coords: np.ndarray) -> np.ndarray:
        """Given the coordinates of the bathymetry data to inpaint, find the land coordinates that are useful for interpolation. Consider an extra distance around the bathymetry data to inpaint.

        Args:
            inpainting_bath_coords (np.ndarray): the coordinates of the bathymetry data to inpaint, as a 2d array [lon, lat]

        Returns:
            np.ndarray: the land coordinates and heights that are within the extra distance around the bathymetry data to inpaint
        """
        
        min_lon = np.min(inpainting_bath_coords[:, 0]) - self.distance
        max_lon = np.max(inpainting_bath_coords[:, 0]) + self.distance
        
        min_lat = np.min(inpainting_bath_coords[:, 1]) - self.distance
        max_lat = np.max(inpainting_bath_coords[:, 1]) + self.distance
        
        useful_land_data = []
        for file_name, land_data in self.land_data_cache.items():
            
            coords = land_data[["longitude", "latitude"]].values
            
            useful_idxs = np.argwhere((coords[:, 0] >= min_lon) & (coords[:, 0] <= max_lon) & (coords[:, 1] >= min_lat) & (coords[:, 1] <= max_lat))

            useful_land_data.append(land_data.iloc[useful_idxs])
            
        data = pd.concat(useful_land_data)
            
        return data[['longitude', 'latitude']], data['height']
            
            
    def transform(self, bath_data: np.ndarray) -> np.ndarray:
        """Fills the missing values in the bathymetry data file with land data

        Args:
            bath_data (np.ndarray): bathymetry data file that we want to fill with land data

        Raises:
            ValueError: The interpolation failed
            ValueError: There are still missing values after interpolation

        Returns:
            np.ndarray: 3d array containing the bathymetry data with the missing values filled
        """
        
        # Flatten the bath data and separate it into coordinates and depths
        depths = bath_data[2, :, :]
        coords = bath_data[0:2, :, :]
        print("Bath data flattened", flush=True)
        
        inpainting_mask = self._find_inpainting_mask(depths)
        
        print("Inpainting mask found", flush=True)
        
        # Get the coordinates that need inpainting
        inpainting_idxs = np.argwhere(inpainting_mask)
        
        print("Inpainting indices shape", inpainting_idxs.shape)

        inpainting_cords = coords[inpainting_idxs[:, 0], inpainting_idxs[:, 1]]
    
        # Find the land coordinates and heights that are useful for interpolation
        try:
            filtered_land_coords, filtered_land_heights = self._find_useful_land_coords(inpainting_cords)
        except Exception as e:
            print(e)
            raise ValueError("Land data does not cover the missing data")
        print("Useful land data found", flush=True)
        
        try:
            interpolator = LinearNDInterpolator(filtered_land_coords, filtered_land_heights)
            new_heights = interpolator(inpainting_cords)
        except Exception as e:
            print(e)
            raise ValueError("Interpolation failed")
        print("Interpolation successful", flush=True)
        
        return inpainting_cords, new_heights
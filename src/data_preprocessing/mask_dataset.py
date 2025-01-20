import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion

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
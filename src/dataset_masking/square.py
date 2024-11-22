import numpy as np
import torch as th

def create_square_mask(image: th.tensor, n_pixels: int, mask_value: int = 2) -> th.tensor:
    """Create a square mask of n_pixels in the image

    Args:
        image (th.tensor): 3d tensor of shape (n_channels, height, width)
        n_pixels (int): number of pixels in the square mask

    Returns:
        th.tensor: 3d tensor of shape (n_channels, height, width) with the square mask
    """
    mask = np.zeros((image.shape[1], image.shape[2]), dtype=bool)
    
    square_width = int(np.sqrt(n_pixels))
    half_square_width = square_width // 2
    
    center = (np.random.randint(half_square_width, image.shape[1] - half_square_width), np.random.randint(half_square_width, image.shape[2] - half_square_width))
    
    mask[center[0] - half_square_width:center[0] + half_square_width, center[1] - half_square_width:center[1] + half_square_width] = True
    
    new_image = image.clone()
    new_image[:, mask] = mask_value
    
    return new_image
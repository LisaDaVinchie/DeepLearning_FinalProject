import numpy as np

class SquareMask:
    def __init__(self, image_width: int, image_height: int, n_pixels: int):
        self.n_pixels = n_pixels
        self.image_width = image_width
        self.image_height = image_height
        square_width = int(np.sqrt(n_pixels))
        self.half_square_width = square_width // 2
        
    

    def create_square_mask(self) -> np.ndarray:
        """Create a square mask of n_pixels in the image

        Args:
            image_width (int): number of columns in the image
            image_height (int): number of rows in the image
            n_pixels (int): number of pixels in the square mask

        Returns:
            np.ndarray: 2d boolean array with the square mask. True values are the pixels that are part of the square mask.
        """
        mask = np.zeros((self.image_width, self.image_height), dtype=bool)
        
        center_row = np.random.randint(self.half_square_width, self.image_height - self.half_square_width)
        center_col = np.random.randint(self.half_square_width, self.image_width - self.half_square_width)
        
        start_row = center_row - self.half_square_width
        end_row = center_row + self.half_square_width
        start_col = center_col - self.half_square_width
        end_col = center_col + self.half_square_width
        
        mask[start_row:end_row, start_col:end_col] = True
        
        return mask
import numpy as np
import cv2

class SquareMask:
    def __init__(self, image_width: int, image_height: int, n_pixels: int):
        self.n_pixels = n_pixels
        self.image_width = image_width
        self.image_height = image_height
        square_width = int(np.sqrt(n_pixels))
        self.half_square_width = square_width // 2
        
    def create_mask(self) -> np.ndarray:
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

class LineMask:
    def __init__(self, n_channels: int, image_width: int, image_height: int, max_tichkness: int = 3, n_lines: int = 5):
        self.n_channels = n_channels
        self.image_width = image_width
        self.image_height = image_height
        self.max_tichkness = max_tichkness
        self.n_lines = n_lines

    def create_mask(self) -> np.ndarray:
        ## Prepare masking matrix
        
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        for _ in range(np.random.randint(1, self.n_lines)):
            # Get random x locations to start line
            x1, x2 = np.random.randint(1, self.image_width), np.random.randint(1, self.image_width)
            # Get random y locations to start line
            y1, y2 = np.random.randint(1, self.image_height), np.random.randint(1, self.image_height)
            # Get random thickness of the line drawn
            thickness = np.random.randint(1, self.max_tichkness)
            # Draw black line on the white mask
            cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),thickness)

        return mask.astype(bool)
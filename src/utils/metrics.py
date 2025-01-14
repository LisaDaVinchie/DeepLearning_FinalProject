import torch as th
from torchmetrics.functional import structural_similarity_index_measure as ssim

def calculate_psnr(original: th.tensor, reconstructed: th.tensor, max_pixel: float = 1.0):
    mse = th.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 10 * th.log10(max_pixel ** 2 / mse)
    return psnr

def calculate_ssim(original: th.tensor, reconstructed: th.tensor):
    score = ssim(original, reconstructed)  # For RGB images
    return score

def dice_coef(y_true: th.tensor, y_pred: th.tensor):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    intersection = (y_true * y_pred).sum()
    
    return (2. * intersection) / (y_true + y_pred).sum()
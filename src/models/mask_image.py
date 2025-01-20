import torch as th

def mask_image(image: th.tensor, mask: th.tensor, placeholder = None) -> th.tensor:
    
    if placeholder is None:
        placeholder = th.mean(image, dim=(2, 3), keepdim=True).detach()
    # print(f"Image shape: {image.shape}, mask shape: {mask.shape}, placeholder shape: {placeholder.shape}")
        
    masked_image = th.where(mask, image, placeholder)
    
    # print(f"Masked image shape: {masked_image.shape}")
        
    return masked_image
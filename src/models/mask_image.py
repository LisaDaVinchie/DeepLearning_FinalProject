import torch as th

def mask_image(image: th.tensor, mask: th.tensor, placeholder = None) -> th.tensor:
    
    if placeholder is None:
        placeholder = th.mean(image).detach()
        
    return th.where(mask, placeholder, image)
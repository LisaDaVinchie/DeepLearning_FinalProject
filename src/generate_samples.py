import matplotlib.pyplot as plt
import torch as th
from pathlib import Path

def sample_generation(model: th.nn.Module, data_loader: th.utils.data.DataLoader, n_samples: int, figs_folder: Path):
    total_images = len(data_loader.dataset)
    
    sample_idx = th.randint(0, total_images, (n_samples,))
    
    for idx in sample_idx:
        image, mask = data_loader.dataset[idx]
        
        model.eval()
        with th.no_grad():
            output = model(image, mask)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].set_title("Original")
        axs[0].axis("off")
        
        axs[1].imshow((image * mask).permute(1, 2, 0))
        axs[1].set_title("Masked")
        axs[1].axis("off")
        
        axs[2].imshow(output.permute(1, 2, 0))
        axs[2].set_title("Inpainted")
        axs[2].axis("off")
        
        fig.savefig(figs_folder / f"sample_{idx}.png")
        plt.close(fig)
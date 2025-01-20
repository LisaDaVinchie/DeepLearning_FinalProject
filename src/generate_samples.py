import matplotlib.pyplot as plt
import torch as th
from pathlib import Path

def reconstruct_image(image, output, mask):
    reconstructed_image = image.clone()
    reconstructed_image[mask] = output[mask]
    return reconstructed_image

def sample_generation(model: th.nn.Module, dataset: dict, n_samples: int, figs_folder: Path):
    total_images = len(dataset["images"])
    
    sample_idx = th.randint(0, total_images, (n_samples,))
    
    for idx in sample_idx:
        image = dataset["images"][idx].unsqueeze(0)
        mask = dataset["masks"][idx].unsqueeze(0)

        print(image.shape, mask.shape)

        model.eval()
        with th.no_grad():
            output = model(image, mask)
            
        rec_image = reconstruct_image(image, output, mask)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image.squeeze().permute(1, 2, 0))
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow((image * mask).squeeze().permute(1, 2, 0))
        axs[1].set_title("Masked")
        axs[1].axis("off")

        axs[2].imshow(rec_image.squeeze().permute(1, 2, 0))
        axs[2].set_title("Inpainted")
        axs[2].axis("off")

        fig.savefig(figs_folder / f"sample_{idx}.png")
        plt.close(fig)
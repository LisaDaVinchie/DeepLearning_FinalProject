import torch as th
from autoencoder import Autoencoder_conv, Autoencoder_unet
import random
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

# Load test data
test_data = th.load("../../data/datasets/test/dataset_500_100_5.pth")

variation = "unet"

if variation == "vanilla":
    figures_folder = Path("../../figs/autoencoder/vanilla/inpainting_results/")
    # Load model
    model = Autoencoder_conv()
    # Load model state
    model.load_state_dict(th.load("../autoencoder.pth"))
    
elif variation == "unet":
    figures_folder = Path("../../figs/autoencoder/unet/inpainting_results/")
    # Load model
    model = Autoencoder_unet()
    # Load model state
    model.load_state_dict(th.load("../autoencoder_unet.pth"))
else:
    print("Invalid variation")
    exit()

index = np.arange(len(test_data["images"]))

if not figures_folder.exists():
    figures_folder.mkdir(parents=True, exist_ok=True)

for i in random.choices(index, k=5):
    # Inpainting
    original_image = test_data["images"][i]
    mask = test_data["masks"][i]
        
    inpainted_image = model(original_image).detach()
    
    image_with_inpainting = original_image * (1 - mask) + inpainted_image * mask

    # Save inpainted image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image.numpy().transpose(1, 2, 0))
    axs[0].set_title("Original image")
    # Highlight mask position
    mask_position = np.where(mask == True)
    y_min, y_max = mask_position[0].min(), mask_position[0].max()
    x_min, x_max = mask_position[1].min(), mask_position[1].max()
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    axs[1].imshow(image_with_inpainting.numpy().transpose(1, 2, 0))
    axs[1].set_title("Inpainted image")
    plt.savefig(figures_folder / f"inpainting_{i}.png")
    plt.close()
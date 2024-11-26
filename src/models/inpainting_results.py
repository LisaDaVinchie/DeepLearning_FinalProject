import torch as th
from autoencoder import Autoencoder_conv, Autoencoder_unet
import random
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

N_TRAIN = 10000
N_TEST = 1000
N_CLASSES = 50
HOLE_PERCENTAGE = 5

identifier = f"{N_TRAIN}_{N_TEST}_{N_CLASSES}_{HOLE_PERCENTAGE}"

# Load test data
test_data_path = Path(f"../../data/datasets/test/dataset_{N_TEST}_{N_CLASSES}_{HOLE_PERCENTAGE}.pth")

if not Path(test_data_path).exists():
    print(f"Path {test_data_path} does not exist")
    exit()
    
weights_folder = Path("../../data/weights/autoencoder/")
figures_base_folder = Path("../../figs/inpainting_results/autoencoder/")

test_data = th.load(test_data_path)

variation = "vanilla"

if variation == "vanilla":
    figures_folder = Path(figures_base_folder / f"vanilla/{identifier}/")
    # Load model
    model = Autoencoder_conv()
    # Load model state
    weights_path = weights_folder / f"vanilla_{identifier}.pth"
    
elif variation == "unet":
    figures_folder = Path(figures_base_folder / f"unet/{identifier}/")
    # Load model
    model = Autoencoder_unet()
    # Load model state
    weights_path = weights_folder / f"unet_{identifier}.pth"
    
else:
    print("Invalid variation")
    exit()

if not weights_path.exists():
    print(f"Weights path {weights_path} does not exist")
    exit()

model.load_state_dict(th.load(weights_path))

index = np.arange(len(test_data["images"]))

if not figures_folder.exists():
    figures_folder.mkdir(parents=True, exist_ok=True)

for i in random.choices(index, k=5):
    # Inpainting
    original_image = test_data["images"][i]
    mask = test_data["masks"][i]
    hole_image = test_data["holes"][i]
        
    inpainted_image = model(hole_image).detach()
    
    image_with_inpainting = hole_image * (1 - mask) + inpainted_image * mask

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
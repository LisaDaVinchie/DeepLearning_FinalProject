import matplotlib.pyplot as plt
import torch as th
from pathlib import Path
import argparse
import json
from models import autoencoder

parser = argparse.ArgumentParser()
parser.add_argument("--path_config", type=Path, required=True, help="Path to the model file")

args = parser.parse_args()
path_config_file = args.path_config

with open(path_config_file, "r") as f:
    config = json.load(f)
    
test_path = Path(config["test_path"])

if not test_path.exists():
    print(f"Path {test_path} does not exist", flush=True)
    exit()

test_dataset = th.load(test_path)



def sample_generation(model: th.nn.Module, dataset: dict, n_samples: int, figs_folder: Path):
    total_images = len(dataset["images"])
    
    sample_idx = th.randint(0, total_images, (n_samples,))
    
    for idx in sample_idx:
        image = dataset["images"][idx]
        mask = dataset["masks"][idx]
        
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
        
model = autoencoder.conv_maxpool(in_channels=3, middle_channels=[16, 32, 64, 128, 256])
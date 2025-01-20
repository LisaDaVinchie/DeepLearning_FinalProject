import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=Path, required=True, help="Path to the paths config file")
parser.add_argument("--figs_folder", type=Path, required=True, help="Path to the parameters config file")
args = parser.parse_args()

results_folder = Path(args.results_folder)
figs_folder = Path(args.figs_folder)

if not results_folder.exists():
    print(f"Results folder {results_folder} does not exist.")
    exit(1)
    
result_files = list(results_folder.glob("*.json"))

if len(result_files) == 0:
    print(f"No result files found in {results_folder}")
    exit(1)

for file in result_files:
    
    file_name = file.stem
    
    print(f"Plotting {file_name}")
    # Load results
    with open(file, "r") as f:
        results = json.load(f)
        

    # Extract metrics
    train_losses = results.get("train_loss", [])
    test_losses = results.get("test_loss", [])
    train_psnr = results.get("train_psnr", [])
    test_psnr = results.get("test_psnr", [])
    train_ssim = results.get("train_ssim", [])
    test_ssim = results.get("test_ssim", [])
    learning_rate = results.get("learning_rate", [])

    # Create subplots
    if len(learning_rate) > 0:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Loss
    axs[0].plot(train_losses, label="Train")
    axs[0].plot(test_losses, label="Test")
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss")

    # Plot PSNR
    axs[1].plot(train_psnr, label="Train")
    axs[1].plot(test_psnr, label="Test")
    axs[1].legend()
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PSNR")
    axs[1].set_title("PSNR")

    # Plot SSIM
    axs[2].plot(train_ssim, label="Train")
    axs[2].plot(test_ssim, label="Test")
    axs[2].legend()
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("SSIM")
    axs[2].set_title("SSIM")

    # Plot Learning Rate (if available)
    if len(learning_rate) > 0:
        axs[3].plot(learning_rate)
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("Learning Rate")
        axs[3].set_title("Learning Rate")

    # Save figure
    plt.tight_layout()
    plot_path = figs_folder / f"{file_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    with open(plot_path.with_suffix(".txt"), "w") as log:
        log.write(f"Result file: {file}\n")
        
    print(f"Plotted {file_name} to {plot_path}\n\n")
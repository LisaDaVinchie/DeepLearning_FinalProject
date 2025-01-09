import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=Path, required=True, help="Path to the paths config file")
parser.add_argument("--params", type=Path, required=True, help="Path to the parameters config file")

args = parser.parse_args()

# Load paths config
with open(args.paths, "r") as f:
    config = json.load(f)

results_path = Path(config["results_path"])
results_plot_path = Path(config["results_plot_path"])

# Ensure output directory exists
results_plot_path.parent.mkdir(parents=True, exist_ok=True)

# Load results
with open(results_path, "r") as f:
    results = json.load(f)

# Extract metrics
train_losses = results.get("train_loss", [])
test_losses = results.get("test_loss", [])
train_psnr = results.get("train_psnr", [])
test_psnr = results.get("test_psnr", [])
train_ssim = results.get("train_ssim", [])
test_ssim = results.get("test_ssim", [])
learning_rate = results.get("learning_rate", [])

if not all([train_losses, test_losses, train_psnr, test_psnr, train_ssim, test_ssim]):
    print("One or more required lists are empty.")
    exit(1)

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
plt.savefig(results_plot_path)
plt.close()
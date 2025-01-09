from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim
import time
import json
import argparse

from models.ImageDataset import CustomImageDataset
from models.transformer import TransformerInpainting
import models.autoencoder as autoencoder
from models.metrics import calculate_psnr, calculate_ssim
from get_workers_number import get_available_cpus
print("Imported all libraries")

parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=Path, required=True, help="Path to the paths config file")
parser.add_argument("--params", type=Path, required=True, help="Path to the parameters config file")

args = parser.parse_args()

paths_config_path = args.paths

with open(paths_config_path, "r") as f:
    config = json.load(f)

train_dataset_path = Path(config["train_path"])
test_dataset_path = Path(config["test_path"])
weights_path = Path(config["weights_path"])
results_path = Path(config["results_path"])

if not train_dataset_path.exists():
    print(f"Path {train_dataset_path} does not exist")
    exit()
    
if not test_dataset_path.exists():
    print(f"Path {test_dataset_path} does not exist")
    exit()

print("Saving model")
if not weights_path.parent.exists():
    weights_path.parent.mkdir(exist_ok=True, parents=True)

if not results_path.parent.exists():
    results_path.parent.mkdir(parents=True, exist_ok=True)

params_config_path = args.params
with open(params_config_path, "r") as f:
    config = json.load(f)
    
useful_keys = ["model_name", "n_train", "n_test", "n_channels", "batch_size", "learning_rate", "epochs", "scheduler", "initialize", "sched_step", "sched_gamma", "image_width"]

for key in useful_keys:
    if key not in config:
        print(f"The key {key} was not found in the parameters config file.")
        exit()

model_name = str(config["model_name"])
n_train = int(config["n_train"])
n_test = int(config["n_test"])
n_channels = int(config["n_channels"])
batch_size = int(config["batch_size"])
learning_rate = float(config["learning_rate"])
epochs = int(config["epochs"])
scheduler = bool(config["scheduler"])
initialize = bool(config["initialize"])
sched_step = int(config["sched_step"])
sched_gamma = bool(config["sched_gamma"])
image_size = int(config["image_width"])

if model_name == "simple_conv":
    useful_keys = ["middle_layers", "kernel_sizes", "strides", "paddings", "output_paddings"]
    for key in useful_keys:
        if key not in config:
            print(f"The key {key} was not found in the parameters config file.")
            exit()
    middle_channels = [int(x) for x in config["middle_layers"].split(" ")]
    kernel_sizes = [int(x) for x in config["kernel_sizes"].split(" ")]
    strides = [int(x) for x in config["strides"].split(" ")]
    paddings = [int(x) for x in config["paddings"].split(" ")]
    output_paddings = [int(x) for x in config["output_paddings"].split(" ")]
    model = autoencoder.simple_conv(in_channels=n_channels,
                                    middle_channels=middle_channels,
                                    kernel_size=kernel_sizes,
                                    stride=strides,
                                    padding=paddings,
                                    output_padding=output_paddings)
elif model_name == "conv_maxpool":
    useful_keys = ["middle_layers"]
    for key in useful_keys:
        if key not in config:
            print(f"The key {key} was not found in the parameters config file.")
            exit()
    middle_channels = [int(x) for x in config["middle_layers"].split(" ")]
    model = autoencoder.conv_maxpool(in_channels=n_channels,
                                     middle_channels=middle_channels)
elif model_name == "conv_unet":
    useful_keys = ["middle_layers", "kernel_sizes", "strides", "paddings", "output_paddings"]
    for key in useful_keys:
        if key not in config:
            print(f"The key {key} was not found in the parameters config file.")
            exit()
    middle_channels = [int(x) for x in config["middle_layers"].split(" ")]
    kernel_sizes = [int(x) for x in config["kernel_sizes"].split(" ")]
    strides = [int(x) for x in config["strides"].split(" ")]
    paddings = [int(x) for x in config["paddings"].split(" ")]
    output_paddings = [int(x) for x in config["output_paddings"].split(" ")]
    model = autoencoder.conv_unet(in_channels=n_channels,
                                  middle_channels=middle_channels,
                                  kernel_sizes=kernel_sizes,
                                  strides=strides)
elif model_name == "transformer":
    useful_keys = ["patch_size", "embedding_dim", "num_heads", "num_layers"]
    for key in useful_keys:
        if key not in config:
            print(f"The key {key} was not found in the parameters config file.")
            exit()
    patch_size = int(config["patch_size"])
    embedding_dim = int(config["embedding_dim"])
    num_heads = int(config["num_heads"])
    num_layers = int(config["num_layers"])
    model = TransformerInpainting(img_size=image_size,
                                  patch_size=patch_size,
                                  embed_dim=embedding_dim,
                                  num_heads=num_heads,
                                  num_layers=num_layers)
else:
    print("Invalid model name")
    exit()

print("Loading datasets")

train_dataset = th.load(train_dataset_path)
print("Train dataset loaded")
test_set = CustomImageDataset(train_dataset)
train_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True)
print("Train dataLoader created")
    

test_dataset = th.load(test_dataset_path)
print("Test dataset loaded")
train_set = CustomImageDataset(test_dataset)
test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
print("Test dataLoader created")

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if scheduler:
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
else:
    scheduler = None
    
def initialize_weights(module):
    if isinstance(module, th.nn.Conv2d) or isinstance(module, th.nn.Linear):
        th.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            th.nn.init.zeros_(module.bias)
if initialize:
    model.apply(initialize_weights)
criterion = nn.MSELoss()

if device == th.device("cpu"):
    n_workers = get_available_cpus()
    th.set_num_threads(n_workers)
    
print("Starting training")
start_time = time.time()
train_losses = []
train_psnr = []
train_ssim = []

test_losses = []
test_psnr = []
test_ssim = []

learning_rates = []

def get_metrics(model: th.nn.Module, device: th.device, criterion: th.nn.Module, image: th.tensor, mask: th.tensor) -> tuple:
    image.to(device)
    mask.to(device)
    output = model(image, mask)
    loss = criterion(output * mask, image * mask) / mask.sum()
    psnr = calculate_psnr(image * mask, output * mask)
    ssim = calculate_ssim(image * mask, output * mask)
    return loss, psnr, ssim

print("\n\n")
for epoch in range(epochs):
    print("Training epoch", epoch)
    model.train()
    batch_loss = 0.0
    batch_psnr = 0.0
    batch_ssim = 0.0
    for batch in train_loader:
        image, mask = batch
        loss, psnr, ssim = get_metrics(model, device, criterion, image, mask)
        batch_loss += loss.item()
        batch_psnr += psnr.item()
        batch_ssim += ssim.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Divide the loss by the number of batches
    train_losses.append(batch_loss / len(train_loader.dataset))
    train_psnr.append(batch_psnr / len(train_loader.dataset))
    train_ssim.append(batch_ssim / len(train_loader.dataset))
    
    model.eval()
    with th.no_grad():
        batch_loss = 0.0
        batch_psnr = 0.0
        batch_ssim = 0.0
        for batch in test_loader:
            image, mask = batch
            loss, psnr, ssim = get_metrics(model, device, criterion, image, mask)
            batch_loss += loss.detach().item()
            batch_psnr += psnr.detach().item()
            batch_ssim += ssim.detach().item()
        test_losses.append(batch_loss / len(test_loader.dataset))
        test_psnr.append(batch_psnr / len(test_loader.dataset))
        test_ssim.append(batch_ssim / len(test_loader.dataset))
        
    if scheduler is not None:
        learning_rates.append(scheduler.get_last_lr()[0])
        scheduler.step()
    print(f"Epoch {epoch} finished\n")
        
print(f"Training finished in {time.time() - start_time} seconds")

metrics_data = {
    "train_loss": train_losses,
    "test_loss": test_losses,
    "learning_rate": learning_rates,
    "train_psnr": train_psnr,
    "train_ssim": train_ssim,
    "test_psnr": test_psnr,
    "test_ssim": test_ssim
}

with open(results_path, "w") as f:
    json.dump(metrics_data, f)
    
th.save(model.state_dict(), weights_path)
print("Model saved")
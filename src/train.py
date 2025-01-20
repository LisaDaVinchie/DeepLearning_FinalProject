from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim
import time
import json
import argparse

from utils.parameter_selection import filter_params
from utils.load_config import load_params
from utils.train_loop import train_loop
from utils.get_workers_number import get_available_cpus
from utils.to_black_and_white import dataset_to_black_and_white
from utils.increment_filepath import increment_filepath
from utils.memory_check import memory_availability_check
from models.ImageDataset import CustomImageDataset
from models.transformer import TransformerInpainting
import models.autoencoder as autoencoder
print("Imported all libraries", flush=True)

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
    print(f"Path {train_dataset_path} does not exist", flush=True)
    exit()
    
if not test_dataset_path.exists():
    print(f"Path {test_dataset_path} does not exist", flush=True)
    exit()

if not weights_path.parent.exists():
    weights_path.parent.mkdir(exist_ok=True, parents=True)

if not results_path.parent.exists():
    results_path.parent.mkdir(parents=True, exist_ok=True)

# Loading parameters
params_config_path = args.params
n_train = None
n_test = None
epochs = None
batch_size = None
learning_rate = None
scheduler = None
rgb = None
initialize = None
sched_step = None
sched_gamma = None
model_name = None

dataset_params = load_params(params_config_path, "dataset_params")
train_params = load_params(params_config_path, "train_params")

locals().update(dataset_params)
locals().update(train_params)


n_channels: int = 1 + 2 * int(rgb)
print("rgb", rgb, flush=True)

with open(params_config_path, "r") as f:
    config = json.load(f)
    
model_params = config.get("model_configs", {}).get(model_name, {})

# Mapping model names to classes
MODEL_CLASSES = {
    "conv_maxpool": autoencoder.conv_maxpool,
    "transformer": TransformerInpainting,
    "simple_conv": autoencoder.simple_conv,
    "conv_unet": autoencoder.conv_unet
}

# Get the corresponding model class
ModelClass = MODEL_CLASSES.get(model_name)

if ModelClass is None:
    raise ValueError(f"Model class for '{model_name}' not found.", flush=True)

print(f"Model class {ModelClass} with parameters {model_params}", flush=True)

filtered_params = filter_params(ModelClass, model_params)

if not filtered_params:
    raise ValueError("No valid parameters found for the model", flush=True)

print(f"Passing parameters n_channels={n_channels}, **filtered_params={filtered_params} to the model", flush=True)
model = ModelClass(n_channels, **filtered_params)

print("Loading datasets", flush=True)

print("Loading datasets", flush=True)
train_dataset = th.load(train_dataset_path)
test_dataset = th.load(test_dataset_path)
print("Datasets loaded", flush=True)

if not rgb:
    train_dataset = dataset_to_black_and_white(train_dataset)
    test_dataset = dataset_to_black_and_white(test_dataset)
    print("Converted datasets to black and white", flush=True)

print("Train dataset loaded", flush=True)
test_set = CustomImageDataset(train_dataset)
train_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

train_set = CustomImageDataset(test_dataset)
test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
print("Test dataLoader created", flush=True)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
else:
    scheduler = None
    
def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
if initialize:
    model.apply(initialize_weights)
    
criterion = nn.MSELoss()

if device == th.device("cpu"):
    n_workers = get_available_cpus()
    th.set_num_threads(n_workers)
    
print("Starting training", flush=True)
start_time = time.time()

if not memory_availability_check(model, train_loader, test_loader):
    print("Not enough memory to train the model", flush=True)
    exit()

print("\nModel type is:", type(model), flush=True)
train_losses, train_psnr, train_ssim, test_losses, test_psnr, test_ssim, learning_rates = train_loop(model = model,
                                                                                                     optimizer = optimizer,
                                                                                                     criterion = criterion,
                                                                                                     train_loader = train_loader,
                                                                                                     test_loader = test_loader,
                                                                                                     epochs = epochs,
                                                                                                     scheduler = scheduler,
                                                                                                     device = device)
        
print(f"Training finished in {time.time() - start_time} seconds", flush=True)


metrics_data = {
    "train_loss": train_losses,
    "test_loss": test_losses,
    "learning_rate": learning_rates,
    "train_psnr": train_psnr,
    "train_ssim": train_ssim,
    "test_psnr": test_psnr,
    "test_ssim": test_ssim,
}

results_path = increment_filepath(results_path)
with open(results_path, "w") as f:
    json.dump(metrics_data, f)

extra_info = f"Model: {model_name}\nModel parameters: {model_params}\nDataset parameters: {dataset_params}\nTraining Time: {time.time() - start_time} seconds\n\n\n"

with open(results_path.with_suffix(".txt"), "w") as f:
    f.write(extra_info)

weights_path = increment_filepath(weights_path)
th.save(model.state_dict(), weights_path)

extra_info = f"Model: {model_name}\nModel parameters: {model_params}\nDataset parameters: {dataset_params}\nTraining Time: {time.time() - start_time} seconds\n\n\n"
with open(weights_path.with_suffix(".txt"), "w") as f:
    f.write(extra_info)
print("Model saved", flush=True)
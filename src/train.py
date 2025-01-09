import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange
import time
import json
import argparse
from models.ImageDataset import CustomImageDataset
from models.transformer import TransformerInpainting
import models.autoencoder as autoencoder
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
losses_figure_path = Path(config["loss_figure_path"])
loss_data_path = Path(config["loss_data_path"])
learning_rates_path = Path(config["learning_rates_path"])

params_config_path = args.params
with open(params_config_path, "r") as f:
    config = json.load(f)

model_name = config["model_name"]
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
    middle_channels = [int(x) for x in config["middle_layers"]]
    kernel_sizes = int(config["kernel_sizes"])
    strides = int(config["strides"])
    paddings = int(config["paddings"])
    output_paddings = int(config["output_paddings"])
    model = autoencoder.simple_conv(in_channels=n_channels,
                                    middle_channels=middle_channels,
                                    kernel_sizes=kernel_sizes,
                                    strides=strides,
                                    paddings=paddings,
                                    output_paddings=output_paddings)
elif model_name == "conv_maxpool":
    middle_channels = [int(x) for x in config["middle_layers"]]
    model = autoencoder.conv_maxpool(in_channels=n_channels,
                                     middle_channels=middle_channels)
elif model_name == "conv_unet":
    middle_channels = [int(x) for x in config["middle_layers"]]
    kernel_sizes = int(config["kernel_sizes"])
    strides = int(config["strides"])
    paddings = int(config["paddings"])
    output_paddings = int(config["output_paddings"])
    model = autoencoder.conv_unet(in_channels=n_channels,
                                  middle_channels=middle_channels,
                                  kernel_sizes=kernel_sizes,
                                  strides=strides)
elif model_name == "transformer":
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

if not train_dataset_path.exists():
    print(f"Path {train_dataset_path} does not exist")
    exit()
    
if not test_dataset_path.exists():
    print(f"Path {test_dataset_path} does not exist")
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
test_losses = []
learning_rates = []

def get_loss(model, device, criterion, image, mask):
    image.to(device)
    mask.to(device)
    output = model(image, mask)
    loss = criterion(output * mask, image * mask) / mask.sum()
    return loss

for epoch in trange(epochs):
    model.train()
    batch_loss = 0.0
    for batch in train_loader:
        image, mask = batch
        loss = get_loss(model, device, criterion, image, mask)
        batch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Divide the loss by the number of batches
    train_losses.append(batch_loss / len(train_loader.dataset))
    
    model.eval()
    with th.no_grad():
        batch_loss = 0.0
        for batch in test_loader:
            image, mask = batch
            loss = get_loss(model, device, criterion, image, mask)
            batch_loss += loss.detach().item()
        test_losses.append(batch_loss / len(test_loader.dataset))
        
    if scheduler is not None:
        learning_rates.append(scheduler.get_last_lr()[0])
        scheduler.step()
        
print(f"Training finished in {time.time() - start_time} seconds")

loss_data = th.stack([th.tensor(train_losses), th.tensor(test_losses)], dim=1).tolist()

if not loss_data_path.parent.exists():
    loss_data_path.parent.mkdir(parents=True, exist_ok=True)
th.save(loss_data, loss_data_path)
    
if scheduler is not None:
    if not learning_rates_path.parent.exists():
        learning_rates_path.parent.mkdir(parents=True, exist_ok=True)
    th.save(learning_rates, learning_rates_path)

print("Saving model")
if not weights_path.parent.exists():
    weights_path.parent.mkdir(exist_ok=True, parents=True)
th.save(model.state_dict(), weights_path)
print("Model saved")

print("Saving training loss plot")
if not losses_figure_path.parent.exists():
    losses_figure_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(losses_figure_path)
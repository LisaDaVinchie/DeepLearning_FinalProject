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

params_config_path = args.params
with open(params_config_path, "r") as f:
    config = json.load(f)

n_train = int(config["n_train"])
n_test = int(config["n_test"])
batch_size = int(config["batch_size"])
learning_rate = float(config["learning_rate"])
epochs = int(config["epochs"])
scheduler = bool(config["scheduler"])
sched_step = int(config["sched_step"])
sched_gamma = bool(config["sched_gamma"])
image_size = int(config["image_width"])

PATCH_SIZE = 16
EMBED_DIM = 1024
NUM_HEADS = 16
NUM_LAYERS = 8

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

model = autoencoder.conv_unet(in_channels=3, middle_channels=[64, 128, 256])
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if scheduler:
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
else:
    scheduler = None
criterion = nn.MSELoss()

if device == th.device("cpu"):
    n_workers = get_available_cpus()
    th.set_num_threads(n_workers)
    
print("Starting training")
start_time = time.time()
train_losses = []
test_losses = []
learning_rates = []

for epoch in trange(epochs):
    model.train()
    batch_loss = 0.0
    for i, batch in enumerate(train_loader):
        image, mask = batch
        optimizer.zero_grad()
        image.to(device)
        mask.to(device)
        output = model(image, mask)
        
        loss = criterion(output * mask, image * mask) / mask.sum()
        batch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Divide the loss by the number of batches
    train_losses.append(batch_loss / len(train_loader.dataset))
    
    model.eval()
    with th.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            image, mask = batch
            image.to(device)
            mask.to(device)
            output = model(image, mask)
            
            loss = criterion(output * mask, image * mask) / mask.sum()
            test_loss += loss.detach().item()
        test_losses.append(test_loss / len(test_loader.dataset))
        
    if scheduler is not None:
        learning_rates.append(scheduler.get_last_lr()[0])
        scheduler.step()
        
print(f"Training finished in {time.time() - start_time} seconds")

loss_data = {"train": train_losses, "test": test_losses}
with open("losses.json", "w") as f:
    json.dump(loss_data, f)
    
if scheduler is not None:
    with open("learning_rates.json", "w") as f:
        json.dump(learning_rates, f)

print("Saving model")
if not weights_path.parent.exists():
    weights_path.parent.mkdir(exist_ok=True, parents=True)
th.save(model.state_dict(), weights_path)
print("Model saved")

print("Saving training loss plot")
if not losses_figure_path.parent.exists():
    losses_figure_path.parent.mkdir(parents=True, exist_ok=True)
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(losses_figure_path)
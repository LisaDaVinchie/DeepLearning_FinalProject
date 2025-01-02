import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
from ImageDataset import CustomImageDataset
from transformer.transformer import TransformerInpainting
from losses import batch_MSE_loss
import torch.optim as optim
from tqdm.auto import trange
import time
import json
import argparse
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

n_train = config["n_train"]
n_test = config["n_test"]
n_classes = config["n_classes"]
mask_percentage = config["mask_percentage"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]

IMG_SIZE = config["image_width"]
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
try:
    train_dataset = th.load(train_dataset_path)
    print("Train dataset loaded")
    th_dataset = CustomImageDataset(train_dataset)
    train_loader = DataLoader(th_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    print("Train dataLoader created")
except Exception as e:
    print(f"Error while loading the dataset: {e}")
    exit()
    
try:
    test_dataset = th.load(test_dataset_path)
    print("Test dataset loaded")
    th_dataset = CustomImageDataset(test_dataset)
    test_loader = DataLoader(th_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    print("Test dataLoader created")
except Exception as e:
    print(f"Error while loading the dataset: {e}")
    exit()

model = TransformerInpainting(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training")
start_time = time.time()
train_losses = []
test_losses = []
for epoch in trange(epochs):
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        img.to(device)
        mask.to(device)
        output = model(img, mask)
        
        loss = batch_MSE_loss(output, img, mask)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    # Divide the loss by the number of batches
    train_losses.append(train_loss / len(train_loader))
    
    model.eval()
    
    with th.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            img, mask, label, masked_image, target = batch
            img.to(device)
            mask.to(device)
            output = model(img, mask)
            
            loss = batch_MSE_loss(output, img, mask)
            test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))
print(f"Training finished in {time.time() - start_time} seconds")

loss_data = {"train": train_losses, "test": test_losses}
with open("losses.json", "w") as f:
    json.dump(loss_data, f)

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
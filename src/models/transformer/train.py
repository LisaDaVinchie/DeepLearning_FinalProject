import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
from data_preprocessing.ImageDataset import CustomImageDataset
from transformer import TransformerInpainting
from models.losses import batch_MSE_loss
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
weights_folder = Path(config["weights_folder"])
losses_folder = Path(config["losses_folder"])

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

IMG_SIZE = 64
PATCH_SIZE = 16
EMBED_DIM = 1024
NUM_HEADS = 16
NUM_LAYERS = 8

identifier = f"{n_train}_{n_test}_{n_classes}_{mask_percentage}"
figure_path = Path(losses_folder / f"transformer_{identifier}.png")
model_weights_name = f"transformer_{identifier}.pth"
model = TransformerInpainting(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)

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
    train_loader = DataLoader(th_dataset, batch_size=batch_size, shuffle=True)
    print("Train dataLoader created")
except Exception as e:
    print(f"Error while loading the dataset: {e}")
    exit()
    
try:
    test_dataset = th.load(test_dataset_path)
    print("Test dataset loaded")
    th_dataset = CustomImageDataset(test_dataset)
    test_loader = DataLoader(th_dataset, batch_size=batch_size, shuffle=True)
    print("Test dataLoader created")
except Exception as e:
    print(f"Error while loading the dataset: {e}")
    exit()

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
        img, mask, label, masked_image, target = batch
        output = model(img, mask)
        
        loss = batch_MSE_loss(output, img, mask)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    # Divide the loss by the number of batches
    train_losses.append(train_loss / len(train_loader))
    
    model.eval()
    
    test_loss = 0.0
    
    for batch in test_loader:
        img, mask, label, masked_image, target = batch
        output = model(img, mask)
        
        loss = batch_MSE_loss(output, img, mask)
        test_loss += loss.item()
    test_losses.append(test_loss / len(test_loader))
    
    
    
print(f"Training finished in {time.time() - start_time} seconds")
    

print("Saving model")
if not weights_folder.exists():
    weights_folder.mkdir(parents=True, exist_ok=True)
th.save(model.state_dict(), weights_folder / model_weights_name)
print("Model saved")

print("Saving training loss plot")
if not losses_folder.exists():
    losses_folder.mkdir(parents=True, exist_ok=True)
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(figure_path)
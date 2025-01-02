import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
from data_preprocessing.ImageDataset import CustomImageDataset
from autoencoder import Autoencoder_conv, Autoencoder_unet
from losses import batch_MSE_loss
import torch.optim as optim
from tqdm.auto import trange
import time
import json
import argparse
print("Imported all libraries")

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, required=True, help="Number of training samples")
parser.add_argument("--n_test", type=int, required=True, help="Number of testing samples")
parser.add_argument("--n_classes", type=int, required=True, help="Number of classes in the dataset")
parser.add_argument("--mask_percentage", type=float, required=True, help="Percentage of the image to be masked")
parser.add_argument("--config_path", type=Path, required=True, help="Path to the config file")

args = parser.parse_args()

n_train = args.n_train
n_test = args.n_test
n_classes = args.n_classes
mask_percentage = args.mask_percentage
config_path = args.config_path

with open(config_path, "r") as f:
    config = json.load(f)

train_dataset_path = Path(config["train_path"])
test_dataset_path = Path(config["test_path"])
weights_folder = Path(config["weights_folder"])
losses_folder = Path(config["losses_folder"])

identifier = f"{n_train}_{n_test}_{n_classes}_{mask_percentage}"

variation = "unet"

if variation == "vanilla":
    model = Autoencoder_conv()
    figure_path = Path(losses_folder / f"vanilla_{identifier}.png")
    model_weights_name = f"vanilla_{identifier}.pth"
    
elif variation == "unet":
    model = Autoencoder_unet()
    figure_path = Path(losses_folder / f"unet_{identifier}.png")
    model_weights_name = f"unet_{identifier}.pth"
    
else:
    print("Invalid variation")
    exit()

if not train_dataset_path.exists():
    print(f"Path {train_dataset_path} does not exist")
    exit()
    
if not test_dataset_path.exists():
    print(f"Path {test_dataset_path} does not exist")
    exit()

batch_size = 32
epochs = 10
learning_rate = 0.01

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
        output = model(masked_image)
        
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
        output = model(masked_image)
        
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
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
from data_preprocessing.ImageDataset import CustomImageDataset
from autoencoder import Autoencoder_conv, Autoencoder_unet
from losses import batch_MSE_loss
import torch.optim as optim
from tqdm.auto import trange
import sys
import time
print("Imported all libraries")

if len(sys.argv) < 3:
    print("Usage: python train.py <train_dataset_path> <test_dataset_path>")
    exit()

train_dataset_path = Path(sys.argv[1])
test_dataset_path = Path(sys.argv[2])

figure_folder = Path("../figs/autoencoder/unet/")
model_folder = Path("./")

if not train_dataset_path.exists():
    print(f"Path {train_dataset_path} does not exist")
    exit()
    
if not test_dataset_path.exists():
    print(f"Path {test_dataset_path} does not exist")
    exit()

batch_size = 32
epochs = 8
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

model = Autoencoder_unet()

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
        img, _, mask, _ = batch
        output = model(img)
        
        loss = batch_MSE_loss(output, img, mask)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    # Divide the loss by the number of batches
    train_losses.append(train_loss / len(train_loader))
    
    model.eval()
    
    test_loss = 0.0
    
    for batch in test_loader:
        img, _, mask, _ = batch
        output = model(img)
        
        loss = batch_MSE_loss(output, img, mask)
        test_loss += loss.item()
    test_losses.append(test_loss / len(test_loader))
    
    
    
print(f"Training finished in {time.time() - start_time} seconds")

if not figure_folder.exists():
    figure_folder.mkdir(parents=True, exist_ok=True)
    

print("Saving model")
th.save(model.state_dict(), model_folder / "autoencoder_unet.pth")
print("Model saved")

print("Saving training loss plot")
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(figure_folder / "losses.png")
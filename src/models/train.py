import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch as th
from data_preprocessing.ImageDataset import CustomImageDataset
from autoencoder import InpaintingAutoencoder
from losses import batch_MSE_loss
import torch.optim as optim
from tqdm.auto import trange
from concurrent.futures import ThreadPoolExecutor
import random
print("Imported all libraries")

dataset_path = Path("data/datasets/train/dataset_2000_100_5.pth")
if not dataset_path.exists():
    print("Path does not exist")
    exit()
figure_path = Path("figs/autoencoder/")

model_path = "model.pth"

batch_size = 32
epochs = 8
learning_rate = 0.01

print("Loading dataset")
try:
    dataset = th.load(dataset_path)
except Exception as e:
    print(f"Error while loading the dataset: {e}")
    exit()
print("Dataset loaded")
th_dataset = CustomImageDataset(dataset)
train_loader = DataLoader(th_dataset, batch_size=batch_size, shuffle=True)
print("DataLoader created")

model = InpaintingAutoencoder()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training")
epoch_losses = []
for epoch in trange(epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        img, _, mask, _ = batch
        output = model(img)
        
        loss = batch_MSE_loss(output, img, mask)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    epoch_losses.append(total_loss / len(train_loader))
    
print("Training finished")

if not figure_path.exists():
    figure_path.mkdir(parents=True, exist_ok=True)
    

print("Saving model")
th.save(model.state_dict(), model_path)
print("Model saved")

print("Saving training loss plot")
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(figure_path / "training_loss.png")
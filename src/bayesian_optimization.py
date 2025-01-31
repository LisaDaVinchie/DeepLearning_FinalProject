import torch as th
from models import autoencoder
import torch.optim as optim
import torch.nn as nn
from utils.train_loop import train_step
import optuna
from pathlib import Path
import argparse
import json
from torch.utils.data import DataLoader
from models.ImageDataset import CustomImageDataset
from utils.get_workers_number import get_available_cpus
from utils.increment_filepath import increment_filepath
import time


start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=Path, required=True, help="Path to the paths config file")

args = parser.parse_args()

paths_config_path = args.paths

with open(paths_config_path, "r") as f:
    config = json.load(f)
    
train_dataset_path = Path(config["train_path"])
test_dataset_path = Path(config["test_path"])
optim_path = Path(config["optim_path"])

if not train_dataset_path.exists():
    print(f"Path {train_dataset_path} does not exist", flush=True)
    exit()
    
if not test_dataset_path.exists():
    print(f"Path {test_dataset_path} does not exist", flush=True)
    exit()
    
train_dataset = th.load(train_dataset_path)
test_dataset = th.load(test_dataset_path)

batch_size = 64
n_trials = 10

print("Train dataset loaded", flush=True)
test_set = CustomImageDataset(train_dataset)
train_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

train_set = CustomImageDataset(test_dataset)
test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
print("Test dataLoader created", flush=True)

search_space = {
    'latent_dim': [16, 32, 64, 128],
    'kernel_size': [3, 5, 7, 9],
    'learning_rate': [1e-2, 1e-3, 1e-4]
}


def objective_function(params):
    # Extract parameters
    latent_dim = params['latent_dim']
    kernel_size = params['kernel_size']
    learning_rate = params['learning_rate']
    
    model = autoencoder.conv_maxpool(in_channels=3,
                                     middle_channels=[latent_dim * (2 ** i) for i in range(5)],
                                     kernel_size=kernel_size,
                                     print_sizes=False)
    
    # Compile the model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    print("Training model", flush=True)
    # Train the model (replace with your dataset and training loop)
    train_loss, train_psnr, train_ssim, test_loss, test_psnr, test_ssim = train_step(model = model,
                                                                                     optimizer = optimizer,
                                                                                     criterion = loss_fn,
                                                                                     train_loader = train_loader,
                                                                                     test_loader = test_loader,
                                                                                     device = device)
    print(f"Train loss: {train_loss}, Test loss: {test_loss}\n", flush=True)
    return test_loss

def optimize_autoencoder(trial):
    # Define the search space
    
    
    latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 64, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7, 9])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.001, 0.0001, 0.00001])
    
    print(f"Optimizing with parameters: latent_dim={latent_dim}, kernel_size={kernel_size}, learning_rate={learning_rate}\n")
    
    # Evaluate the objective function
    params = {
        'latent_dim': latent_dim,
        'kernel_size': kernel_size,
        'learning_rate': learning_rate
    }
    return objective_function(params)

# Run optimization
ncpus = get_available_cpus()
study = optuna.create_study(direction='minimize')
try:
    study.optimize(optimize_autoencoder, n_trials=n_trials, n_jobs=ncpus)
except Exception as e:
    print(f"Study failed for parameters: {study.best_params} with exception: {e}")
    

# Save the best parameters
best_params = study.best_params

optim_path.parent.mkdir(parents=True, exist_ok=True)
optim_path = increment_filepath(optim_path)

with open(optim_path, "w") as f:
    json.dump(best_params, f)
    
# Save training set information
train_info_path = train_dataset_path.with_suffix('.txt')
with open(train_info_path, "w") as f:
    f.write(f"Training set path: {train_dataset_path}\n")
    f.write(f"Test set path: {test_dataset_path}\n")
    f.write(f"Number of training samples: {len(train_set)}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Searching range: {search_space}\n")
    
print(f"Best parameters: {best_params}", flush=True)

end_time = time.time()

print(f"Time taken: {end_time - start_time}", flush=True)
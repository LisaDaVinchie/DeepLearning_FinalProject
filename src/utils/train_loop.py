import torch as th
import torch.nn as nn
from utils.metrics import calculate_psnr, calculate_ssim
from torch.utils.data import DataLoader

def get_metrics(model: nn.Module, criterion: nn.Module, image: th.tensor, mask: th.tensor, device: th.device = th.device("cpu")) -> tuple:
    image = image.to(device)
    mask = mask.to(device)
    output = model(image, mask)
    loss = criterion(output * mask, image * mask) / mask.sum()
    psnr = calculate_psnr(image * mask, output * mask)
    ssim = calculate_ssim(image * mask, output * mask)
    return loss, psnr, ssim

def train_step(model: nn.Module, optimizer: th.optim.Optimizer, criterion: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: th.device = th.device("cpu")):
    model.train()
    batch_loss = 0.0
    batch_psnr = 0.0
    batch_ssim = 0.0
    for batch in train_loader:
        image, mask = batch
        loss, psnr, ssim = get_metrics(model = model,
                                        criterion = criterion,
                                        image = image,
                                        mask = mask,
                                        device = device)
        batch_loss += loss.detach().item()
        batch_psnr += psnr.detach().item()
        batch_ssim += ssim.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = batch_loss / len(train_loader)
    train_psnr = batch_psnr / len(train_loader)
    train_ssim = batch_ssim / len(train_loader)
    
    model.eval()
    with th.no_grad():
        batch_loss = 0.0
        batch_psnr = 0.0
        batch_ssim = 0.0
        for batch in test_loader:
            image, mask = batch
            loss, psnr, ssim = get_metrics(model = model,
                                            criterion = criterion,
                                            image = image,
                                            mask = mask,
                                            device = device)
            batch_loss += loss.detach().item()
            batch_psnr += psnr.detach().item()
            batch_ssim += ssim.detach().item()
    
    test_loss = batch_loss / len(test_loader)
    test_psnr = batch_psnr / len(test_loader)
    test_ssim = batch_ssim / len(test_loader)
    
    return train_loss, train_psnr, train_ssim, test_loss, test_psnr, test_ssim

def train_loop(model: nn.Module, optimizer: th.optim.Optimizer, criterion: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int, scheduler: th.optim.lr_scheduler = None, device: th.device = th.device("cpu")):
    train_losses = []
    train_psnrs = []
    train_ssims = []

    test_losses = []
    test_psnrs = []
    test_ssims = []

    learning_rates = []

    for epoch in range(epochs):
        print("Training epoch", epoch, flush=True)
        model.train()
        train_loss, train_psnr, train_ssim, test_loss, test_psnr, test_ssim = train_step(model = model,
                                                                                          optimizer = optimizer,
                                                                                          criterion = criterion,
                                                                                          train_loader = train_loader,
                                                                                          test_loader = test_loader,
                                                                                          device = device)
        train_losses.append(train_loss)
        train_psnrs.append(train_psnr)
        train_ssims.append(train_ssim)
        test_losses.append(test_loss)
        test_psnrs.append(test_psnr)
        test_ssims.append(test_ssim)
        if scheduler is not None:
            learning_rates.append(scheduler.get_last_lr()[0])
            scheduler.step()
        print(f"Epoch {epoch} finished\n", flush=True)
        
    return train_losses, train_psnrs, train_ssims, test_losses, test_psnrs, test_ssims, learning_rates
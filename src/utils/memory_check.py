import psutil

def check_available_ram():
    total_memory_bytes = psutil.virtual_memory().total
    available_memory_bites = psutil.virtual_memory().available
    
    available_memory_MB = available_memory_bites / 1024**2
    total_memory_MB = total_memory_bytes / 1024**2

    return available_memory_MB, total_memory_MB

def check_model_memory(model):
    model_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
    return model_memory / 1024**2

def check_batch_loader_memory(data_loader):
    for image, mask in data_loader:
        image_size = image.element_size() * image.nelement()  # Memory size of one image in bytes
        mask_size = mask.element_size() * mask.nelement()    # Memory size of one mask in bytes
        batch_memory = (image_size + mask_size) * image.size(0)
        break

    return batch_memory / 1024**2

def check_optimizer_memory(optimizer):
    optimizer_memory = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                optimizer_memory += param.grad.element_size() * param.grad.nelement()
    return optimizer_memory / 1024**2


def memory_availability_check(model, train_loader, test_loader, optimizer):
    model_memory = check_model_memory(model)
    
    train_batch_memory = check_batch_loader_memory(train_loader)
    test_batch_memory = check_batch_loader_memory(test_loader)
    
    optimizer_memory = check_optimizer_memory(optimizer)
    
    required_memory = model_memory + train_batch_memory + test_batch_memory + optimizer_memory
    
    available_memory, _ = check_available_ram()
    
    if available_memory > required_memory:
        print(f"Memory check passed model needs {required_memory} MB, and {available_memory} MB is available", flush=True)
        return True
    else:
        print(f"Memory check failed, model needs at least {required_memory} MB, but only {available_memory} MB is available", flush=True)
        return False
    
    
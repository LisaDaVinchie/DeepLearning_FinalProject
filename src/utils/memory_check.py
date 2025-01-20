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


def memory_availability_check(model, train_loader, test_loader):
    model_memory = check_model_memory(model)
    
    train_batch_memory = check_batch_loader_memory(train_loader)
    test_batch_memory = check_batch_loader_memory(test_loader)
    
    required_memory = model_memory + train_batch_memory + test_batch_memory
    
    available_memory, total_memory = check_available_ram()
    
    if available_memory > required_memory:
        print("Memory check passed")
        return True
    else:
        print(f"Memory check failed, model needs at least {required_memory} MB, but only {available_memory} MB is available")
        return False
    
    
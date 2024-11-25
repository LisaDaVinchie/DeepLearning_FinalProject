import torch
from models.autoencoder import InpaintingAutoencoder

def test_autoencoder_forward():
    """Test the forward pass of the autoencoder"""
    model = InpaintingAutoencoder()
    model.eval()  # Set to evaluation mode
    
    # Create dummy inputs
    batch_size = 2
    input_image = torch.rand(batch_size, 3, 64, 64)  # Simulated input
    mask = torch.zeros(batch_size, 1, 64, 64, dtype=torch.bool)  # Mostly False
    mask[:, :, 16:32, 16:32] = True  # Define a square mask (nxn)

    # Run forward pass
    with torch.no_grad():  # Disable gradient calculation for testing
        output = model(input_image)
        target_region = output * mask

    # Assertions
    assert output.shape == (batch_size, 3, 64, 64), "Output shape is incorrect.\n"
    assert target_region.shape == (batch_size, 3, 64, 64), "Target region shape is incorrect.\n"
    assert not torch.isnan(output).any(), "Output contains NaN values.\n"
    
    print("Forward pass test passed.")

def test_mask_application():
    """Test that the mask is applied correctly"""
    batch_size = 1
    mask = torch.zeros(batch_size, 1, 64, 64, dtype=torch.bool)
    mask[:, :, 16:32, 16:32] = True  # Define mask
    
    dummy_output = torch.rand(batch_size, 3, 64, 64)
    target_region = dummy_output * mask
    
    # Check that non-masked areas are zeroed out
    assert (target_region[:, :, :16, :].sum() == 0), "Non-masked area has non-zero values."
    assert (target_region[:, :, 16:32, 16:32].sum() > 0), "Masked area has zero values."
    
    print("Mask application test passed.")

def test_model_trainable():
    """Test that the model has trainable parameters"""
    model = InpaintingAutoencoder()
    assert any(p.requires_grad for p in model.parameters()), "Model parameters are not trainable."
    
    print("Model trainable test passed.")

import torch
from models.autoencoder import simple_conv, conv_unet, conv_maxpool
import pytest

@pytest.mark.skip(reason="Helper function for testing")
def test_autoencoder_forward(model: torch.nn.Module):
    """Test the forward pass of the autoencoder"""
    model.eval()  # Set to evaluation mode
    
    # Create dummy inputs
    batch_size = 2
    input_image = torch.rand(batch_size, 3, 64, 64)  # Simulated input
    mask = torch.zeros(batch_size, 3, 64, 64, dtype=torch.bool)  # Mostly False
    mask[:, :, 16:32, 16:32] = True  # Define a square mask (nxn)

    # Run forward pass
    with torch.no_grad():  # Disable gradient calculation for testing
        output = model(input_image, mask)
        target_region = output * mask

    # Assertions
    assert output.shape == (batch_size, 3, 64, 64), "Output shape is incorrect.\n"
    assert target_region.shape == (batch_size, 3, 64, 64), "Target region shape is incorrect.\n"
    assert not torch.isnan(output).any(), "Output contains NaN values.\n"
    
    print("Forward pass test passed.")
    
@pytest.mark.skip(reason="Helper function for testing")
def test_mask_application():
    """Test that the mask is applied correctly"""
    batch_size = 1
    mask = torch.zeros(batch_size, 3, 64, 64, dtype=torch.bool)
    mask[:, :, 16:32, 16:32] = True  # Define mask
    
    dummy_output = torch.rand(batch_size, 3, 64, 64)
    target_region = dummy_output * mask
    
    # Check that non-masked areas are zeroed out
    assert (target_region[:, :, :16, :].sum() == 0), "Non-masked area has non-zero values."
    assert (target_region[:, :, 16:32, 16:32].sum() > 0), "Masked area has zero values."
    
    print("Mask application test passed.")

@pytest.mark.skip(reason="Helper function for testing")
def test_model_trainable(model: torch.nn.Module):
    """Test that the model has trainable parameters"""
    assert any(p.requires_grad for p in model.parameters()), "Model parameters are not trainable."
    
    print("Model trainable test passed.")
    
@pytest.mark.skip(reason="Helper function for testing")
def test_output_shape(model: torch.nn.Module):
    """Test the output shape of the model"""
    input_image = torch.rand(2, 3, 64, 64)  # Simulated input
    mask = torch.zeros(2, 3, 64, 64, dtype=torch.bool)  # Mostly False
    
    output = model(input_image, mask)
    
    assert output.shape == (2, 3, 64, 64), "Output shape is incorrect."
    
def test_simple_conv():
    """Test the vanilla autoencoder"""
    model = simple_conv()
    test_autoencoder_forward(model)
    test_model_trainable(model)
    test_mask_application()
    test_output_shape(model)

def test_conv_unet():
    """Test the U-Net autoencoder"""
    model = conv_unet()
    test_autoencoder_forward(model)
    test_model_trainable(model)
    test_mask_application()
    test_output_shape(model)
    
def test_conv_maxpool():
    """Test the maxpool autoencoder"""
    model = conv_maxpool(in_channels=3, middle_channels=[64, 128, 256, 512, 1024])
    test_autoencoder_forward(model)
    test_model_trainable(model)
    test_mask_application()
    test_output_shape(model)
    
    


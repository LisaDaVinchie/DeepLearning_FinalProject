import torch as th

def image_MSE_loss(output: th.tensor, target: th.tensor, mask: th.tensor) -> th.tensor:
    """Mean Squared Error loss function for 2d tensor with mask.

    Args:
        output (th.tensor): the full image output by the model
        target (th.tensor): the original image
        mask (th.tensor): the mask of the image, with true values where the data to inpaint is

    Returns:
        th.tensor: the loss value
    """
    return th.mean((output - target) ** 2 * mask)

def batch_MSE_loss(output: th.tensor, target: th.tensor, mask: th.tensor) -> th.tensor:
    """MSE loss of a batch of images, normalized by the number of images in the batch.

    Args:
        output (th.tensor): the full images output by the model, as a batch_size x height x width tensor
        target (th.tensor): the original images, as a batch_size x height x width tensor
        mask (th.tensor): the masks of the images, as a batch_size x height x width tensor. True values where the data to inpaint is.

    Returns:
        th.tensor: the loss value
    """
    images_per_batch = output.shape[0]
    
    
    loss = 0.0
    for i in range(images_per_batch):
        loss += th.mean((output[i] - target[i]) ** 2 * mask[i])
        
    return loss / images_per_batch
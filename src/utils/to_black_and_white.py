import torch as th


def img_to_black_and_white(image: th.tensor, weights: list = None) -> th.tensor:
    """
    Convert an RGB image to black and white.
    """
    if weights is None:
        weights = th.tensor([0.2989, 0.5870, 0.1140], dtype=th.float32)

    return th.tensordot(weights, image, dims=1).unsqueeze(0)

def dataset_to_black_and_white(dataset: dict):
    bw_dataset = dataset.copy()
    bw_dataset["images"] = [img_to_black_and_white(image) for image in bw_dataset["images"]]
    bw_dataset["masks"] = [mask[0, :, :].unsqueeze(0) for mask in bw_dataset["masks"]]
        
    return bw_dataset
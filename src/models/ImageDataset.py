from torch.utils.data import Dataset
import torch as th

class CustomImageDataset(Dataset):
    def __init__(self, dataset: dict):
        """Custom dataset for images

        Args:
            dataset (dict): A dictionary with the following keys: 'images', 'labels', 'masks'
        """
        
        assert isinstance(dataset, dict), "The dataset should be a dictionary"
        assert "images" in dataset.keys(), "The dataset should have a key 'images'"
        assert "masks" in dataset.keys(), "The dataset should have a key 'masks'"
        if not isinstance(dataset["images"], list):
            raise AssertionError(f"Images should be a list but are of type {type(dataset['images'])}")
        if not isinstance(dataset["masks"], list):
            raise AssertionError(f"Masks should be a list but are of type {type(dataset['masks'])}")
        
        for i, image in enumerate(dataset["images"]):
            if not isinstance(image, th.Tensor):
                raise AssertionError(f"Image at index {i} is not a tensor but of type {type(image)}")
        for i, mask in enumerate(dataset["masks"]):
            if not isinstance(mask, th.Tensor):
                raise AssertionError(f"Mask at index {i} is not a tensor but of type {type(mask)}")
        
        self.data = dataset["images"]
        self.masks = dataset["masks"]
        
        assert len(self.data) == len(self.masks), "Images and masks must have the same length"
    
    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """Returns the image, label and mask at the given index

        Args:
            idx (int): The index of the image, label and mask

        Returns:
            tuple: A tuple containing the image, label and mask at the given index
        """
        image = self.data[idx]
        mask = self.masks[idx]
        
        return image, mask
    
    
            
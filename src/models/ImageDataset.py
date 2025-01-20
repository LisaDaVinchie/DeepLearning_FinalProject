from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataset: dict):
        """Custom dataset for inpainting

        Args:
            dataset (dict): A dictionary with the following keys: 'names', 'images', 'targets'
        """
        
        assert isinstance(dataset, dict), "Dataset must be a dictionary"
        assert 'names' in dataset, "Dataset must contain 'names'"
        assert'coords' in dataset, "Dataset must contain 'coords'"
        assert 'images' in dataset, "Dataset must contain 'images'"
        assert 'masks' in dataset, "Dataset must contain 'masks'"
        
        self.names = dataset['names']
        self.coords = dataset['coords']
        self.data = dataset['images']
        self.masks = dataset['masks']
    
    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """Returns the name, image and target at the given index

        Args:
            idx (int): The index of the item to return

        Returns:
            tuple: A tuple containing the name, image and target at the given index
        """
        name = self.names[idx]
        coords = self.coords[idx]
        image = self.data[idx]
        mask = self.masks[idx]
        
        return name, coords, image, mask
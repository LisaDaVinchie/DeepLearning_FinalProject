from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataset: dict):
        """Custom dataset for images

        Args:
            dataset (dict): A dictionary with the following keys: 'images', 'labels', 'masks'
        """
        
        assert isinstance(dataset, dict), "The dataset should be a dictionary"
        assert "images" in dataset.keys(), "The dataset should have a key 'images'"
        assert "labels" in dataset.keys(), "The dataset should have a key 'labels'"
        assert "masks" in dataset.keys(), "The dataset should have a key 'masks'"
        
        self.data = dataset["images"]
        self.labels = dataset["labels"]
        self.masks = dataset["masks"]
    
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
        label = self.labels[idx]
        mask = self.masks[idx]
        
        return image, mask, label
    
    
            
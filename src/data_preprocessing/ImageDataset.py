from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, data_list: list):
        """Custom dataset for images

        Args:
            data_dict (dict): dictionary with the images data, where the key is the image name and the value is the image tensor. The key should be made as <class_name>_<image_number>
        """
        
        assert isinstance(data_list, list), "The data_list should be a list"
        assert len(data_list) > 0, "The data_list should have at least one element"
        assert isinstance(data_list[0], tuple), "The data_list elements should be tuples"
        assert len(data_list[0]) == 2, "The data_list elements should have two elements"
        
        self.data = []
        self.labels = []
        
        for image, label in data_list:
            self.labels.append(label)
            self.data.append(image)
            
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        return image, label
    
    
            
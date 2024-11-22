from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, data_dict: dict):
        """Custom dataset for images

        Args:
            data_dict (dict): dictionary with the images data, where the key is the image name and the value is the image tensor. The key should be made as <class_name>_<image_number>
        """
        
        self.data = []
        self.keys = list(data_dict.keys())
        
        for name, data in data_dict.items():
            self.data.append(data)
            self.classes = name.split("_")[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.classes[idx]
        
        return image, label
    
    
            
from PIL import Image
from pathlib import Path
from torchvision import transforms
import random
import torch as th
from ImageDataset import CustomImageDataset

# Declare the path to the train images folder
train_images_path = Path("../data/tiny-imagenet-200/train/")

# Declare the path to the train images inner folder
image_subpath = "images"

# Declare the destination folder
destination_folder = Path("../data/datasets/")

images_per_class: int = 10

n_classes = 200

# Check if the train images folder exists
if not train_images_path.exists():
    raise FileNotFoundError("The train images folder does not exist.")

# Get a list of the train images class folders
train_images_folder = [f for f in train_images_path.iterdir() if f.is_dir()]

# Create a dictionary to store the images, with the key being the image name and the value being the image tensor
# taking 10 random images from each class
images_dict = {}
for folder in train_images_folder:
    images = list((folder / image_subpath).glob("*.JPEG"))
    selected_images = random.sample(images, images_per_class)
    
    folder_name = folder.name
    
    images_dict[folder_name] = []
    
    for image in selected_images:
        img = Image.open(image).convert("RGB")

        images_dict[folder_name].append(transforms.ToTensor()(img))


# Create the dataset

dataset = CustomImageDataset(images_dict)

# Check if the destination folder exists
destination_folder.mkdir(parents=True, exist_ok=True)

# Save the dataset
destination_path = destination_folder / f"dataset_{images_per_class}_images_per_class.pth"
th.save(dataset, destination_path)
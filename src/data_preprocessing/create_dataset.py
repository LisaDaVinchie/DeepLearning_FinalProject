from PIL import Image
from pathlib import Path
from torchvision import transforms
import random
import torch as th
import sys



# Declare the path to the train images folder
train_images_path = Path("../../data/tiny-imagenet-200/train/")

# Declare the path to the train images inner folder
image_subpath = "images"

# Declare the destination folder
destination_folder = Path("../../data/datasets/")



images_per_class: int = 10

n_classes = 200

# Check if the train images folder exists
if not train_images_path.exists():
    raise FileNotFoundError("The train images folder does not exist.")

# Get a list of the train images class folders
train_images_folder = [f for f in train_images_path.iterdir() if f.is_dir()]

# Create a dictionary to store the images, with the key being the image name and the value being the image tensor
# taking 10 random images from each class
images_list = []
for folder in train_images_folder:
    images = list((folder / image_subpath).glob("*.JPEG"))
    selected_images = random.sample(images, images_per_class)
    
    label = folder.name
    
    for image in selected_images:
        img = Image.open(image).convert("RGB")
        
        tensor_image = transforms.ToTensor()(img)

        images_list.append((tensor_image, label))


# # Create the dataset
# dataset = CustomImageDataset(images_list)

# Check if the destination folder exists
destination_folder.mkdir(parents=True, exist_ok=True)

# Save the dataset
destination_path = destination_folder / f"dataset_{images_per_class}_images_per_class.pth"
th.save(images_list, destination_path)
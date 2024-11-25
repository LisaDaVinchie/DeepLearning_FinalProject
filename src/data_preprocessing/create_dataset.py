from PIL import Image
from pathlib import Path
from torchvision import transforms
import random
import torch as th
import sys
from dataset_masking.masks import SquareMask


# Declare the path to the train images folder
train_images_path = Path("../data/tiny-imagenet-200/train/")

if not train_images_path.exists():
    sys.exit("The train images folder does not exist.")

# Declare the path to the train images inner folder
image_subpath = "images"

# Declare images extension
images_extension: str = ".JPEG"

# Declare the dataset extension
dataset_extension: str = ".pth"

# Declare the destination folder
destination_folder = Path("../data/datasets/")

# Declare the total number of images and classes
total_images: int = 2000
n_classes = 100

# Declare the image width and height and calculate the number of pixels in the square mask
image_width: int = 64
image_height: int = 64
mask_percentage: int = 5 # 5% of the image

n_pixels = int((mask_percentage / 100) * image_width * image_height)

# Get a list of the train images class folders
train_images_folder = [f for f in train_images_path.iterdir() if f.is_dir()]

if len(train_images_folder) <= 0:
    sys.exit("The train images folder is empty.")
    
if n_classes > len(train_images_folder):
    sys.exit(f"The train images folder contains {len(train_images_folder)} classes. The number of classes should be less than or equal to the number of classes in the train images folder.")

# Calculate the number of classes and images per class
images_per_class: int = total_images // n_classes

# Create the SquareMask object
mask_class = SquareMask(image_width, image_height, n_pixels)

# Declare the lists to store the images, labels and masks
images_list = [None] * total_images
labels_list = [None] * total_images
targets_list = [None] * total_images
masks_list = [None] * total_images

# Iterate over the train images class folders
print(f"Creating the dataset with {total_images} images, {n_classes} classes and {images_per_class} images per class.")
i: int = 0
for folder in random.sample(train_images_folder, n_classes):
    
    # Get a list of the images in the class folder
    
    images = list((folder / image_subpath).glob(f"*{images_extension}"))

    if len(images) <= 0:
        print(f"The class folder {folder.name} does not contain any images.")
        continue
    
    # Get the label of the class folder
    label = folder.name
    
    # Iterate over the selected images
    for image in random.sample(images, images_per_class):
        
        # Open the image
        img = Image.open(image).convert("RGB")
        
        # Transform the image to a tensor
        tensor_image = transforms.ToTensor()(img)
        
        # Append the image tensor and the label to the images_list and labels_list
        images_list[i] = tensor_image
        labels_list[i] = label
        
        # Create the square maskx
        mask = mask_class.create_square_mask()
        
        # Append the mask to the masks_list
        masks_list[i] = mask
        
        target = tensor_image * mask
        
        # Append the mask to the masks_list
        targets_list[i] = target

        
        i += 1
        
data = {"images": images_list, "labels": labels_list, "targets": targets_list, "masks": masks_list}

print("Dataset created.")

# Check if the destination folder exists
destination_folder.mkdir(parents=True, exist_ok=True)

# Save the dataset
print("Saving the dataset.")
destination_path = destination_folder / f"dataset_{total_images}_{n_classes}_{mask_percentage}{dataset_extension}"
th.save(data, destination_path)
print(f"Dataset saved in {destination_path}.")
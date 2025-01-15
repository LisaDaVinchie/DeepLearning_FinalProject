from PIL import Image
from pathlib import Path
from torchvision import transforms
import random
import torch as th
import sys
import json
import argparse
import sys
from pathlib import Path

from utils.masks import SquareMask, LineMask

# Add the parent directory to sys.path
parent_folder = Path(__file__).resolve().parent.parent.parent

print(f"\nAdding {str(parent_folder)} to sys.path\n")
sys.path.append(str(parent_folder))

# Import your module or function
from src.utils import parameter_selection as ps

parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=Path, required=True, help="Path to the paths config file")
parser.add_argument("--params", type=Path, required=True, help="Path to the parameters config file")

args = parser.parse_args()

paths_file_path = args.paths
params_file_path = args.params

if not paths_file_path.exists():
    exit(f"Paths file {paths_file_path} does not exist.")
if not params_file_path.exists():
    exit(f"Parameters file {params_file_path} does not exist.")

with open (paths_file_path, "r") as f:
    paths = json.load(f)

with open (params_file_path, "r") as f:
    params = json.load(f)

original_images_folder = Path(paths["raw_data_folder"])
train_path = Path(paths["train_path"])
test_path = Path(paths["test_path"])
n_train = int(params["n_train"])
n_test = int(params["n_test"])
n_classes = int(params["n_classes"])
mask_percentage = int(params["mask_percentage"])
rgb = ps.typecast_bool(str(params["rgb"]))
image_width = int(params["image_width"])
image_height = int(params["image_height"])
images_extension = str(paths["image_extension"])

if not original_images_folder.exists():
    sys.exit("The original images folder does not exist.")

# Declare the path to the train images inner folder
image_subpath = "images"

# Declare the dataset extension
dataset_extension: str = ".pth"

# Declare the image width and height and calculate the number of pixels in the square mask

n_pixels = int((mask_percentage / 100) * image_width * image_height)

# Get a list of the train images class folders
folder_list = [f for f in original_images_folder.iterdir() if f.is_dir()]

if len(folder_list) <= 0:
    sys.exit("The train images folder is empty.")
    
if n_classes > len(folder_list):
    sys.exit(f"The train images folder {original_images_folder} contains {len(folder_list)} classes. The number of classes should be less than or equal to the number of classes in the train images folder.")

if n_train % n_classes != 0:
    sys.exit(f"The number of train images {n_train} is not divisible by the number of classes {n_classes}.")
if n_test % n_classes != 0:
    sys.exit(f"The number of test images {n_test} is not divisible by the number of classes {n_classes}.")

# Calculate the number of classes and images per class
train_images_per_class: int = n_train // n_classes
test_images_per_class: int = n_test // n_classes

images_per_class_in_dataset = len(list(Path(folder_list[0]/image_subpath).glob(f"*{images_extension}")))

if images_per_class_in_dataset <= 0:
    sys.exit(f"The train images folder {folder_list[0]} is empty.")
# Check if the train images folder contains enough images to create the dataset
if train_images_per_class + test_images_per_class > images_per_class_in_dataset:
    sys.exit(f"The train images folder does not contain enough images to create the dataset with {n_train} train images and {n_test} test images.")

# Create the SquareMask object
mask_class = SquareMask(image_width, image_height, n_pixels)
print("Mask class created with the following parameters:", image_width, image_height, n_pixels, "\n")

def extract_image_info(image_list: list, rgb: bool = True):
    tensor_list = [None] * len(image_list)
    mask_list = [None] * len(image_list)
    
    if rgb:
        mode = "RGB"
    else:
        mode = "L"
    
    for i, image in enumerate(image_list):
        try:
            img = Image.open(image).convert(mode)
        except Exception as e:
            print(f"Error opening image {image}: {e}")
            continue
        tensor_image = transforms.ToTensor()(img)
        
        mask = th.tensor(mask_class.create_mask()).unsqueeze(0)
        
        if rgb:
            mask = mask.repeat(3, 1, 1)
        
        tensor_list[i] = tensor_image
        mask_list[i] = mask
    return tensor_list, mask_list

# Iterate over the train images class folders
print(f"Creating the dataset with:\n{n_train} train images\n{n_test} test images\n{n_classes} classes\n{mask_percentage}% mask\nrgb={rgb}\n")

def create_dicts(folder_list, n_classes, n_train, n_test, train_images_per_class, test_images_per_class, rgb):
    # Declare the lists to store the images, labels and masks
    train_images_list = [None] * n_train
    train_masks_list = [None] * n_train
    train_labels_list = [None] * n_train

    test_images_list = [None] * n_test
    test_masks_list = [None] * n_test
    test_labels_list = [None] * n_test

    # Iterate over the train images class folders
    i: int = 0
    j: int = 0
    for folder in random.sample(folder_list, n_classes):
        print(f"Processing class folder {folder.name}")
        # Get a list of the images in the class folder
        
        images = list(Path(folder / image_subpath).glob(f"*{images_extension}"))
        
        # Get the label of the class folder
        label = folder.name
        
        images = random.sample(images, train_images_per_class + test_images_per_class)
        
        train_images = random.sample(images, train_images_per_class)
        test_images = [img for img in images if img not in train_images]
                
        # Iterate over the train images
        train_images_list[i:i + train_images_per_class], train_masks_list[i:i + train_images_per_class] = extract_image_info(train_images, rgb)
        
        train_labels_list[i:i + train_images_per_class] = [label] * train_images_per_class
        i += train_images_per_class
        
        # Iterate over the test images
        test_images_list[j:j + test_images_per_class], test_masks_list[j:j + test_images_per_class] = extract_image_info(test_images, rgb)
        
        test_labels_list[j:j + test_images_per_class] = [label] * test_images_per_class
        j += test_images_per_class
        
    print("Lists created.")
            
    train_data = {"images": train_images_list,
                "masks": train_masks_list,
                "labels": train_labels_list}
    test_data = {"images": test_images_list,
                "masks": test_masks_list,
                "labels": test_labels_list}
    
    return train_data, test_data

train_data, test_data = create_dicts(folder_list, n_classes, n_train, n_test, train_images_per_class, test_images_per_class, rgb)

for data in [train_data, test_data]:
    for key in data.keys():
        if any([x is None for x in data[key]]):
            print(f"Error in the {key} list: list not filled completely.")
            exit()

print("Datasets created.")

# Save the dataset
print("Saving the datasets.")
train_path.parent.mkdir(parents=True, exist_ok=True)
th.save(train_data, train_path)
print(f"Train dataset saved in {train_path}.")

test_path.parent.mkdir(parents=True, exist_ok=True)
th.save(test_data, test_path)
print(f"Test dataset saved in {test_path}.")
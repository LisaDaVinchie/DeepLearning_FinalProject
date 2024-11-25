from PIL import Image
from pathlib import Path
from torchvision import transforms
import random
import torch as th
import sys
from dataset_masking.masks import SquareMask

if len(sys.argv) < 4:
    sys.exit("Usage: python create_dataset.py <original_images_path> <train_dataset_path> <test_dataset_path>")

original_images_path = Path(sys.argv[1])
train_folder = Path(sys.argv[2])
test_folder = Path(sys.argv[3])

# Declare the total number of images and classes
total_train_images: int = 2000
total_test_images = 500
n_classes = 100

if not original_images_path.exists():
    sys.exit("The original images folder does not exist.")

# Declare the path to the train images inner folder
image_subpath = "images"

# Declare images extension
images_extension: str = ".JPEG"

# Declare the dataset extension
dataset_extension: str = ".pth"

# Declare the image width and height and calculate the number of pixels in the square mask
image_width: int = 64
image_height: int = 64
mask_percentage: int = 5 # 5% of the image

n_pixels = int((mask_percentage / 100) * image_width * image_height)

# Get a list of the train images class folders
folder_list = [f for f in original_images_path.iterdir() if f.is_dir()]

if len(folder_list) <= 0:
    sys.exit("The train images folder is empty.")
    
if n_classes > len(folder_list):
    sys.exit(f"The train images folder contains {len(folder_list)} classes. The number of classes should be less than or equal to the number of classes in the train images folder.")

# Calculate the number of classes and images per class
train_images_per_class: int = total_train_images // n_classes
test_images_per_class: int = total_test_images // n_classes

# Check if the train images folder contains enough images to create the dataset
if train_images_per_class + test_images_per_class > len(list(folder_list[0].glob(f"{image_subpath}/*{images_extension}"))):
    sys.exit(f"The train images folder does not contain enough images to create the dataset with {total_train_images} train images and {total_test_images} test images.")

# Create the SquareMask object
mask_class = SquareMask(image_width, image_height, n_pixels)

def extract_image_info(image_list: list):
    tensor_list = [None] * len(image_list)
    mask_list = [None] * len(image_list)
    target_list = [None] * len(image_list)
    
    for i, image in enumerate(image_list):
        try:
            img = Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image}: {e}")
            continue
        tensor_image = transforms.ToTensor()(img)
        mask = mask_class.create_square_mask()
        target = tensor_image * mask
        
        tensor_list[i] = tensor_image
        mask_list[i] = mask
        target_list[i] = target
    return tensor_list, mask_list, target_list

# Iterate over the train images class folders
print(f"Creating the dataset with:\n{total_train_images} train images\n{total_test_images} test images\n{n_classes} classes\n{mask_percentage}% mask")

# Declare the lists to store the images, labels and masks
train_images_list = [None] * total_train_images
train_labels_list = [None] * total_train_images
train_targets_list = [None] * total_train_images
train_masks_list = [None] * total_train_images

test_images_list = [None] * total_test_images
test_labels_list = [None] * total_test_images
test_targets_list = [None] * total_test_images
test_masks_list = [None] * total_test_images

# Iterate over the train images class folders
i: int = 0
j: int = 0
for folder in random.sample(folder_list, n_classes):
    
    # Get a list of the images in the class folder
    
    images = list((folder / image_subpath).glob(f"*{images_extension}"))

    if len(images) <= 0:
        print(f"The class folder {folder.name} does not contain any images.")
        continue
    
    # Get the label of the class folder
    label = folder.name
    
    train_images = random.sample(images, train_images_per_class)
    candidate_test_images = [image for image in images if image not in train_images]
    test_images = random.sample(candidate_test_images, test_images_per_class)
            
    # Iterate over the train images
    train_images_list[i:i + train_images_per_class], train_masks_list[i:i + train_images_per_class], train_targets_list[i:i + train_images_per_class] = extract_image_info(train_images)
    i += train_images_per_class
    
    # Iterate over the test images
    test_images_list[j:j + test_images_per_class], test_masks_list[j:j + test_images_per_class], test_targets_list[j:j + test_images_per_class] = extract_image_info(test_images)
    j += test_images_per_class
    
print("Lists created.")
        
train_data = {"images": train_images_list, "labels": train_labels_list, "targets": train_targets_list, "masks": train_masks_list}
test_data = {"images": test_images_list, "labels": test_labels_list, "targets": test_targets_list, "masks": test_masks_list}

print("Datasets created.")

# Check if the destination folder exists
train_folder.mkdir(parents=True, exist_ok=True)
test_folder.mkdir(parents=True, exist_ok=True)

# Save the dataset
print("Saving the datasets.")
destination_path = train_folder / f"dataset_{total_train_images}_{n_classes}_{mask_percentage}{dataset_extension}"
th.save(train_data, destination_path)
print(f"Train dataset saved in {destination_path}.")

destination_path = test_folder / f"dataset_{total_test_images}_{n_classes}_{mask_percentage}{dataset_extension}"
th.save(test_data, destination_path)
print(f"Test dataset saved in {destination_path}.")
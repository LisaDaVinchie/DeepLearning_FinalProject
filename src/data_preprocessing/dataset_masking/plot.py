import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys

def plot_image(path: Path, masks_folder: Path, figs_folder: Path):
    image = np.load(path, allow_pickle=False)
    depths = image[2, :, :]

    try:
        mask = np.load(masks_folder / path.name, allow_pickle=False)
    except FileNotFoundError:
        print(f"Mask for {masks_folder / path.name} not found")
    
    masked_image = depths.copy()
    mask = mask.astype(bool)
    masked_image[mask] = np.nan
    
    try:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(masked_image)
        axs[0].set_title("Masked Image")
        plt.colorbar(axs[0].imshow(masked_image), ax=axs[0])
        axs[1].imshow(depths)
        axs[1].set_title("Original Image")
        plt.tight_layout()
    except Exception as e:
        print(f"Error plotting {path}: {e}")
        return
    
    plt.savefig(figs_folder / f"{path.stem}.png")
    plt.close(fig)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python plot.py <path_to_mask_folder> <path_to_interpolated_data_folder> <path_to_save_folder>")
        sys.exit(1)

    masks_folder = Path(sys.argv[1])
    interpolated_data_folder = Path(sys.argv[2])
    figs_folder = Path(sys.argv[3])

    assert masks_folder.exists(), f"Folder {masks_folder} does not exist"
    assert interpolated_data_folder.exists(), f"Folder {interpolated_data_folder} does not exist"
    print(f"Plotting images from {interpolated_data_folder} and masks from {masks_folder}")

    masks_paths = list(masks_folder.glob("*.npy"))
    interpolated_data_paths = list(interpolated_data_folder.glob("*.npy"))
    print(f"Found {len(masks_paths)} masks and {len(interpolated_data_paths)} interpolated data")

    figs_folder.mkdir(exist_ok=True, parents=True)
    print(f"Saving figures to {figs_folder}")
    
    args = [(path, masks_folder, figs_folder) for path in interpolated_data_paths]

    print("Plotting images...")
    with ProcessPoolExecutor() as executor:
        executor.map(plot_image, *zip(*args))
        
    print("Done!")
        

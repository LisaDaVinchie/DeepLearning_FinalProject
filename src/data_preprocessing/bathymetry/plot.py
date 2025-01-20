import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys

def plot_image(path: Path, figs_folder: Path):
    image = np.load(path, allow_pickle=False)
    depths = image[2, :, :]
    
    try:
        fig = plt.figure()
        plt.imshow(depths)
        xticks = image[0, 0, :]
        yticks = image[1, :, 0]
        plt.xticks(range(0, len(xticks), 20), xticks[::20], rotation=45)
        plt.yticks(range(0, len(yticks), 20), yticks[::20])
        plt.colorbar(plt.imshow(depths))
        plt.tight_layout()
    except Exception as e:
        print(f"Error plotting {path}: {e}")
        return
    
    plt.savefig(figs_folder / f"{path.stem}.png")
    plt.close(fig)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python plot.py <path_to_bathymetry_folder> <path_to_save_folder>")
        sys.exit(1)

    bath_folder = Path(sys.argv[1])
    figs_folder = Path(sys.argv[2])

    assert bath_folder.exists(), f"Folder {bath_folder} does not exist"
    print(f"Plotting images from {bath_folder}")

    bath_paths = list(bath_folder.glob("*.npy"))
    print(f"Found {len(bath_paths)} bathymetry data")

    figs_folder.mkdir(exist_ok=True, parents=True)
    print(f"Saving figures to {figs_folder}")
    
    args = [(path, figs_folder) for path in bath_paths]

    with ProcessPoolExecutor() as executor:
        executor.map(plot_image, *zip(*args))
        
    print("Done!")
        

import requests
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

urls = ["https://downloads.emodnet-bathymetry.eu/v11/F5_2022.nc.zip",
        "https://downloads.emodnet-bathymetry.eu/v11/F6_2022.nc.zip",
        "https://downloads.emodnet-bathymetry.eu/v11/E5_2022.nc.zip",
        "https://downloads.emodnet-bathymetry.eu/v11/E6_2022.nc.zip"]

if len(sys.argv) < 2:
    print("Usage: python download_bathymetry_data.py <download_folder>")
    exit()
    
# Get the download folder
download_folder = Path(sys.argv[1])

# Initialize the session
session = requests.Session()

# Define the function to download the file
def download_url(url: str, session: requests.Session):
    print(f"Downloading {url}")
    file_name = url.split('/')[-1]
    file_path = download_folder / file_name
    
    with (session.get(url, stream=True)) as response :
        response.raise_for_status()
        with open(file_path, "wb") as file_url:
            for chunk in response.iter_content(chunk_size=8192):
                file_url.write(chunk)
        print(f"Downloaded {file_name}")
    return file_name

n_files: int = 0
start_time = time.time()

print("Downloading files...")

# Create the download folder if it does not exist
download_folder.mkdir(parents=True, exist_ok=True)

max_w = min(os.cpu_count(), len(urls))

# Download the files
with requests.Session() as session, ThreadPoolExecutor(max_workers = max_w) as executor:
    futures = [executor.submit(download_url, url, session) for url in urls]
    
    for future in as_completed(futures):
        try:
            future.result()
            n_files += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            # If the file was not downloaded, remove the download folder
            if download_folder.exists() and not any(download_folder.iterdir()):
                download_folder.rmdir()

print(f"Downloaded {n_files} files to {download_folder} in {time.time() - start_time:.2f} seconds.")
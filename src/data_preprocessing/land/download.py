import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
from pathlib import Path
import os

if len(sys.argv) < 2:
    print("Please provide the Copernicus Land Ortho ID and download path.")
    exit()

id = sys.argv[1]
base_url = "https://egms.land.copernicus.eu/insar-api/archive/download/EGMS_L3_{}_100km_E_2018_2022_1.zip?id=" + id

min_easting, max_easting = 34, 56
min_northing, max_northing = 13, 34
MAX_ATTEMPTS = 10
MAX_TIMEOUT = 1000

# Create the list of URLs
urls = [base_url.format(f"E{e:02}N{n:02}") for e in range(min_easting, max_easting+1) for n in range(min_northing, max_northing+1)]

print(f"Created {len(urls)} URLs.")

downloaded_folder_path = Path(sys.argv[2])

downloaded_folder_path.mkdir(parents=True, exist_ok=True)
non_existent_file_list_path = downloaded_folder_path.parent / "non_existent_files.txt"

try:
    with open(non_existent_file_list_path, "r") as f:
        non_existing_files = [line.strip() for line in f.readlines()]
        
        non_existing_urls = [base_url.format(file.split('/')[-1].split('?')[0]) for file in non_existing_files]
    
    valid_urls = [url for url in urls if url not in non_existing_urls]
    
except FileNotFoundError:
    non_existing_urls = []
    valid_urls = urls

already_downloaded_files = list(downloaded_folder_path.glob("*.zip"))

already_downloaded_urls = [url.split('/')[-1].split('?')[0] for url in already_downloaded_files]

valid_urls = [url for url in valid_urls if url.split('/')[-1].split('?')[0] not in already_downloaded_urls]

print(f"Found {len(non_existing_urls)} non-existent files.")
        
n_files: int = 0

start_time = time.time()

max_workers = min(os.cpu_count(), len(valid_urls))

with open(non_existent_file_list_path, "a") as non_existent_file_list:
    session = requests.Session()
    
    def download_url(url, attempts = MAX_ATTEMPTS):
        file_name = url.split('/')[-1].split('?')[0]

        file_path = downloaded_folder_path / file_name
        for attempt in range(attempts):
            try:
                response = session.get(url, timeout = MAX_TIMEOUT)
                if response.status_code == 200:
                    with open(file_path, "wb") as file_url:
                        print(f"Downloading {url} to {file_path}.")
                        file_url.write(response.content)
                        return True
                elif response.status_code == 429 or response.status_code == 502:
                    print("Server overload, waiting...")
                    time.sleep(2**attempt)
                    print("Retrying...")
                else:
                    non_existing_file = url.split('/')[-1].split('?')[0]
                    non_existent_file_list.write(non_existing_file + "\n")
                    print(f"File {non_existing_file} was not found, status code {response.status_code}.")
                    return False
                    
            
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {url}: {e}")
                return False
    
    print(f"Downloading {len(valid_urls)} files.")
         
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = [executor.submit(download_url, url) for url in valid_urls]
        
        for future in as_completed(futures):
            if future.result():
                n_files += 1

print(f"Downloaded {n_files} files in {time.time() - start_time} seconds.")
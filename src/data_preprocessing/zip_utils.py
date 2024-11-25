import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def unzip(zip_file_path: Path, extract_folder_path: Path):
    """Extract the zip file to the given folder path.

    Args:
        zip_file_path (Path): Path to the zip file to be extracted.
        extract_folder_path (Path): Path of the folder where the file will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        try:
            zip_ref.extractall(extract_folder_path)
        except Exception as e:
            print(f"Error: {e}")
            raise e
    

def zip_to_file(file_paths_list: list, zip_file_path: Path):
    """Compress a list of files to a single zip file.

    Args:
        file_paths_list (list): List of file paths to be compressed.
        zip_file_path (Path): Path of the zip file that will be created.
    """
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths_list:
            try:
                zipf.write(file, file.name)
            except Exception as e:
                print(f"Error: {e}")
                return
    print(f"Created {zip_file_path}.")

def zip_to_group(self, file_paths_list: list, group_size_MB: int, zip_folder_path: Path):
    """Compress list of files to multiple zip files based on the group size.

    Args:
        file_paths_list (list): List of file paths to be compressed.
        group_size_MB (int): Maximum size of each group of files to be compressed.
        zip_folder_path (Path): Path of the folder where the zip files will be created.
    """

    current_group_size = 0
    current_group_files = []
    zip_folder_index = 1
    zip_tasks = []
    
    if len(file_paths_list) == 1:
        file_to_zip = file_paths_list[0]
        zip_to_file(file_to_zip, zip_folder_path / f'{file_to_zip.stem}.zip')
    
    
    workers = min(len(file_paths_list), 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for file_path in file_paths_list:
            file_size = file_path.stat().st_size

            if current_group_size + file_size > group_size_MB * 1024 * 1024:
                # Submit the current group for compression
                zip_file_name = f'group_{zip_folder_index}.zip'
                zip_file_path = zip_folder_path / zip_file_name
                zip_tasks.append(executor.submit(self.compress_to_file, current_group_files, zip_file_path))
                print(f"Created {zip_file_path}.")
                
                zip_folder_index += 1
                current_group_files = []
                current_group_size = 0

            current_group_files.append(file_path)
            current_group_size += file_size

        # Handle any remaining files in the last group
        if current_group_files:
            zip_file_name = f'group_{zip_folder_index}.zip'
            zip_file_path = zip_folder_path / zip_file_name
            zip_tasks.append(executor.submit(self.compress_to_file, current_group_files, zip_file_path))
            print(f"Created {zip_file_path}.")

        # Wait for all tasks to complete
        for task in zip_tasks:
            task.result()

    print("Done.")

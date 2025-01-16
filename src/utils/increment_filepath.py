from pathlib import Path

# Check if weights_path already exists and modify the path if necessary
def increment_filepath(file_path: Path) -> Path:
    if file_path.exists():
        base = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent
        counter = 1
        while file_path.exists():
            file_path = parent / f"{base}_{counter}{suffix}"
            counter += 1
    return file_path
import os
import zipfile

# Define expected directories
source_dir = "Trainning_SET"
labels_dir = os.path.join(source_dir, "Labels")
images_dir = os.path.join(source_dir, "Images")

# Function to check directory existence
def ensure_directories():
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found.")
        return False

    if not os.path.exists(labels_dir):
        print(f"Labels directory '{labels_dir}' not found.")
        return False

    if not os.path.exists(images_dir):
        print(f"Images directory '{images_dir}' not found.")
        return False

    print("All necessary directories are in place.")
    return True

# Function to extract ZIP file if needed
def extract_zip_if_needed(zip_path, extract_to):
    if os.path.exists(zip_path):
        print(f"Found ZIP file '{zip_path}'. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction completed to '{extract_to}'.")
    else:
        print(f"ZIP file '{zip_path}' not found.")

# Main logic
if not ensure_directories():
    zip_file = "Trainning_SET.zip"  # Replace with your ZIP file name if necessary
    extract_zip_if_needed(zip_file, ".")  # Extract in the current directory

    # Re-check after extraction
    if ensure_directories():
        print("Ready to process the data.")
    else:
        print("Directories are still missing. Please check your data structure.")

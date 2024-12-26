import os
import shutil
import random

def create_folders(base_path):
    paths = [
        os.path.join(base_path, "images/train"),
        os.path.join(base_path, "images/val"),
        os.path.join(base_path, "labels/train"),
        os.path.join(base_path, "labels/val"),
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

def filter_labels_and_copy(input_folder, output_folder, a, b):
    images_path = os.path.join(input_folder, "Images")
    labels_path = os.path.join(input_folder, "Labels")

    train_images_path = os.path.join(output_folder, "images/train")
    train_labels_path = os.path.join(output_folder, "labels/train")

    for label_file in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, label_file)
        
        # Check if the label file has exactly 572 lines
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 572:
                # Filter labels based on the vertical rank (y-coordinate)
                filtered_lines = []
                y_coords = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, _, y_center, _, _ = map(float, parts)
                        y_coords.append(y_center)

                # Sort indices based on y-coordinates
                sorted_indices = sorted(range(len(y_coords)), key=lambda i: y_coords[i])
                selected_indices = sorted_indices[a-1:b]  # Select ranks from a to b (inclusive)

                for idx in selected_indices:
                    filtered_lines.append(lines[idx])

                # Write filtered labels to the new file
                filtered_label_file_path = os.path.join(train_labels_path, label_file)
                with open(filtered_label_file_path, 'w') as out_f:
                    out_f.writelines(filtered_lines)

                # Copy the corresponding image file
                image_file = label_file.replace(".txt", ".jpg")
                image_file_path = os.path.join(images_path, image_file)
                if os.path.exists(image_file_path):
                    shutil.copy(image_file_path, train_images_path)

def split_train_val(output_folder, val_ratio=0.05):
    train_images_path = os.path.join(output_folder, "images/train")
    train_labels_path = os.path.join(output_folder, "labels/train")
    val_images_path = os.path.join(output_folder, "images/val")
    val_labels_path = os.path.join(output_folder, "labels/val")

    image_files = os.listdir(train_images_path)
    
    # Shuffle and split data for validation
    val_count = max(1, int(len(image_files) * val_ratio))
    val_files = random.sample(image_files, val_count)

    for val_file in val_files:
        # Move image to validation folder
        shutil.move(os.path.join(train_images_path, val_file), val_images_path)

        # Move corresponding label to validation folder
        label_file = val_file.replace(".jpg", ".txt")
        shutil.move(os.path.join(train_labels_path, label_file), val_labels_path)

# Define paths
input_folder = "Trainning_SET"
output_folder = "data_part2"

# Parameters for label filtering
a = 251
b = 314

# Create required folder structure
create_folders(output_folder)

# Filter label files with exactly 572 lines and copy corresponding images
filter_labels_and_copy(input_folder, output_folder, a, b)

# Split data into train and validation sets
split_train_val(output_folder)

import cv2

def draw_labels_on_single_image(image_path, label_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    height, width, _ = image.shape

    # Read the label file and draw the boxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id, x_center, y_center, box_width, box_height = map(float, data)

            # Convert YOLO format (normalized) to pixel values
            x_center_pixel = int(x_center * width)
            y_center_pixel = int(y_center * height)
            box_width_pixel = int(box_width * width)
            box_height_pixel = int(box_height * height)

            # Calculate top-left and bottom-right coordinates
            x1 = int(x_center_pixel - box_width_pixel / 2)
            y1 = int(y_center_pixel - box_height_pixel / 2)
            x2 = int(x_center_pixel + box_width_pixel / 2)
            y2 = int(y_center_pixel + box_height_pixel / 2)

            # Draw the bounding box and label
            color = (0, 255, 0)  # Green for the box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_text = f"Class {int(class_id)}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)
    print(f"Saved image with labels: {output_path}")

# Example usage
image_path = "data_part2/images/val/IMG_1584_iter_1.jpg"
label_path = "data_part2/labels/val/IMG_1584_iter_1.txt"
output_path = "output_image.jpg"

draw_labels_on_single_image(image_path, label_path, output_path)

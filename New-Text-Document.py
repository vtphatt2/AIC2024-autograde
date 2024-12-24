import cv2
import matplotlib.pyplot as plt
import numpy as np

original_image = cv2.imread(img_path)

image_height, image_width = original_image.shape[:2]

colors = [
    (255, 0, 0),    # Đỏ
    (0, 255, 0),    # Xanh lá
    (0, 0, 255),    # Xanh dương
    (255, 255, 0),  # Vàng
    (255, 0, 255),  # Hồng
    (0, 255, 255),  # Xanh ngọc
    (128, 0, 0),    # Đỏ đậm
    (0, 128, 0),    # Xanh lá đậm
    (0, 0, 128),    # Xanh dương đậm
    (128, 128, 0),  # Ô liu
    (128, 0, 128),  # Tím
    (0, 128, 128),  # Xanh cổ vịt
    (192, 192, 192),# Xám sáng
    (128, 128, 128),# Xám
    (64, 64, 64),   # Xám đậm
    (255, 165, 0)   # Cam
]

image_with_boxes = original_image.copy()

last_16_boxes = sorted_boxes[-16:]

for idx, box in enumerate(last_16_boxes):
    x_center = int(box[0] * image_width)
    y_center = int(box[1] * image_height)
    box_width = int(box[2] * image_width)
    box_height = int(box[3] * image_height)
    
    x_min = x_center - box_width // 2
    y_min = y_center - box_height // 2
    x_max = x_center + box_width // 2
    y_max = y_center + box_height // 2
    
    cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), colors[idx], 2)

image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

# Hiển thị ảnh
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Image with Last 16 Boxes in Different Colors")
plt.show()

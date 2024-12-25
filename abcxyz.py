import numpy as np

# Giả sử tensor ban đầu là một mảng NumPy, ví dụ:
tensor = np.array([
    [50, 50, 100, 100],  # Ô ở hàng 2, cột 1
    [10, 10, 60, 60],    # Ô ở hàng 1, cột 1
    [70, 10, 120, 60],   # Ô ở hàng 1, cột 2
    [10, 70, 60, 120],   # Ô ở hàng 2, cột 2
])

# Sắp xếp trước tiên theo tung độ (y1), sau đó theo hoành độ (x1)
sorted_tensor = tensor[np.lexsort((tensor[:, 0], tensor[:, 1]))]

print("Tensor đã sắp xếp:")
print(sorted_tensor)

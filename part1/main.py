import imutils
import numpy as np
import cv2
from math import ceil
from collections import defaultdict

def get_x(s):
    return s[1][0]

def get_y(s):
    return s[1][1]

def get_h(s):
    return s[1][3]

def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]

def crop_image(img):
    # chuyển đổi hình ảnh từ BGR sang GRAY để áp dụng thuật toán phát hiện cạnh Canny
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # loại bỏ nhiễu bằng cách làm mờ hình ảnh
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # áp dụng thuật toán phát hiện cạnh Canny
    img_canny = cv2.Canny(blurred, 100, 200)

    # tìm các đường viền
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # đảm bảo rằng ít nhất một đường viền được tìm thấy
    if len(cnts) > 0:
        # sắp xếp các đường viền theo kích thước giảm dần
        cnts = sorted(cnts, key=get_x_ver1)

        # lặp qua các đường viền đã sắp xếp
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:
                # kiểm tra sự chồng lấn của các đường viền
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # nếu danh sách các khối trả về còn trống
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # cập nhật tọa độ (x, y) và (chiều cao, chiều rộng) của các đường viền đã thêm
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # cập nhật tọa độ (x, y) và (chiều cao, chiều rộng) của các đường viền đã thêm
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

        # sắp xếp lại các khối trả về theo tọa độ x
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        
        # Vẽ các đường viền với độ dày tùy chỉnh (ví dụ: 5 pixels)
        for contour in cnts:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)  # Độ dày đường viền là 5 pixels

        return sorted_ans_blocks

# Tải hình ảnh (thay 'image_path' bằng đường dẫn hình ảnh của bạn)
img = cv2.imread('../img_with_box.jpg')

# Gọi hàm crop_image
cropped_blocks = crop_image(img)

# Hiển thị các khối hình ảnh bị cắt
for i, (block, _) in enumerate(cropped_blocks):
    cv2.imshow(f'Khối {i + 1}', block)

# Chờ nhấn phím để đóng cửa sổ hình ảnh
cv2.waitKey(0)
cv2.destroyAllWindows()

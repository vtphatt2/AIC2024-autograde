from ultralytics import YOLO
import numpy as np
import os
import cv2
import utlis

def convert_data(result):

    boxes = result.boxes.xyxy.tolist()

    labels = result.boxes.cls.tolist()

    confs = result.boxes.conf.tolist()
    datas = {}
    for box, label, conf in zip(boxes, labels, confs):
        # Chuyển box thành danh sách và mở rộng lên 8 phần tử
        extended_box = list(box)  # Chuyển sang danh sách nếu box là tuple
        # Thêm nhãn và các giá trị mặc định
        extended_box.extend([int(label), 0, 0, conf])

        # Đảm bảo luôn có 8 phần tử
        datas[tuple(extended_box[:4])] = list(tuple(extended_box[4:]))

    return datas


def find_cor_min_max(datas):
    val_x = [float("inf"), float("-inf")]  # [min_x, max_x]
    val_y = [float("inf"), float("-inf")]  # [min_y, max_y]

    for data_x in datas:
        if data_x[0] < val_x[0]:
            val_x[0] = data_x[0]
        if data_x[2] > val_x[1]:
            val_x[1] = data_x[2]

        if data_x[1] < val_y[0]:
            val_y[0] = data_x[1]
        if data_x[3] > val_y[1]:
            val_y[1] = data_x[3]

    return val_x, val_y


def update_row_col(datas):
    val_x, val_y = find_cor_min_max(datas)
    val = ((val_x[1] - val_x[0])/36, (val_y[1] - val_y[0])/10)

    for data in datas.keys():
        for i in range(0, 37):
            if (data[0] + data[2])/2 >= val_x[0] + val[0]*i and (data[0] + data[2])/2 <= val_x[0] + val[0]*(i+1):
                datas[data][1] = i + 1

                break

    for data in datas.keys():
        for i in range(0, 10):

            if (data[1] + data[3])/2.0 >= val_y[0] + val[1]*i and (data[1] + data[3])/2.0 <= val_y[0] + val[1]*(i+1):
                datas[data][2] = i + 1
                break


def convert_matrix(datas):
    matrix = np.zeros((11, 38), dtype=dict)
    for data in datas.keys():
        row_index = datas[data][2]
        col_index = datas[data][1]
        if matrix[row_index][col_index] == 0:
            matrix[row_index][col_index] = data
        

    return matrix


def update(results_end, index_x, index_y, start, datas, matrix):

    if matrix[index_x][index_y] == 0:
        return
    if results_end[start] is None or (datas[matrix[index_x][index_y]][0] == 0 and datas[matrix[index_x][index_y]][3] > datas[results_end[start]][3]):
        results_end[start] = matrix[index_x][index_y]

    if results_end[start] is None and datas[matrix[index_x][index_y]][0] == 0:
        results_end[start] = matrix[index_x][index_y]

    if datas[matrix[index_x][index_y]][0] == 1 and (results_end[start] is None):
        results_end[start] = matrix[index_x][index_y]

    if datas[matrix[index_x][index_y]][0] == 0 and datas[results_end[start]][0] == 1: # and datas[matrix[index_x][index_y]][3] > datas[results_end[start]][3]:
        results_end[start] = matrix[index_x][index_y]

    if datas[matrix[index_x][index_y]][0] == 1 and datas[results_end[start]][0] == 1 and datas[matrix[index_x][index_y]][3] < datas[results_end[start]][3]:
        results_end[start] = matrix[index_x][index_y]

def create_result(results_end, size, datas):

    matrix = convert_matrix(datas)
    for i in range(1, 11):
        for j in range(0, 8):
            update(results_end, i, j, size, datas, matrix)
        
        for j in range(10, 18):
            update(results_end, i, j, size+10, datas, matrix)

        for j in range(20, 28):
            update(results_end, i, j, size+20, datas, matrix)

        for j in range(30, 38):
            update(results_end, i, j, size+30, datas, matrix)

        size += 1


model_path = 'best_part1.pt'
model_part1 = YOLO(model_path)
# img_path = 'image.png'
# results = model_part1(img_path)
output_file = os.path.join('submission', 'results_part1.txt')


def process_and_write_to_file(results, image_path, WIDTH , HEIGHT):
    datas = convert_data(results)
    update_row_col(datas)
    num = 1
    results_end = [None] * 41
    create_result(results_end, num, datas)
    ticked_boxes = {}
    for i in range(1, 41):
        if results_end[i] is not None:

            centerx, centery = (results_end[i][0] + results_end[i][2]) / 2, (results_end[i][1] + results_end[i][3]) / 2
            w, h = results_end[i][2] - results_end[i][0], results_end[i][3] - results_end[i][1]

            w, h = w / WIDTH, h / HEIGHT

            ticked_boxes["1." + str(i)] = [centerx , centery , w, h]
    return ticked_boxes



# Tìm 4 chấm đen
def get_points(img):
    copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.blur(gray, (3, 3), 1)
    blur = cv2.blur(blur, (5, 5), 3)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
    canny = cv2.Canny(thresh, 50, 50)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square = []
    tmpcompy = copy.copy()
    for contour in contours:

        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if (area < 100):
            continue
        if (len(approx) == 4):
            tmpImg = utlis.getTransform(thresh, approx)
            
            if min(tmpImg.shape[0], tmpImg.shape[1]) / max(tmpImg.shape[0], tmpImg.shape[1]) < 0.7:
                continue

            percent = cv2.countNonZero(tmpImg) / (tmpImg.shape[0] * tmpImg.shape[1]) * (tmpImg.shape[0] + tmpImg.shape[1]) * (min(tmpImg.shape[0], tmpImg.shape[1]) / max(tmpImg.shape[0], tmpImg.shape[1]))
            square.append([utlis.Get_Conner_Points(contour), percent])

            tmpcompy = cv2.drawContours(tmpcompy, [approx], -1, (0, 255, 0), 2)

    square = sorted(square, key=lambda x: x[1], reverse=True)
    square = square[:8]

    cv2.imwrite('tmp.jpg', tmpcompy)
    cv2.imwrite('tmp2.jpg', thresh)

    original = img.copy()

    def get_area(A, B, C, D):


        A = [(A[0][0][0] + A[1][0][0] + A[2][0][0] + A[3][0][0]) // 4, (A[0][0][1] + A[1][0][1] + A[2][0][1] + A[3][0][1]) // 4]
        B = [(B[0][0][0] + B[1][0][0] + B[2][0][0] + B[3][0][0]) // 4, (B[0][0][1] + B[1][0][1] + B[2][0][1] + B[3][0][1]) // 4]
        C = [(C[0][0][0] + C[1][0][0] + C[2][0][0] + C[3][0][0]) // 4, (C[0][0][1] + C[1][0][1] + C[2][0][1] + C[3][0][1]) // 4]
        D = [(D[0][0][0] + D[1][0][0] + D[2][0][0] + D[3][0][0]) // 4, (D[0][0][1] + D[1][0][1] + D[2][0][1] + D[3][0][1]) // 4]

        res = utlis.reorder2(np.array([A, B, C, D]))
        area = cv2.contourArea(res)

        return [res, area]


    res = [[], 0]

    n = len(square)

    for i in range(0, n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for t in range(k + 1, n):
                    A = square[i][0]
                    B = square[j][0]
                    C = square[k][0]
                    D = square[t][0]

                    # print(i, j, k, t)

                    temp = get_area(A, B, C, D)
                    # print(temp[1])
                    # print(temp[0])
                    if (temp[1] > res[1]):
                        res = temp
                        # print(res[0], res[1])

    rectangle_img = cv2.drawContours(original, [res[0]], -1, (0, 255, 0), 2)
    cv2.imwrite('rectangle.jpg', rectangle_img)

    reorder = utlis.reorder(res[0])
    coordinates = [list(reorder[0][0]), list(reorder[1][0]), list(reorder[2][0]), list(reorder[3][0])]
    return np.array(coordinates)

# Tính góc cần xoay lại
def calculate_angle(points):
    vector = points[1] - points[0]
    # Tính góc alpha (độ) so với trục hoành
    alpha = np.arctan2(vector[1], vector[0]) * 180 / np.pi
    return alpha

# Tính ma trận để xoay ảnh
def rotate_image(image, center, alpha):
    # Tính tâm O của hình chữ nhật
    h, w = image.shape[:2]
    # Tạo ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), alpha, 1.0)
    # Xoay ảnh
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated_image, rotation_matrix

def draw_debug(image, x, y, w, h):

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    resized = cv2.resize(image, (int(image.shape[1] / 3), int(image.shape[0] / 3)))
    cv2.imshow('debug', resized)
    cv2.waitKey(0)

def submit_part1(testset_image_files):
    print("hello")
    with open('./submission/results_part1.txt', 'w') as f:
        print(len(testset_image_files))
        for image_file in testset_image_files:
            #results = model_part1(image_file)

            print(os.path.basename(image_file))
                
            
            img = cv2.imread(image_file)
            width = img.shape[1]
            height = img.shape[0]

            
            points = get_points(img)

            alpha = calculate_angle(points)
            center = np.mean(points, axis=0)


            rotated_image, rotation_matrix = rotate_image(img, center, alpha)
            #img, _ = rotate_image(rotated_image, center, -alpha)
            # cv2.imshow('rotated', rotated_image)
            cv2.imwrite('rotated.jpg', rotated_image)
            cv2.imwrite('beforrotated.jpg', img)
            results = model_part1(rotated_image)

            resp1 = process_and_write_to_file(results[0], image_file, width, height)

            
            rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle=-alpha, scale=1.0)

            outputline = ''

            for key, value in resp1.items():
                #print(key, value)
                nparr = np.array(value)
                coor = np.array(value[:2])
                
                coor = np.append(coor, 1)
                coor = coor @ rotation_matrix.T

                # draw_debug(img, int(coor[0] - value[2] * width / 2), int(coor[1] - value[3] * height / 2), int(value[2] * width), int(value[3] * height))


                value[0] = coor[0] / width
                value[1] = coor[1] / height

                outputline += f'{key} {value[0]:6f},{value[1]:6f},{value[2]:6f},{value[3]:6f} '
            filename = os.path.basename(image_file)

            line = filename + ' ' + outputline + '\n'
            # print(line)
            f.write(line)


# submit_part1(['beforrotated.jpg'])

from glob import glob

testset_image_files = glob(os.path.join('testset2', 'images', '*.jpg'))
testset_image_files.sort()


os.makedirs('submission', exist_ok=True)
# submit_sbd_mdt(testset_image_files)
submit_part1(testset_image_files)
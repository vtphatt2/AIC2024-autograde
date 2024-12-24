from ultralytics import YOLO
import numpy as np
import os
import cv2


def convert_data(result):

    boxes = result.boxes.xyxy.tolist()

    labels = result.boxes.cls.tolist()

    confs = result.boxes.conf.tolist()
    datas = {}
    for box, label, conf in zip(boxes, labels, confs):
        # Chuyển box thành danh sách và mở rộng lên 8 phần tử
        extended_box = list(box)  # Chuyển sang danh sách nếu box là tuple
        extended_box.extend([int(label), 0, 0, conf])  # Thêm nhãn và các giá trị mặc định
        
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
        for i in range(0,37):
            if (data[0] + data[2])/2 >= val_x[0] + val[0]*i and (data[0] + data[2])/2 <= val_x[0] + val[0]*(i+1):
                datas[data][1] = i + 1

                break

    
    for data in datas.keys():
        for i in range(0,10):
            
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
        else:
            if datas[matrix[row_index][col_index]][0] == 0 and datas[data][0] == 0 and datas[matrix[row_index][col_index]][3] < datas[data][3]:
                matrix[row_index][col_index] = data
            
            elif datas[matrix[row_index][col_index]][0] == 1 and datas[data][0] == 1 and datas[matrix[row_index][col_index]][3] > datas[data][3]:
                matrix[row_index][col_index] = data
                
            elif datas[matrix[row_index][col_index]][0] == 1 and datas[data][0] == 0:
                matrix[row_index][col_index] = data
            
    return matrix


def update(results_end,index_x ,index_y, start, datas, matrix):

    if matrix[index_x][index_y] == 0:
        return
    if results_end[start] is None or (datas[matrix[index_x][index_y]][0] == 1 and datas[matrix[index_x][index_y]][3] < datas[results_end[start]][3]):
        results_end[start] = matrix[index_x][index_y]
            
    if datas[matrix[index_x][index_y]][0] == 1 and (results_end[start] is None or datas[results_end[start]][0] == 0):
        results_end[start] = matrix[index_x][index_y]
    if datas[matrix[index_x][index_y]][0] == 0 and datas[results_end[start]][0] == 0 and datas[matrix[index_x][index_y]][3] > datas[results_end[start]][3]:
        results_end[start] = matrix[index_x][index_y]

def create_result(results_end, size, datas):
    
    matrix = convert_matrix(datas)
    for i in range(1, 11):
        for j in range(1, 8):
            update(results_end, i, j, size, datas, matrix)

        for j in range(11,18):
            update(results_end, i, j, size+10, datas, matrix)

        for j in range(21, 28):
            update(results_end, i, j, size+20, datas, matrix)
        
        for j in range(31, 38):
            update(results_end, i, j, size+30, datas, matrix)
        
        size += 1

model_path = 'best_part1.pt'
model_part1 = YOLO(model_path)
# img_path = 'image.png'
# results = model_part1(img_path)
output_file = os.path.join('submission', 'results_part1.txt')
def process_and_write_to_file(results, image_path):
    WIDTH = 2255
    HEIGHT = 3151
    datas = convert_data(results)
    update_row_col(datas)
    num = 0
    results_end = [None] * 41
    create_result(results_end, num, datas)
    ticked_boxes = {}
    for i in range(1, 41):
        if results_end[i] is not None:
            ticked_boxes["1." + str(i)] = f'{results_end[i][0]/WIDTH:6f},{results_end[i][1]/HEIGHT:6f},{results_end[i][2]/WIDTH:6f},{results_end[i][3]/HEIGHT:6f}'

    return ticked_boxes


def submit_part1(testset_image_files):
    print("hello")
    with open('./submission/results_part1.txt', 'w') as f:
        print(len(testset_image_files))
        for image_file in testset_image_files:
            results = model_part1(image_file)
            # print(results)
            line = image_file.split('\\')[-1] + ' ' + ' '.join([f'{k} {v}' for k, v in process_and_write_to_file(results[0], image_file).items()]) + '\n'
            # print(line)
            f.write(line)

submit_part1(['image.png'])

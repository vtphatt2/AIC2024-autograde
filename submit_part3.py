import os
from ultralytics import YOLO
import cv2
import numpy as np

model_path = 'best_part3.pt'
model = YOLO(model_path)

def process(img_path):
    results = model(img_path)
    result = results[0]
    boxes = result.boxes.xywhn
    labels = result.boxes.cls
    conf = result.boxes.conf

    centers = np.column_stack((boxes[:, 0], boxes[:, 1]))
    sorted_indices = np.argsort(centers[:, 0])

    sorted_boxes = result.boxes[sorted_indices].xywhn
    sorted_labels = result.boxes[sorted_indices].cls
    sorted_conf = result.boxes[sorted_indices].conf

    # xóa các ô bị lặp nếu có
    delete_idx = []
    for i in range(len(sorted_boxes)-1):
        x1, y1, w, h = sorted_boxes[i]
        x2, y2, w, h = sorted_boxes[i+1]
        if abs(x2 - x1) < 0.01 and abs(y2 - y1) < 0.01:
            delete_idx.append(i+1)

    sorted_boxes = np.delete(sorted_boxes, delete_idx, axis=0)
    sorted_labels = np.delete(sorted_labels, delete_idx, axis=0)
    sorted_conf = np.delete(sorted_conf, delete_idx, axis=0)

    final_sorted_boxes = []
    i = 0
    while i < len(sorted_boxes):
        if i%43 == 33:
            step = 10
        else:
            step = 11
            
        cluster_boxes = sorted_boxes[i:i + step]
        cluster_labels = sorted_labels[i:i + step]
        cluster_conf = sorted_conf[i:i + step]

        cluster_sorted_boxes = cluster_boxes[np.argsort(cluster_boxes[:, 1])]
        cluster_sorted_labels = cluster_labels[np.argsort(cluster_boxes[:, 1])]
        cluster_sorted_conf = cluster_conf[np.argsort(cluster_boxes[:, 1])]

        sorted_boxes[i:i + step] = cluster_sorted_boxes
        sorted_labels[i:i + step] = cluster_sorted_labels
        sorted_conf[i:i + step] = cluster_sorted_conf

        i += step

    string = ""

    for i in range(6):
        string += "3." + str(i+1) + " "
        for k in range(4):
            step = 11
            if k == 3: step = 10

            ticked_idx = [j for j in range(i*43+k*11, i*43+k*11+step) if sorted_labels[j] == 0.]
            if len(ticked_idx) == 1:
                string += str(sorted_boxes[ticked_idx[0]])[8:-2].replace(" ","") + " "
                # string += str(ticked_idx[0]+1) + " "
            elif len(ticked_idx) == 0:
                min_conf_idx = np.argmin(sorted_conf[i*43+k*33:i*43+k*33+step]) + i*43+k*11
                string += str(sorted_boxes[min_conf_idx])[8:-2].replace(" ","") + " "
                # string += str(min_conf_idx+1) + " "
            else:
                max_conf = 0
                max_conf_idx = i*43+k*11
                for j in ticked_idx:
                    if sorted_conf[j] > max_conf:
                        max_conf = sorted_conf[j]
                        max_conf_idx = j
                string += str(sorted_boxes[max_conf_idx])[8:-2].replace(" ","") + " "
                # string += str(max_conf_idx+1) + " "

    return string[:-1]

def submit_part3(testset_image_files):
    with open('./submission/results_part3.txt', 'w') as f:
        # i = 0
        for image_file in testset_image_files:
            # i += 1
            # if i == 10: break
            try:
                line = image_file.split('\\')[-1] + ' ' + process(image_file) + '\n'
            except:
                line = ""
            f.write(line)

from glob import glob 

testset_image_files = glob(os.path.join('testset1', 'images', '*.jpg'))
testset_image_files.sort()

submit_part3(testset_image_files)
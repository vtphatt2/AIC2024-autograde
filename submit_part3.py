import os
from ultralytics import YOLO
import cv2
import numpy as np
import random
import utlis
from glob import glob

model_path = 'best_part3.pt'
model = YOLO(model_path)

def stackImages(imgArray,scale=1,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver


def rectContour(contours):

    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 75:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def Get_Conner_Points(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("add", add)
    # print("myPoints", myPoints)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getTransform(img, contour):
    tmpContour = reorder(contour)
    pts1 = np.float32(tmpContour)

    def dist(a, b):
        return np.round(((a[0] - b[0]) ** 2 +  (a[1] - b[1]) ** 2) ** 0.5)

    widthImg = int(dist(pts1[0][0], pts1[1][0]))
    heightImg = int(dist(pts1[0][0], pts1[2][0]))

    # print(widthImg, heightImg)
    # widthImg = 300
    # heightImg = 600

    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    return imgOutput


def drawGrid(img,questions=5,choices=5):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)
    for i in range (0, secH):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 0, 0),2)
        cv2.line(img, pt3, pt4, (255, 0, 0),2)

    return img


def splitBoxes(img, vsplit=5, hsplit=5):
    rows = np.vsplit(img, vsplit)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r, hsplit)
        for box in cols:
            boxes.append(box)
            # print(cv2.countNonZero(box), end=' ')
        # print()
    return boxes



def reorder2(myPoints):
    def sort_by_first_then_second(x):
        return (x[0], x[1])
    myPoints = sorted(myPoints, key=sort_by_first_then_second)
    firstPoint = myPoints[0]

    res = [firstPoint]
    tmp = []
    for i in range(1, 4):
        a = myPoints[i][0] - firstPoint[0]
        b = myPoints[i][1] - firstPoint[1]

        if a == 0:
            tmp.append([10000000000, i])
        else:
            tmp.append([b / a, i])

    tmp = sorted(tmp, key=lambda x: x[0])

    for p in tmp:
        res.append(myPoints[p[1]])

    res = np.array(res)

    return res


def getTransformFix(img, contour, widthImg=1448, heightImg=2136):
    tmpContour = reorder(contour)
    pts1 = np.float32(tmpContour)
    # widthImg = 1448
    # heightImg = 2136

    # print(widthImg, heightImg)
    # widthImg = 300
    # heightImg = 600

    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    return imgOutput

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

def process(path):
    img = cv2.imread(path)
    points = get_points(img)

    # print(points)

    alpha = calculate_angle(points)
    center = np.mean(points, axis=0)

    rotated_image, rotation_matrix = rotate_image(img, center, alpha)

    results = model(rotated_image)
    result = results[0]
    boxes = result.boxes.xywh.cpu()
    labels = result.boxes.cls.cpu()
    conf = result.boxes.conf.cpu()

    # print(boxes.size())

    centers = np.column_stack((boxes[:, 0], boxes[:, 1]))
    sorted_indices = np.argsort(centers[:, 0])

    sorted_boxes = result.boxes[sorted_indices].xywh.cpu()
    sorted_labels = result.boxes[sorted_indices].cls.cpu()
    sorted_conf = result.boxes[sorted_indices].conf.cpu()

    # xóa các ô bị lặp nếu có
    delete_idx = []
    for i in range(len(sorted_boxes)-1):
        x1, y1, w, h = sorted_boxes[i]
        x2, y2, w, h = sorted_boxes[i+1]
        if abs(x2 - x1) < 0.005 and abs(y2 - y1) < 0.005:
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

    boxes = sorted_boxes[:, :2]
    boxes = np.hstack([boxes, np.ones((boxes.shape[0], 1))])
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle=-alpha, scale=1.0)
    rotated_boxes = boxes @ rotation_matrix.T
    sorted_boxes = np.hstack([rotated_boxes, sorted_boxes[:, 2:]])

    image_height, image_width = img.shape[:2]
    sorted_boxes[:,0] /= image_width
    sorted_boxes[:,2] /= image_width
    sorted_boxes[:,1] /= image_height
    sorted_boxes[:,3] /= image_height

    sorted_boxes = sorted_boxes.round(6)

    string = ""

    for i in range(6):
        string += "3." + str(i+1) + " "
        for k in range(4):
            step = 11
            if k == 3: step = 10

            ticked_idx = [j for j in range(i*43+k*11, i*43+k*11+step) if sorted_labels[j] == 0.]
            if len(ticked_idx) == 1:
                string += str(list(sorted_boxes[ticked_idx[0]]))[1:-2].replace(" ","") + " "
                # string += str(ticked_idx[0]+1) + " "
            elif len(ticked_idx) == 0:
                min_conf_idx = np.argmin(sorted_conf[i*43+k*33:i*43+k*33+step]) + i*43+k*11
                string += str(list(sorted_boxes[min_conf_idx]))[1:-2].replace(" ","") + " "
                # string += str(min_conf_idx+1) + " "
            else:
                max_conf = 0
                max_conf_idx = i*43+k*11
                for j in ticked_idx:
                    if sorted_conf[j] > max_conf:
                        max_conf = sorted_conf[j]
                        max_conf_idx = j
                string += str(list(sorted_boxes[max_conf_idx]))[1:-2].replace(" ","") + " "
                # string += str(max_conf_idx+1) + " "

    return string[:-1]

# process("/kaggle/input/testset2/images/IMG_0.jpg")

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

testset_image_files = glob(os.path.join('testset2', 'images', '*.jpg'))
testset_image_files.sort()


os.makedirs('submission', exist_ok=True)
# submit_sbd_mdt(testset_image_files)
submit_part3(testset_image_files)
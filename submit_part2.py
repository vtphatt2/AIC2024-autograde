from ultralytics import YOLO
from glob import glob 
import os
import cv2
import numpy as np
import utlis

model_part2 = YOLO('best_part2.pt')
output_file = os.path.join('submission', 'results_part2.txt')

def get_sbd(boxes, img):
    boxes = sorted(boxes, 
                   key=lambda x: {x['box'][0], x['box'][1]}, 
                   )
    
    result = []
    
    for i in range(0, 6):
        ticks = []
        col = boxes[i * 10 : (i + 1) * 10]
        col = sorted(col, key=lambda x: x['box'][1])
        print(len(col))
        for j in range(0, 10):
            if col[j]['label'] == 0:
                ticks.append(j)
                # print(boxes[j])
        print(ticks)

        if (len(ticks) == 1):
            print(f'SBD{i + 1}')
            result.append({f'SBD{i + 1}' : col[0]})
            continue

        if len(ticks) == 0:
            minConf = 10
            resId = 0
            for j in range(0, 10):
                if col[j]['conf'] < minConf:
                    minConf = col[j]['conf']
                    resId = j
        else:
            maxConf = 0
            resId = 0
            for j in ticks:
                if col[j]['conf'] > maxConf:
                    maxConf = col[j]['conf']
                    resId = j

        print(f'SBD{i + 1}')
        result.append({f'SBD{i + 1}' : col[resId]})

    return result
        

def get_mdt(boxes, img):
    boxes = sorted(boxes, 
                   key=lambda x: {x['box'][0], x['box'][1]}, 
                   )
    
    result = []
    
    for i in range(0, 3):
        ticks = []
        col = boxes[i * 10 : (i + 1) * 10]
        col = sorted(col, key=lambda x: x['box'][1])
        print(len(col))
        for j in range(0, 10):
            if col[j]['label'] == 0:
                ticks.append(j)
                # print(boxes[j])
        print(ticks)

        if (len(ticks) == 1):
            print(f'MDT{i + 1}')
            result.append({f'MDT{i + 1}' : col[0]})
            continue

        if len(ticks) == 0:
            minConf = 10
            resId = 0
            for j in range(0, 10):
                if col[j]['conf'] < minConf:
                    minConf = col[j]['conf']
                    resId = j
        else:
            maxConf = 0
            resId = 0
            for j in ticks:
                if col[j]['conf'] > maxConf:
                    maxConf = col[j]['conf']
                    resId = j

        print(f'MDT{i + 1}')
        result.append({f'MDT{i + 1}' : col[resId]})

    return result
        

def to_grid(boxes, numcol):
    left = 10 ** 9
    right = 0
    sumW = 0

    for box in boxes:
        x, y, w, h = box['box']

        x -= w / 2
        left = min(left, x)

        x += w
        right = max(right, x)

        sumW += w

    avgW = (right - left) / numcol

    result = []

    for i in range(0, numcol):
        # result.append([])
        col = []
        l = left + i * avgW
        r = left + (i + 1) * avgW
        for box in boxes:
            x, y, w, h = box['box']
            if x >= l and x <= r:
                col.append(box)
        
        result.append(col)

    return result

def to_grid_row(boxes, numrow):
    top = 10 ** 9
    bot = 0

    for box in boxes:
        x, y, w, h = box['box']

        y -= h / 2
        top = min(top, y)

        y += h
        bot = max(bot, y)


    avgW = (bot - top) / numrow

    result = []

    for i in range(0, numrow):
        # result.append([])
        col = []
        l = top + i * avgW
        r = top + (i + 1) * avgW
        for box in boxes:
            x, y, w, h = box['box']
            if y >= l and y <= r:
                col.append(box)
        
        result.append(col)

    return result

def get_ticked(col, top, bot, numRow):
    resid = 0

    ticked = []
    unticked = []

    avgH = (bot - top) / numRow
    if (len(col) == 0):
        # print('empty')
        return 0, None
    for i in range(0, numRow):
        t = top + i * avgH
        b = top + (i + 1) * avgH

        ls = []

        for j in range(0, len(col)):
            if col[j]['box'][1] >= t and col[j]['box'][1] <= b:
                if col[j]['label'] == 0:
                    ticked.append([i, col[j]['box'], col[j]['conf']])
                else:
                    unticked.append([i, col[j]['box'], col[j]['conf']])

    if len(ticked) > 0:
        mxConf = 0
        resid = 0
        
        bx = None

        for i in range(0, len(ticked)):
            if ticked[i][2] > mxConf:
                mxConf = ticked[i][2]
                resid = ticked[i][0]
                bx = ticked[i][1]

        return resid, bx
    
    mnConf = 10
    resid = 0

    bx = None

    for i in range(0, len(unticked)):
        if unticked[i][2] < mnConf:
            mnConf = unticked[i][2]
            resid = unticked[i][0]
            bx = unticked[i][1]

    return resid, bx

def get_ticked_row(col, left, right, numCol):
    resid = 0

    ticked = []
    unticked = []

    avgH = (right - left) / numCol
    if (len(col) == 0):
        print('empty')
        return 0, None
    for i in range(0, numCol):
        t = left + i * avgH
        b = left + (i + 1) * avgH

        ls = []

        for j in range(0, len(col)):
            if col[j]['box'][0] >= t and col[j]['box'][0] <= b:
                if col[j]['label'] == 0:
                    ticked.append([i, col[j]['box'], col[j]['conf']])
                else:
                    unticked.append([i, col[j]['box'], col[j]['conf']])

    if len(ticked) > 0:
        mxConf = 0
        resid = 0
        
        bx = None

        for i in range(0, len(ticked)):
            if ticked[i][2] > mxConf:
                mxConf = ticked[i][2]
                resid = ticked[i][0]
                bx = ticked[i][1]

        return resid, bx
    
    mnConf = 10
    resid = 0

    bx = None

    for i in range(0, len(unticked)):
        if unticked[i][2] < mnConf:
            mnConf = unticked[i][2]
            resid = unticked[i][0]
            bx = unticked[i][1]

    return resid, bx

def draw_debug_(img, col, name):
    for box in col:
        x, y, w, h = box['box']
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)

        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        if (box['label'] == 0):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(name, img)

def process_and_write_to_file(results, image_path):
    ticked_boxes = {
        'SBD1': '',
        'SBD2': '',
        'SBD3': '',
        'SBD4': '',
        'SBD5': '',
        'SBD6': '', 
        'MDT1': '',
        'MDT2': '',
        'MDT3': ''
    }

    # img = cv2.imread(image_path)
    WIDTH = 2255
    HEIGHT = 3151

    # print('dimension: ', WIDTH, HEIGHT)
    # for result in results:
    #     print(result.boxes)

    boxes = []

    # print(len(results.boxes))
    for box in results.boxes:
        boxes.append({
            'label': int(box.cls[0].item()),
            'box': box.xywh[0].tolist(),
            'conf': box.conf.item()
        })


    # print('len',len(boxes))
    
    top = 100000
    bot = 0

    sumH = 0

    for box in boxes:
        x, y, w, h = box['box']
        top = min(top, y - h / 2)
        bot = max(bot, y + h / 2)
        sumH += h

    avgH = sumH / len(boxes)
    
    grid = to_grid(boxes, 11)
    
    res_sbd_mdt = []
    for i in range(0, 6):
        col = grid[i]

        resid, bx = get_ticked(col, top, bot, 10)
        if bx == None:
            continue
        # res_sbd_mdt.append({f'SBD{i + 1}': {'box': bx, 'conf': boxes[resid]['conf']}})
        key = f'SBD{i + 1}'
        # print('box', bx)

        # print('get', key, resid)

        x = bx[0] / WIDTH
        y = bx[1] / HEIGHT
        w = bx[2] / WIDTH
        h = bx[3] / HEIGHT

        ticked_boxes[key] = f'{x:6f},{y:6f},{w:6f},{h:6f}'

    for i in range(8, 11):
        col = grid[i]

        resid, bx = get_ticked(col, top, bot, 10)
        if bx == None:
            continue
        # res_sbd_mdt.append({f'MDT{i - 7}': {'box': bx, 'conf': boxes[resid]['conf']}})
        key = f'MDT{i - 7}'
        
        # print('box', bx)
        # print('get', key, resid)

        x = bx[0] / WIDTH
        y = bx[1] / HEIGHT
        w = bx[2] / WIDTH
        h = bx[3] / HEIGHT

        ticked_boxes[key] = f'{x:6f},{y:6f},{w:6f},{h:6f}'


    return ticked_boxes

part2_checklist = ['T', 'x', 'F', 'x', 'T', 'x', 'F', 'x', 'x', 'T', 'x', 'F', 'x', 'T', 'x', 'F', 'x', 'x', 'T', 'x', 'F', 'x', 'T', 'x', 'F', 'x', 'x', 'T', 'x', 'F', 'x', 'T', 'x', 'F' ]
def part2_process(results, image_path, WIDTH = 2255, HEIGHT = 3151):
    ticked_boxes = {}
    for i in range(1, 9):
        ticked_boxes[f'2.{i}.a'] = None
        ticked_boxes[f'2.{i}.b'] = None
        ticked_boxes[f'2.{i}.c'] = None
        ticked_boxes[f'2.{i}.d'] = None

    boxes = []

    # print(len(results.boxes))
    for box in results.boxes:
        boxes.append({
            'label': int(box.cls[0].item()),
            'box': box.xywh[0].tolist(),
            'conf': box.conf.item()
        })

    
    top = 100000
    bot = 0

    sumH = 0

    for box in boxes:
        x, y, w, h = box['box']
        top = min(top, y - h / 2)
        bot = max(bot, y + h / 2)
        sumH += h

    avgH = sumH / len(boxes)
    
    # grid = to_grid(boxes, 34)
    grid_row = to_grid_row(boxes, 4)

    option = ['a', 'b', 'c', 'd']

    opt = 0
    for row in grid_row:
        column = to_grid(row, 34)

        curLetter = option[opt]
        opt += 1
        quest = 0

        curBox = []

        for i in range(0, 34):
            if part2_checklist[i] == 'T':
                curBox = []
                quest += 1
                for tmpBox in column[i]:
                    curBox.append(tmpBox)
                continue

            if part2_checklist[i] == 'F':
                for tmpBox in column[i]:
                    curBox.append(tmpBox)
                left = 100000000
                right = 0

                for box in curBox:
                    x, y, w, h = box['box']
                    x -= w / 2
                    left = min(left, x)
                    x += w
                    right = max(right, x)

                resid, bx = get_ticked_row(curBox, left, right, 2)
                # print(len(curBox))
                # resid, bx = get_ticked(col, top, right, 10)
                # if bx == None:
                #     continue
                # # res_sbd_mdt.append({f'MDT{i - 7}': {'box': bx, 'conf': boxes[resid]['conf']}})
                
                key = f'2.{quest}.{curLetter}'
                
                # # print('box', bx)
                # # print('get', key, resid)

                x = bx[0]
                y = bx[1]
                w = bx[2] / WIDTH
                h = bx[3] / HEIGHT

                ticked_boxes[key] = [x,y,w,h]

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

def submit_part2(testset_image_files):
    with open('./submission/results_part2.txt', 'w') as f:
        for image_file in testset_image_files:
            # if os.path.basename(image_file) != 'IMG_2184.jpg':
            #     continue

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
            results_part2 = model_part2(rotated_image)

            # skip when no data

            if len(results_part2[0].boxes) == 0:
                print(f'No detection in {image_file}')
                continue
            res_part2 = part2_process(results_part2[0], rotated_image, width, height)

            # if no detection


            rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle=-alpha, scale=1.0)

            outputline = ''

            for key, value in res_part2.items():
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




testset_image_files = glob(os.path.join('testset2', 'images', '*.jpg'))
testset_image_files.sort()


os.makedirs('submission', exist_ok=True)
# submit_sbd_mdt(testset_image_files)
submit_part2(testset_image_files)
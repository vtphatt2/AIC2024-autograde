from ultralytics import YOLO
import os

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
        # print('empty')
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
def part2_process(results, image_path):
    ticked_boxes = {}
    for i in range(1, 9):
        ticked_boxes[f'2.{i}.a'] = ''
        ticked_boxes[f'2.{i}.b'] = ''
        ticked_boxes[f'2.{i}.c'] = ''
        ticked_boxes[f'2.{i}.d'] = ''


    WIDTH = 2255
    HEIGHT = 3151

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

                x = bx[0] / WIDTH
                y = bx[1] / HEIGHT
                w = bx[2] / WIDTH
                h = bx[3] / HEIGHT

                ticked_boxes[key] = f'{x:6f},{y:6f},{w:6f},{h:6f}'

    return ticked_boxes

def submit_part2(testset_image_files):
    with open('./submission/results_part2.txt', 'w') as f:
        for image_file in testset_image_files:
            results_part2 = model_part2(image_file)
            res_part2 = part2_process(results_part2[0], image_file)

            line = image_file.split('\\')[-1] + ' ' + ' '.join([f'{k} {v}' for k, v in res_part2.items()]) + '\n'
            f.write(line)
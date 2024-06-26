import time
import cv2
import os
import torch
import pytesseract
import numpy as np
from collections import Counter
import glob
import re

# Settings 改這些東西就行
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # pytesseract安裝位置，基本上不用換
path = "影片資料集"  # 放影片的資料夾路徑
pt_path = 'best_weight/best_ex'  # 放參數檔的路徑 檔名都取best_ex4.pt、best_ex5.pt、best_ex6.pt ...
start_pt = 4  # 起始best_ex
end_pt = 9  # 結束best_ex


# 存放辨識的dict 用於投票多數決
def ocr_map(ocr_result, ocr_total_map):
    if ocr_result in ocr_total_map:
        ocr_total_map[ocr_result] += 1
    else:
        ocr_total_map[ocr_result] = 1
    return ocr_total_map

# 英文字母帶碼對照表
char_to_num = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
    'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
    'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
}

# 判斷貨櫃號碼 回傳 答案、狀態

def is_valid_container_number(container_number):
    container_number = re.sub(r'[^0-9A-Z]', '', container_number).replace(" ", "")
    print(container_number)
    # 不到11
    if len(container_number) < 11:
        print('不到11')
        return 'False', False
    # 前4不為英文
    if not all(letter.isalpha() for letter in container_number[:4]):
        print('前4不為英文')
        return 'False', False
    try:
        # 將前10個字符轉換為數值
        values = []
        for char in container_number[:10]:
            if char.isdigit():
                values.append(int(char))
            elif char.isalpha():
                values.append(char_to_num[char])
            else:
                return 'False', False  # 不合法字符
        
        # 計算公式 S
        S = sum(values[i] * (2 ** i) for i in range(10)) % 11
        # S的個位數為第11位數的值
        check_digit = S % 10

        # 檢查第11個字符是否等於計算出的 check_digit
        if check_digit == int(container_number[10]):
            return container_number[:11], True
        print('驗證錯誤')
        return 'False', False
    except (KeyError, ValueError):
        print('錯誤')
        return 'False', False


def video(video_path, model):

    ocr_total_map = {}
    frame_total_map = {}

    # 抓影片
    cap = cv2.VideoCapture(video_path)

    while True:
        # ret為true or false
        # frame為當前禎
        ret,frame=cap.read()
        if frame is None:
            break
        # 開始辨識 先圖片pp
        real_frame = frame
        # 同1中 RGB BGR?
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 同1
        alpha = 1
        beta = 5 
        frame = cv2.convertScaleAbs(frame, alpha, beta)
        # 對比與亮度 OpenCV
        contrast = -30
        brightness = 0
        frame = frame * (contrast/127 + 1) - contrast + brightness # 轉換公式
        frame = np.clip(frame, 0, 255)
        frame = np.uint8(frame)
        
        # 抓bb
        results = model(frame)
        if len(results.xyxyn[0]) > 0:
            # 0是當前圖片 4是信心度欄位 抓最大的索引
            f_index = results.xyxyn[0][:, 4].argmax()
            # 最大索引的結果
            f_result = results.xyxyn[0][f_index]
        else:
            continue
        
        width, height = frame.shape[1], frame.shape[0]

        xmin, ymin, xmax, ymax = int(f_result[0]*width), int(f_result[1]*height), int(f_result[2]*width), int(f_result[3]*height)
        # boundingbox
        bbox = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        CTNid = pytesseract.image_to_string(bbox)
        CTNid, id_stat = is_valid_container_number(CTNid)

        if id_stat == True:
            ocr_total_map = ocr_map(CTNid, ocr_total_map)
            frame_total_map[CTNid] = bbox
        
        detections = [['None', [0, 0], 0]]
        detections[0][0] = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        detections[0][1] = [xmin, ymin]
            
        print(CTNid)
        # 在图像上绘制文本
        cv2.putText(real_frame, str(CTNid), (detections[0][1][0], detections[0][1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.rectangle(real_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # BBox
            
        # 显示图像
        cv2.imshow('oxxostudio', real_frame) 
            
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 鍵退出迴圈
            break
    if ocr_total_map:    
        max_key = max(ocr_total_map, key=ocr_total_map.get)
    else:
        max_key = ''
    print("多數決結果:" + max_key)
    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite(f"./plate_crop/{max_key}.jpg", frame_total_map.get(max_key))
    return max_key

start_time = time.time()

avi_files = glob.glob(path + "/*.avi")
result_list = []
for i in range(start_pt, end_pt + 1):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'{pt_path}{i}.pt')  # 注意參數檔的路徑
    answer = 0
    fail = 0
    for avi_file in avi_files:
        # 在這裡處理每個 .avi 檔案
        text = video(avi_file, model)
        file_name = os.path.basename(avi_file)
        file_name_without_extension = os.path.splitext(file_name)[0]
        print(text + "正解 "+ file_name_without_extension)
        if text == file_name_without_extension:
            answer +=1
        else:
            fail +=1
    result_list.append((f'ex{i}', f'辨識正確:{answer}個影片', f'辨識失敗:{fail}個影片', f'準確率:{answer/(answer+fail)}'))
# 記錄結束時間
end_time = time.time()
# 計算執行時間
execution_time = end_time - start_time
print(f"程式執行時間為 {execution_time} 秒")
# 輸出結果
for l in result_list:
    for t in l:
        print(t, end=' ')
    print()

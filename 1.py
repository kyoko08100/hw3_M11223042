# 議題1

# 使用貨櫃號碼資料集內訓練和驗證集進行物件偵測的訓練，
# 之後透過測試資料集來檢測模型性能。隨後裁切物件偵測所抓取區域，
# 並透過文字辨識將該區域的貨櫃號碼辨識出來，
# 並查看辨識出來的號碼是否和圖片中的一致
import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import cv2
import re
# 改一下AAAAAAA
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# model 

model = torch.hub.load('yolov5', 'custom', path='/best/best.pt')


# main 

def ocr(imageP, gt):

    results = model(imageP)

    # label coord from model
    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    image = cv2.imread(imageP)
    # 轉灰   RGB? BGR? to GRAY
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 用於存放增強對比
    # 參 OpenCV
    alpha = 1
    beta = 5 
    image = cv2.convertScaleAbs(image, alpha, beta)
    # 對比與亮度 OpenCV
    contrast = -30
    brightness = 0
    image = image * (contrast/127 + 1) - contrast + brightness # 轉換公式
    # 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python
    # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
    # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    width, height = image.shape[1], image.shape[0]
    print(f'Photo width,height: {width},{height}. Detected container: {len(labels)}')

    plot = plt.subplots(1, 2)

    for i in range(len(labels)):
        row = coordinates[i]

        # row  0 1 2 3 4 = xmin ymin xmax ymax confidence 

        if row[4] >= 0.6:
            xmin, ymin, xmax, ymax = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
            # boundingbox
            bbox = image[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
            text = pytesseract.image_to_string(bbox)
            if text:
                cv2.rectangle(image, (xmin,ymin), (xmax, ymax), (0, 255, 0), 6) # boundingbox
                cv2.putText(image, f"{text}", (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                # BGR RGB GRAY?
                plot[0].imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

                # 擷取文字 把0~9、大寫字母、空格-
                text = re.sub(r'[^0-9A-Z]', '', text)
                text = text.replace(" ", "")
                text = text[:11]
                # BGR RGB GRAY?
                plot[1].imshow(cv2.cvtColor(bbox, cv2.COLOR_GRAY2RGB))

                print(f'YOLOv5:{row[4]:.2f}, pytesseract:{text}, gt:{gt}')
                if text == gt:
                    return 1 
                else:
                    return 0
            else:
                print("00000")
                return 0


# 準確率績效公式 辨識正確/測試資料

def accuracy(fail, true):
    return true / (fail+true)

# run
 
path = f"./test"
file_list = glob.glob(os.path.join(path, "*.jpg"))

fail = 0
true = 0

for file in file_list:
    file_name = os.path.splitext(os.path.basename(file))[0]
    result = ocr(file,file_name)
    if result == 0:
        fail+=1
    if result == 1:
        true+=1

print(true)
acc = accuracy(fail, true)
print("準確度:", acc)
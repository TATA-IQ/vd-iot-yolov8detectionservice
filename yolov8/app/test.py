import cv2
import requests
import random
import base64
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
import io

# vehicle_yolov8_url = "http://172.16.0.204:6500/classes/"
# vehicle_yolov8_url = "http://172.16.0.204:6500/detect/"
vehicle_yolov8_url = "http://172.16.0.178:6500/detect/"
vehicle_yolov8_url = "http://172.16.0.178:7000/detect/"
firesmoke_yolov8_url = "http://172.16.0.178:6501/detect/"
crowd_yolov8_url = "http://172.16.0.178:6502/detect/"
ppe_yolov8_url = "http://172.16.0.178:6503/detect/"
paddleocr_url = "http://172.16.0.204:6060/detect/"
garbage_url = "http://172.16.0.178:6505/detect/"
yolov8_test = "http://172.16.0.204:6055/detect/"


# img = cv2.imread(r"C:\Users\SridharBondla\Downloads\human_train1_jusco_till_31\aditya_2023_07_27_14_30_46_4572.jpg")
img = cv2.imread("test1.jpg")
# print(img.shape)
img_str = cv2.imencode(".jpg", img)[1].tobytes().decode("ISO-8859-1")

np_coord = {1:[150,95,455,145], 2:[590,385,950,450]} # xmin, ymin, xmax, ymax

ocr_query = {"image":img_str, "image_name":"image3.jpg", "np_coord":np_coord, "model_config":{}}
det_query = {
    'image': img_str,
    'image_name': 'image3.jpg',
    'camera_id': '123',
    'image_time': '2023-09-04 22:13:23.123456',
    'model_type': 'object detection',
    'model_framework': 'yolov8',
    'model_config': {
    'is_track':False,
    'conf_thres': 0.5,
    'iou_thres': 0.5,
    'max_det': 300,
    'agnostic_nms': True,
    'augment': False,
    
   },
    "split_columns": 2,
    "split_rows": 2}

r = requests.post(vehicle_yolov8_url, json=det_query)
# # r = requests.post(yolov8_test, json=det_query)
# data = r.json()
print(r.json())

detections = r.json()['data']['result']
print(len(detections))
if len(detections)>0:
    for a,i in enumerate(detections):
        img1 = cv2.rectangle(img,(i['xmin'],i['ymin']),(i["xmax"],i["ymax"]),(255,255,0),2)
        cv2.putText(img, i["class_name"], (i['xmin'],i['ymin']), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    cv2.imwrite("output/"+str(i["class_name"])+"_"+str(a)+".jpg",img1)
else:
    print("no detections")
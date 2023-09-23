import cv2
import requests
import random
import base64
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
import io
import time


# url = "http://172.16.0.178:7000/detect/"
url = "http://172.16.0.178:6500/detect/"

# img = cv2.imread("image3.jpg")
for i in range(40):
    img = cv2.imread("img1.png")
    # img = cv2.imread("Screenshot (105).png")
    # print(img.shape)
    img_str = cv2.imencode(".jpg", img)[1].tobytes().decode("ISO-8859-1")
    # print("*"*100)
    # print(type(img_str))

    stream = BytesIO(img_str.encode("ISO-8859-1"))
    image = Image.open(stream).convert("RGB")
    open_cv_image = np.array(image) 

    # query = {"image":img_str, "image_name":"test.jpg", "model_config":{"conf_thres":0.9,"iou_thres":0.9, "max_det":300, "agnostic_nms":True,"augment":False,}}
    # query = {"image":img_str, "image_name":"test.jpg", 'model_config': {'conf_thres': 0.1,
    #     'iou_thres': 0.1,
    #     'max_det': 300,
    #     'agnostic_nms': True,
    #     'augment': False,
    #     "classes":[0,1,2]}}
    query = {
        'image': img_str,
        'image_name': 'image3.jpg',
        'camera_id': '123',
        'image_time': '2023-09-04 22:13:23.123456',
        'model_type': 'object detection',
        'model_framework': 'yolov8',
        'model_config': {
        'is_track':True,
        'conf_thres': 0.1,
        'iou_thres': 0.1,
        'max_det': 300,
        'agnostic_nms': True,
        'augment': False,
        
    }}

    # # query={"model_name": "vehicle.zip","model_path": "/object_detection/usecase4/model_id_4/vehicle.zip",  "model_id": "model_id_4",  "model_framework": "yolov8",}

    r = requests.post(url, json=query)
    data = r.json()
    time.sleep(5)
    print(f"================={i+1}===============")
    print(r.json())
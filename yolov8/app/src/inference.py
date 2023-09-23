import os
import glob
import copy
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class InferenceModel:
    """
    Yolov8 inference
    """
    def __init__(self, model_path=None, gpu=False):
        """
        Initialize Yolov8 inference
        
        Args:
            model_path (str): path of the downloaded and unzipped model
            gpu=True, if the system have NVIDIA GPU compatibility
        """
        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.model_path = model_path
        self.model = None
        self.augment = None
        self.object_confidence = 0.01
        self.iou_threshold = 0.01
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.half = False
        self.isTrack = False

    def initializeVariable(
        self,
        conf=0.01,
        iou_threshold=0.01,
        classes=None,
        agnostic_nms=False,
        max_det=1000,
        half=False,
        augment=False,
    ):
        """
        This will initialize the model parameters of inference. This configuration is specific to camera group
        Args:
            conf (float): confidence of detection
            iou_threshold (float): intersection over union threshold of detection
            classes (obj): classes of the detection
            agnostic_ms (boolean): Non max supression
            half (boolean): precision of the detection
            augment (boolean): Augmentation while detection
        """
        self.object_confidence = conf
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.half = half

    def loadmodel(self):
        '''
        This will load Yolov8 model
        '''
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            print("MODEL NOT FOUND")
            sys.exit()
        # self.stride = int(self.model.stride.max())
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )

    def getClasses(self):
        """
        Get the classes of the model

        """
        return self.model.names

    def infer(self, image, model_config=None):
        """
        This will do the detection on the image
        Args:
            image (array): image in numpy array
            model_config (dict): configuration specific to camera group for detection
        Returns:
            list: list of dictionary. It will have all the detection result.
        """
        image_height, image_width, _ = image.shape
        print("image shape====",image.shape)
        raw_image = copy.deepcopy(image)
        img0 = copy.deepcopy(image)
        img = copy.deepcopy(image)
        print("model====>", self.model)
        if model_config is not None:
            self.isTrack = model_config['is_track']
            self.object_confidence = model_config["conf_thres"]
            self.iou_threshold = model_config["iou_thres"]
            self.max_det = model_config["max_det"]
            self.agnostic_nms = model_config["agnostic_nms"]
            self.augment = model_config["augment"]
            # self.classes=model_config["classes"]
        
        if self.isTrack == False:

            results = self.model.predict(img, conf=self.object_confidence, iou=self.iou_threshold, boxes=True, classes=self.classes)
            listresult=[]
            for i,det in enumerate(results[0].boxes):
                print(det.data[0])
                listresult.append({
                    "class": int(det.data[0][5].numpy()),
                    "id": None,
                    "class_name": self.model.names[int(det.data[0][5].numpy())],
                    "score":round(float(det.data[0][4].numpy()), 2),
                    "xmin": int(det.data[0][0].numpy()),
                    "ymin": int(det.data[0][1].numpy()),
                    "xmax": int(det.data[0][2].numpy()),
                    "ymax": int(det.data[0][3].numpy()),
                    "xmin_c": round(float(det.xyxyn[0][0].numpy()),5),
                    "ymin_c": round(float(det.xyxyn[0][1].numpy()),5),
                    "xmax_c": round(float(det.xyxyn[0][2].numpy()),5),
                    "ymax_c": round(float(det.xyxyn[0][3].numpy()),5),  
                
                })
            print("listresult===",listresult)
            return listresult
        if self.isTrack == True:
            results = self.model.track(img, conf=self.object_confidence, iou=self.iou_threshold, boxes=True, classes=self.classes)
            listresult=[]
            print("*"*100)
            print(results[0].boxes)
            if len(results[0].boxes)>0:
                for i,det in enumerate(results[0].boxes):
                    # print(det.data[0])
                    if det.id is not None:
                        listresult.append({
                            "class": int(det.data[0][6].numpy()),
                            "id": int(det.id[0].numpy()),
                            "class_name": self.model.names[int(det.data[0][6].numpy())],
                            "score":round(float(det.data[0][5].numpy()), 2),
                            "xmin": int(det.data[0][0].numpy()),
                            "ymin": int(det.data[0][1].numpy()),
                            "xmax": int(det.data[0][2].numpy()),
                            "ymax": int(det.data[0][3].numpy()),
                            "xmin_c": round(float(det.xyxyn[0][0].numpy()),5),
                            "ymin_c": round(float(det.xyxyn[0][1].numpy()),5),
                            "xmax_c": round(float(det.xyxyn[0][2].numpy()),5),
                            "ymax_c": round(float(det.xyxyn[0][3].numpy()),5),  
                        
                        })
                    else:
                        print("no ids")
                        listresult.append({
                            "class": int(det.data[0][5].numpy()),
                            "id": None,
                            "class_name": self.model.names[int(det.data[0][5].numpy())],
                            "score":round(float(det.data[0][4].numpy()), 2),
                            "xmin": int(det.data[0][0].numpy()),
                            "ymin": int(det.data[0][1].numpy()),
                            "xmax": int(det.data[0][2].numpy()),
                            "ymax": int(det.data[0][3].numpy()),
                            "xmin_c": round(float(det.xyxyn[0][0].numpy()),5),
                            "ymin_c": round(float(det.xyxyn[0][1].numpy()),5),
                            "xmax_c": round(float(det.xyxyn[0][2].numpy()),5),
                            "ymax_c": round(float(det.xyxyn[0][3].numpy()),5),  
                        
                        })
            else:
                print("no detections")
            print("listresult===",listresult)
            return listresult

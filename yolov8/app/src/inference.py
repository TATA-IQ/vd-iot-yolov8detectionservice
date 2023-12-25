"""inference module"""
import os
# import glob
import copy
import sys
# import time

# import cv2
# import numpy as np
import torch
from ultralytics import YOLO

from console_logging.console import Console
console=Console()


class ImageSplit():
    def __init__(self):
        pass
    
    def mark_res(self,res,origin_y,origin_x,H,W):
        listresult=[]
        for i,det in enumerate(res[0].boxes):
            print(det.data[0])
            listresult.append({
                "class": int(det.data[0][5].numpy()),
                "id":None,
                "class_name": model.names[int(det.data[0][5].numpy())],
                "score":round(float(det.data[0][4].numpy()), 2),
                "xmin": int(det.data[0][0].numpy()) + origin_x,
                "ymin": int(det.data[0][1].numpy()) + origin_y,
                "xmax": int(det.data[0][2].numpy()) + origin_x,
                "ymax": int(det.data[0][3].numpy()) + origin_y,
                "xmin_c": round((int(det.data[0][0].numpy()) + origin_x)/W,5),
                "ymin_c": round((int(det.data[0][1].numpy()) + origin_y)/H,5),
                "xmax_c": round((int(det.data[0][2].numpy()) + origin_x)/W,5),
                "ymax_c": round((int(det.data[0][3].numpy()) + origin_y)/H,5),  

            })
        return listresult
    def deduplication(self,det_list, h,w):
        H = h
        W = w
        final_det = []
        clist_i = []
        for i in range(0, len(det_list)):
            for j in range(i+1, len(det_list)):
                if(i not in clist_i and (det_list[i][0]['class'] == det_list[j][0]['class']) and 
                   (((abs(det_list[i][0]['xmin'] - det_list[j][0]['xmax']) <= 5 or abs(det_list[i][0]['xmax'] - det_list[j][0]['xmin']) <= 5) and 
                   ((det_list[i][0]['ymax']-det_list[j][0]['ymin'])/(det_list[j][0]['ymax']-det_list[j][0]['ymin']) > 0.5 and
                    (det_list[j][0]['ymax']-det_list[i][0]['ymin'])/(det_list[i][0]['ymax']-det_list[i][0]['ymin']) > 0.5)) or 
                   (((abs(det_list[i][0]['ymin'] - det_list[j][0]['ymax']) <= 5 or abs(det_list[i][0]['ymax'] - det_list[j][0]['ymin']) <= 5)) and 
                   ((det_list[i][0]['xmax']-det_list[j][0]['xmin'])/(det_list[j][0]['xmax']-det_list[j][0]['xmin']) > 0.5 and
                    (det_list[j][0]['xmax']-det_list[i][0]['xmin'])/(det_list[i][0]['xmax']-det_list[i][0]['xmin']) > 0.5)))):
                    #print(str(i)+"_"+str(j))
                    #print(det_list[i][0])
                    #print(det_list[j][0])
                    cord =   [{'class': det_list[i][0]['class_id'],
                               "id":None,
                              'class_name': det_list[i][0]['class'],
                              'score': det_list[i][0]['score'],
                              'xmin': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']),
                              'ymin': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']),
                              'xmax': max(det_list[i][0]['xmax'], det_list[j][0]['xmax']),
                              'ymax': max(det_list[i][0]['ymax'], det_list[j][0]['ymax']),
                              'xmin_c': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']/W),
                              'ymin_c': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']/H),
                              'xmax_c': max(det_list[i][0]['xmax'], det_list[j][0]['xmax'])/W,
                              'ymax_c': max(det_list[i][0]['ymax'], det_list[j][0]['ymin'])/H}]
                    #print(cord)
                    final_det.append(cord)
                    clist_i.append(i)
                    clist_i.append(j)
        for i in range(0, len(det_list)):
            if(i not in clist_i):
                final_det.append(det_list[i])
        return final_det
    def split(self,frame,split_col,split_row,model):
        swidth_col =  int(frame.shape[1]/split_col)
        sheight_row =  int(frame.shape[0]/split_row)
        det_list = []
        h,w,_=frame.shape
        for i in range(0, split_row):
            for j in range(0, split_col):
                sub_img = frame[i*sheight_row:(i+1)*sheight_row, j*swidth_col:(j+1)*swidth_col]
                # cv2.imwrite("C:/Users/SridharBondla/Downloads/output/"+str(i)+"_"+str(j)+".jpg",sub_img)
                res=model.predict(sub_img)
                listres=self.mark_res(res, i*sheight_row, j*swidth_col, frame.shape[0], frame.shape[1])
                if len(listres)>0:
                    det_list.append(listres)
        return sum(self.deduplication(det_list,h,w),[])
                


class InferenceModel:
    """
    Yolov8 inference
    """
    def __init__(self, model_path=None, gpu=False, logger=None):
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
        self.log = logger

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
        This will initialize the model parameters of inference. 
        This configuration is specific to camera group
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
            self.log.error("MODEL NOT FOUND")
            console.error("MODEL NOT FOUND")
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
           results (list):  list of dictionary. It will have all the detection result.
        """
        # image_height, image_width, _ = image.shape
        print("image shape====", image.shape)
        self.log.info(f"image shape===={image.shape}")
        console.info(f"image shape===={image.shape}")
        # raw_image = copy.deepcopy(image)
        # img0 = copy.deepcopy(image)
        img = copy.deepcopy(image)
        # print("model====>", self.model)
        if model_config is not None:
            self.isTrack = model_config['is_track']
            self.object_confidence = model_config["conf_thres"]
            self.iou_threshold = model_config["iou_thres"]
            self.max_det = model_config["max_det"]
            self.agnostic_nms = model_config["agnostic_nms"]
            self.augment = model_config["augment"]
            # self.classes=model_config["classes"]
        if self.isTrack is False:
            results = self.model.predict(img, conf=self.object_confidence,
                                         iou=self.iou_threshold, boxes=True,classes=self.classes)
            listresult=[]
            for i,det in enumerate(results[0].boxes):
                print(i,det.data[0])
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
            self.log.info(f"listresult==={listresult}")
            console.info(f"listresult==={listresult}")
            return listresult
        if self.isTrack is True:
            results = self.model.track(img, conf=self.object_confidence, iou=self.iou_threshold, boxes=True, classes=self.classes)
            listresult=[]
            # print("*"*100)
            # self.(results[0].boxes)
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
                        self.log.info("no ids")
                        console.info("no ids")
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
                self.log.info("no detections")
                console.info("no detections")
            self.log.info(f"listresult==={listresult}")
            console.info(f"listresult==={listresult}")
            return listresult
        
    def mark_res(self,res,origin_y,origin_x,H,W):
        listresult=[]
        for i,det in enumerate(res[0].boxes):
            # print(det.data[0])
            listresult.append({
                "class": int(det.data[0][5].numpy()),
                "id":None,
                "class_name": self.model.names[int(det.data[0][5].numpy())],
                "score":round(float(det.data[0][4].numpy()), 2),
                "xmin": int(det.data[0][0].numpy()) + origin_x,
                "ymin": int(det.data[0][1].numpy()) + origin_y,
                "xmax": int(det.data[0][2].numpy()) + origin_x,
                "ymax": int(det.data[0][3].numpy()) + origin_y,
                "xmin_c": round((int(det.data[0][0].numpy()) + origin_x)/W,5),
                "ymin_c": round((int(det.data[0][1].numpy()) + origin_y)/H,5),
                "xmax_c": round((int(det.data[0][2].numpy()) + origin_x)/W,5),
                "ymax_c": round((int(det.data[0][3].numpy()) + origin_y)/H,5),  

            })
        return listresult
    def deduplication(self,det_list, h,w):
        H = h
        W = w
        final_det = []
        clist_i = []
        for i in range(0, len(det_list)):
            for j in range(i+1, len(det_list)):
                if(i not in clist_i and (det_list[i][0]['class'] == det_list[j][0]['class']) and 
                   (((abs(det_list[i][0]['xmin'] - det_list[j][0]['xmax']) <= 5 or abs(det_list[i][0]['xmax'] - det_list[j][0]['xmin']) <= 5) and 
                   ((det_list[i][0]['ymax']-det_list[j][0]['ymin'])/(det_list[j][0]['ymax']-det_list[j][0]['ymin']) > 0.5 and
                    (det_list[j][0]['ymax']-det_list[i][0]['ymin'])/(det_list[i][0]['ymax']-det_list[i][0]['ymin']) > 0.5)) or 
                   (((abs(det_list[i][0]['ymin'] - det_list[j][0]['ymax']) <= 5 or abs(det_list[i][0]['ymax'] - det_list[j][0]['ymin']) <= 5)) and 
                   ((det_list[i][0]['xmax']-det_list[j][0]['xmin'])/(det_list[j][0]['xmax']-det_list[j][0]['xmin']) > 0.5 and
                    (det_list[j][0]['xmax']-det_list[i][0]['xmin'])/(det_list[i][0]['xmax']-det_list[i][0]['xmin']) > 0.5)))):
                    #print(str(i)+"_"+str(j))
                    #print(det_list[i][0])
                    #print(det_list[j][0])
                    cord =   [{'class': det_list[i][0]['class_id'],
                               "id":None,
                              'class_name': det_list[i][0]['class'],
                              'score': det_list[i][0]['score'],
                              'xmin': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']),
                              'ymin': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']),
                              'xmax': max(det_list[i][0]['xmax'], det_list[j][0]['xmax']),
                              'ymax': max(det_list[i][0]['ymax'], det_list[j][0]['ymax']),
                              'xmin_c': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']/W),
                              'ymin_c': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']/H),
                              'xmax_c': max(det_list[i][0]['xmax'], det_list[j][0]['xmax'])/W,
                              'ymax_c': max(det_list[i][0]['ymax'], det_list[j][0]['ymin'])/H}]
                    #print(cord)
                    final_det.append(cord)
                    clist_i.append(i)
                    clist_i.append(j)
        for i in range(0, len(det_list)):
            if(i not in clist_i):
                final_det.append(det_list[i])
        return final_det
    def split(self,frame,split_col,split_row,model):
        print(f"===split_col,split_row=={split_col},{split_row}")
        self.log.info(f"===split_col,split_row=={split_col},{split_row}")
        console.info(f"===split_col,split_row=={split_col},{split_row}")
        swidth_col =  int(frame.shape[1]/split_col)
        sheight_row =  int(frame.shape[0]/split_row)
        det_list = []
        h,w,_=frame.shape
        for i in range(0, split_row):
            for j in range(0, split_col):
                sub_img = frame[i*sheight_row:(i+1)*sheight_row, j*swidth_col:(j+1)*swidth_col]
                # cv2.imwrite("C:/Users/SridharBondla/Downloads/output/"+str(i)+"_"+str(j)+".jpg",sub_img)
                res=model.predict(sub_img)
                listres=self.mark_res(res, i*sheight_row, j*swidth_col, frame.shape[0], frame.shape[1])
                if len(listres)>0:
                    det_list.append(listres)
        return sum(self.deduplication(det_list,h,w),[])
    
        
    def infer_v2(self, image, model_config=None, split_columns=1, split_rows=1):
        """
        This will do the detection on the image
        Args:
            image (array): image in numpy array
            model_config (dict): configuration specific to camera group for detection
        Returns:
           results (list):  list of dictionary. It will have all the detection result.
        """
        # image_height, image_width, _ = image.shape
        print("image shape====", image.shape)
        self.log.info(f"image shape===={image.shape}")
        console.info(f"image shape===={image.shape}")
        self.log.info(split_columns,split_rows)
        # raw_image = copy.deepcopy(image)
        # img0 = copy.deepcopy(image)
        img = copy.deepcopy(image)
        # print("model====>", self.model)
        if model_config is not None:
            self.isTrack = model_config['is_track']
            self.object_confidence = model_config["conf_thres"]
            self.iou_threshold = model_config["iou_thres"]
            self.max_det = model_config["max_det"]
            self.agnostic_nms = model_config["agnostic_nms"]
            self.augment = model_config["augment"]
            # self.classes=model_config["classes"]
        
        
        listresult = self.split(img, split_columns, split_rows,self.model)
        if len(listresult)==0:
            self.log.info("no detections")
            console.info("no detections")
        else:
            console.info("detections")
            self.log.info(listresult)
        return listresult
        
        
        
        # if self.isTrack is False:
        #     results = self.model.predict(img, conf=self.object_confidence,
        #                                  iou=self.iou_threshold, boxes=True,classes=self.classes)
        #     listresult=[]
        #     for i,det in enumerate(results[0].boxes):
        #         print(i,det.data[0])
        #         listresult.append({
        #             "class": int(det.data[0][5].numpy()),
        #             "id": None,
        #             "class_name": self.model.names[int(det.data[0][5].numpy())],
        #             "score":round(float(det.data[0][4].numpy()), 2),
        #             "xmin": int(det.data[0][0].numpy()),
        #             "ymin": int(det.data[0][1].numpy()),
        #             "xmax": int(det.data[0][2].numpy()),
        #             "ymax": int(det.data[0][3].numpy()),
        #             "xmin_c": round(float(det.xyxyn[0][0].numpy()),5),
        #             "ymin_c": round(float(det.xyxyn[0][1].numpy()),5),
        #             "xmax_c": round(float(det.xyxyn[0][2].numpy()),5),
        #             "ymax_c": round(float(det.xyxyn[0][3].numpy()),5),
        #             })
        #     print("listresult===",listresult)
        #     return listresult
        # if self.isTrack is True:
        #     results = self.model.track(img, conf=self.object_confidence, iou=self.iou_threshold, boxes=True, classes=self.classes)
        #     listresult=[]
        #     print("*"*100)
        #     print(results[0].boxes)
        #     if len(results[0].boxes)>0:
        #         for i,det in enumerate(results[0].boxes):
        #             # print(det.data[0])
        #             if det.id is not None:
        #                 listresult.append({
        #                     "class": int(det.data[0][6].numpy()),
        #                     "id": int(det.id[0].numpy()),
        #                     "class_name": self.model.names[int(det.data[0][6].numpy())],
        #                     "score":round(float(det.data[0][5].numpy()), 2),
        #                     "xmin": int(det.data[0][0].numpy()),
        #                     "ymin": int(det.data[0][1].numpy()),
        #                     "xmax": int(det.data[0][2].numpy()),
        #                     "ymax": int(det.data[0][3].numpy()),
        #                     "xmin_c": round(float(det.xyxyn[0][0].numpy()),5),
        #                     "ymin_c": round(float(det.xyxyn[0][1].numpy()),5),
        #                     "xmax_c": round(float(det.xyxyn[0][2].numpy()),5),
        #                     "ymax_c": round(float(det.xyxyn[0][3].numpy()),5),                        
        #                 })
        #             else:
        #                 print("no ids")
        #                 listresult.append({
        #                     "class": int(det.data[0][5].numpy()),
        #                     "id": None,
        #                     "class_name": self.model.names[int(det.data[0][5].numpy())],
        #                     "score":round(float(det.data[0][4].numpy()), 2),
        #                     "xmin": int(det.data[0][0].numpy()),
        #                     "ymin": int(det.data[0][1].numpy()),
        #                     "xmax": int(det.data[0][2].numpy()),
        #                     "ymax": int(det.data[0][3].numpy()),
        #                     "xmin_c": round(float(det.xyxyn[0][0].numpy()),5),
        #                     "ymin_c": round(float(det.xyxyn[0][1].numpy()),5),
        #                     "xmax_c": round(float(det.xyxyn[0][2].numpy()),5),
        #                     "ymax_c": round(float(det.xyxyn[0][3].numpy()),5),
        #                 })
        #     else:
        #         print("no detections")
        #     print("listresult===",listresult)
        #     return listresult

""" model download """
import os
import shutil
from zipfile import ZipFile
from minio import Minio
from minio.error import S3Error
# import urllib3
# import cv2
# import numpy as np
# from datetime import datetime


class DownloadModel:
    """
    Class to download model from miniodb and remove it after validation is done.
    """

    def __init__(self, bucket_name, minioconf):
        """
        args: bucket_name-> Bucket name of miniodb
        """
        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint=minioconf["endpoint"],
            access_key=minioconf["access_key"],
            secret_key=minioconf["secret_key"],
            secure=minioconf["secure"],
        )

    def save_data(self, object_name, local_path):
        """
        Download the file from minio db and save it to local.
        Args:
            object_name (str): full path of file from miniodb
            local_path (str): path to save the model locally
        """
        print(object_name)
        obj_name = object_name.split("/")[-1]
        print(obj_name)
        save_path = os.path.join(local_path, obj_name)
        try:
            self.client.fget_object(self.bucket_name, object_name, save_path)
            print(f"{object_name} is saved into {save_path}")
        except S3Error as expection:
            print(expection)
            print(f"{object_name} {expection.message} ")

    # def save_model_files(self, object_path, local_path):
    #     """
    #     Args:
    #         object_name (str): full path of file from miniodb
    #         local_path (str): path to save the model locally
    #     """
    #     obj_name = object_name.split("/")[-1]
    #     # print(obj_name)
    #     save_path = os.path.join(local_path, obj_name)
    #     try:
    #         self.client.fget_object(bucket_name, object_name, save_path)
    #         print(f"{object_name} is saved into {save_path}")
    #     except S3Error as e:
    #         print(e)
    #         print(f"{object_name} {e.message} ")

    def unzip(self, path, unzippath, modelname):
        """
        Unzip the downloaded model
        Args:
            path (str): path of the downloaded model in zip file
            unzippath (str): path to unzip the downloaded model
            modelname (str): name of the model
        """
        print("Zip path===>", path, modelname)
        with ZipFile(path, "r") as zObject:
            zObject.extractall(path=unzippath)
        os.remove(path)

    def removeData(self, path):
        '''
        Remove the downloaded zip file
        Args:
            path (str): path of the .zip file
        '''
        shutil.rmtree(path)

    def createLocalFolder(self,foldername):
        '''
        This function will create a local path for the model
        Args:
            foldername (str): name of the model folder

        '''
        if os.path.exists(foldername):
            if len(os.listdir(foldername))>0:
                shutil.rmtree(foldername)
                os.mkdir(foldername)
        else:
            os.mkdir(foldername)
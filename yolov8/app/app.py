import base64
import os
import socket
from urllib.parse import urlparse
from zipfile import ZipFile

import cv2
import glob2
import numpy as np
import requests
import uvicorn
import torch
from src.parser import Config
from fastapi import FastAPI
from src.image_model import Image_Model
from src.download import SFTPClient
from src.inference import InferenceModel
from src.model_download import DownloadModel


class SetupModel:
    """
    Class to Setup the Infernce Model
    """

    def __init__(self, config_path="config/config.yaml", model_config_path="config/model.yaml"):
        conf = Config.yamlconfig(config_path)[0]
        modelconf = Config.yamlModel(model_config_path)[0]
        self.apis = conf["apis"]
        self.sftp = conf["sftp"]
        self.minio = conf["minio"]
        self.modelconf = modelconf

    def get_local_ip(self):
        """
        Get the ip of server
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(("192.255.255.255", 1))
            IP = s.getsockname()[0]
        except ConnectionRefusedError as ex:
            print("Exception while connecting: ", ex)
            IP = "127.0.0.1"
        finally:
            s.close()
        return IP

    # subprocess.getoutput("docker ps -aqf name=containername")
    """
    hostid = subprocess.getoutput("cat /etc/hostname")
    cgroupout = subprocess.getoutput("cat /proc/self/cgroup")
    print(cgroupout)
    print("===>", cgroupout.split(":")[2][8:20])
    hostid = cgroupout.split(":")[2][8:20]
    # hostid="ce09255b0b13"
    query = {"container_id": hostid}
    print("query====>", query)
    responseContainer = requests.get(self.apis["container"], json=query)
    print(responseContainer.json())
    responseModel = requests.get(self.apis["container_model"], json=query)
    print("found====>", responseModel.json())
    """

    def getModelMasterCofig(self):
        """
        Get the Master Configuration of model
        """
        modelmaster = requests.get(
            self.apis["model_master_config"], json={"model_id": self.modelconf["model_id"]}
        ).json()

        print("====Model Master Response======")
        print(modelmaster)
        return modelmaster["data"]
        # model_path = responseModel.json()["data"][0]["model_path"]
        # time.sleep(10)

    def downloadSftp(self, model_path):
        """
        This function will dowbload the model from the server

        Args:
            model_path (dict): sftp server connection configuration
        """
        print("downloading from server")

        sf = SFTPClient(self.sftp["host"], self.sftp["port"], self.sftp["username"], self.sftp["password"])

        sf.downloadyolov5(model_path, "model/test.zip")
        print("downloaded from server")

        # model_nm = model_path.split(".")[0]
        with ZipFile("model/test.zip", "r") as zObject:
            zObject.extractall(path="model/temp")

    def downloadMinio(self, modelmaster):
        """
        Args:
            modelmaster (dict): configuration to connect with minio
        """
        local_path = "model/"
        yolodownload = DownloadModel(modelmaster["model_framework"], self.minio)
        yolodownload.save_data(modelmaster["model_path"], local_path)
        # downloadData(data.model_path,local_path)
        modelpathparse = urlparse(modelmaster["model_path"])
        model_name = os.path.basename(modelpathparse.path)
        yolodownload.unzip(local_path + "/" + model_name, local_path, model_name)

    def createIP(self):
        """
        This function will create the api and update the endpoint url of model
        """
        ip = self.get_local_ip()
        url = "http://" + str(ip) + ":" + str(self.modelconf["port"]) + "/detect"

        model_id = self.modelconf["model_id"]
        print("updating for model id===>", model_id)

        responseupdate = requests.post(
            self.apis["update_endpoint"], json={"model_end_point": url, "model_id": model_id}
        )
        print(responseupdate.json())

    def getModelConfig(self):
        """
        This function call the model configuration api
        """
        model_config = requests.get(self.apis["model_config"], json={"model_id": self.modelconf[" model_id"]})
        return model_config

    def startModel(self):
        """
        This function does the model setup
        """
        modelmasterdata = self.getModelMasterCofig()
        self.downloadMinio(modelmasterdata[0])
        gpu = False
        model_name = modelmasterdata[0]["model_name"]
        framework = modelmasterdata[0]["model_framework"]

        if torch.cuda.is_available():
            gpu = True
        model_list = os.listdir("model/")
        im = InferenceModel(model_path="model/" + model_list[0] + "/saved_model", gpu=gpu)
        files = glob2.glob("model/" + model_list[0] + "/*.pbtxt")
        im.read_label_map(files[0])
        im.loadmodel()
        print("====Model Loaded====")
        print("Running api on GPU {}".format(gpu))
        self.createIP()

        return im, self.modelconf, model_name, framework, "model/" + model_list[0]


st = SetupModel()
im, modelconf, model_name, framework, path = st.startModel()

app = FastAPI()


def strToImage(imagestr):
    imagencode = imagestr.encode()
    decodedimage = base64.b64decode(imagencode)
    nparr = np.frombuffer(decodedimage, np.uint8)
    img_np = cv2.imdecode(nparr, flags=1)
    return img_np


@app.get("/test")
def test_fetch():
    """
    Test Api: Just for testing if model is running
    """
    return {"status": "active", "message": f"Model Name: {0} Framework {1}".format(model_name, framework)}


@app.post("/detect")
def detection(data: Image_Model):
    """
    Args:
        data (Image_Model): Accepts image, image name and configuration specific to the camera group

    Returns:
        dict: inferred result of the images
    """
    image = strToImage(data.image)
    print(image)

    if data.model_config is None:
        res = im.infer(image)
    else:
        res = im.infer(image, data.model_config)
    print("======inference done**********")
    print(res)
    print(type(res))
    return {"data": res}


@app.get("/classes")
def getClasses():
    """
    This function return all the classes of the model
    """
    res = im.getClasses()
    return {"data": res}


if __name__ == "__main__":
    print("=====inside main************")
    uvicorn.run(app, host="0.0.0.0", port=int(modelconf["port"]))


# app = FastAPI()
# @app.get("/test")
# async def ClassMaster_fetch(ImageModel):
#     return {"data":"Hi From TesorFlow "+responseModel.json()["data"][0]["model_usecase"]}

#         ##im.infer(ImageModel.image)

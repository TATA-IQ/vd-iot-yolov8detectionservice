import requests
from config_parser.parser import Config
# url = "http://127.0.0.1:8000/detection/tensorflow/validate"
# # correct input
# message = {
#     "model_framework": "tensorflow",
#     "model_path": "/object_detection/usecase1/model_id_1/Person.zip",
#     "model_id": "model_id_1",
#     "model_name": "Person.zip",
# }
# # wrong input
# #message={"model_framework":"yolov5","model_path":"/object_detection/usecase1/model_id_2/ppe.zip","model_id":"model_id_2","model_name":"ppe.zip"}

# response = requests.post(url, json=message)
# print(response.json())
# # from testhydra import *
# from config_parser.parser import *

# cf=getMinio("config/config.yaml")
data = Config.yamlconfig("config/config.yaml")
print("===>", data[0]["minio"])
data = Config.yamlconfig("config/model.yaml")
print("data==>",data)

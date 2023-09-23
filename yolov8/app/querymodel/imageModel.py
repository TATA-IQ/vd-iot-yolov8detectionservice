from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union


class Image_Model(BaseModel):
    '''
    Model generated for FastApi Query
    '''
    # image_name: Union[str, None] = None
    image_name: Union[str, None] = None
    image: Union[str, None] = None
    model_config: Union[dict, None] = None
    camera_id: Union[str, None] = None
    image_time: Union[str, None] = None
    model_type: Union[str, None] = None
    model_framework: Union[str, None] = None
  


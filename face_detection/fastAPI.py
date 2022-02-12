from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from utils import txt_to_npImage, npImage_to_txt, decode_img
from face_detection import get_face
import numpy as np


app = FastAPI()
class Item(BaseModel):
    image : str
    id_ : int
    

        
class ItemOut(BaseModel):
    face: str
    id_ : int
    shape: tuple
        
def process_request(request ):
    '''
    Do you request processing here
    '''
    
    face = get_face(request)
    return face

@app.post("/detect_face/")
async def detect_face(item: Item):
    '''
    Detect Face from the image
    
    '''
    
    image = decode_img(bytes(item.image, encoding="utf-8"))
    face = process_request(image)
    detected_face = npImage_to_txt(np.asarray(face))
    
    return ItemOut(face = detected_face, id_ = item.id_, shape = face.size )
#     return "ok"
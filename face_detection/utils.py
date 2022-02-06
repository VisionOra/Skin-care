import time
import base64
from io import BytesIO
import numpy as np
from PIL import Image

def log_error(e):
    '''
    use this function to log your errors
    
    '''
    print(e)
    time.sleep(1)

def decode_img(image):
    '''
    Decode String Image getting from request
    
    '''
    image = base64.b64decode(image)
    image = BytesIO(image)
    image = np.asarray(Image.open(image))
    return image


def img_to_txt(filename):
    '''
    Encoding Image
    '''
    with open(filename, "rb") as imageFile:
        image = base64.b64encode(imageFile.read()).decode()
    return image
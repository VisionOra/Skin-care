import time
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import cv2


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


def imgPath_to_txt(filename):
    '''
    Encoding Image
    '''
    with open(filename, "rb") as imageFile:
        image = base64.b64encode(imageFile.read()).decode()
    return image


def npImage_to_txt(image):
    '''
    Convert numpy image to base64
    '''
    _, im_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64


def txt_to_npImage(im_64):
    '''
    convert base64 to image
    
    '''
    im_bytes = base64.b64decode(im_64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

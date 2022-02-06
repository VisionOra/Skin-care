import os
import time
import numpy as np
import redis
from IPython.display import clear_output
from PIL import Image
from io import BytesIO
import base64
import json
import matplotlib.pyplot as plt
from face_detection import get_face
from utils import img_to_txt, decode_img, log_error


##########################
#
#   Global Variables
#
#
##########################

# Selecting Server
server = os.environ['redis_server'] if 'os.environ' in os.environ and len(os.environ['redis_server']) > 1 else 'localhost'
# connect with redis server as Bob
r = redis.Redis(host=server, port=6379)
# Publish and suscribe redis
req_p = r.pubsub()
# subscribe to request Channel
req_p.subscribe('new_request')







def process_request(request ):
    '''
    Do you request processing here
    '''
    im =  decode_img(request['image'])
    face = get_face(im)
    plt.imshow(face)
    plt.show()

def listen_stream():
    '''
    Listening to the stream. 
    
    IF got any request from the stream then process it at the same time.
    '''
    count = 0
    requests =[]  
    while 1:

        try:
            try:
                # Listening To the stream
                request = str(req_p.get_message()['data'].decode())
                if request is not None :requests.append(request)
            except TypeError as e: log_error(e)
            
            # If got any request from stream then process the function
            if len(requests) > 0:
                req_id = requests.pop(0)
                process_request(json.loads(request) )
                count += 1
                
            print(count)

        except Exception as e: log_error(e)

    
listen_stream()       
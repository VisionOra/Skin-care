


import mtcnn
from PIL import Image
import numpy as np
# print version

detector = mtcnn.MTCNN()

# draw an image with detected objects
def get_face(pixels):
    print("Image Received")
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # load the image
    
    
    # plot each box
    for result in faces:
    # get coordinates
        x, y, width, height = result['box']
        area = (x, y, x+width, y+height)
        # Crop, show, and save image
        cropped_img = Image.fromarray(pixels).crop(area)
    
    print("Image returned")
    return cropped_img
    
    
    
    
# filename = 'Annotated_images/eye_bags/images25.jpg' # filename is defined above, otherwise uncomment
# # load image from file
# pixels = plt.imread(filename) # defined above, otherwise uncomment
# # detector is defined above, otherwise uncomment

# # display faces on the original image
# draw_facebox(filename, faces)


# In[ ]:





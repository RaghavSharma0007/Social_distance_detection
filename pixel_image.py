import base64
import operator
import os
import re
from collections import defaultdict
from functools import reduce
from io import BytesIO
from random import choice, randint, shuffle
import draw
from nltk import FreqDist
from PIL import Image, ImageChops, ImageOps
from PIL import UnidentifiedImageError
import cv2
import numpy as np


#############################social distancing and camera parameters############################################
CONFIDENCE_THRESHOLD = 0
NMS_THRESHOLD = 0.4
net = cv2.dnn.readNet('model/yolov4.weights', 'model/yolov4.cfg')
model = cv2.dnn_DetectionModel(net)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
######################################################################################################################

basedir = os.path.abspath(os.path.dirname(__file__))
# face_xml = os.path.join(basedir,'static','data', 'haarcascade_frontalface_alt.xml')
# eye_xml = os.path.join(basedir,'static','data', 'haarcascade_eye_tree_eyeglasses.xml')
# face_cascade = cv2.CascadeClassifier(face_xml)
# eye_cascade = cv2.CascadeClassifier(eye_xml)


#reference: https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string
def PIL_image_to_base64(pil_image):
    buffered = BytesIO()
    img1=pil_image.save(buffered, format="JPEG")
    # print(img1)
    return base64.b64encode(buffered.getvalue())


def base64_to_PIL_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))

def atkinson_dither(image_string):
    min_dist=300
    # b = image_string.encode('utf-8')
    image = base64_to_PIL_image(image_string)
    if image is not None:

        # print(image)
        frame1 = np.array(image)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # frame1 = cv2.imread("./static/uploads/webcam_frame.png")
        classes, scores, boxes = detector(frame1)
        # print(boxes)
        draw.drawing(classes, scores, boxes, frame1, min_dist)
        imageRGB = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        PIL_image = Image.fromarray(imageRGB)

        # PIL_image.save('fromArray.jpg')
    # image.convert('RGB')
    # pix = image.load()
    # w, h = image.size
    # print(frame1)
    return PIL_image_to_base64(PIL_image)

def detector(frame):
    classes, scores, boxes=model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return classes, scores, boxes





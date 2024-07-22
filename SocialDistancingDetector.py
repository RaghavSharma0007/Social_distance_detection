from flask import Flask, flash, request, redirect, url_for, render_template, Response
from flask_cors import CORS, cross_origin
import json
import time
import os
import cv2
import draw
from io import BytesIO
import base64
from PIL import Image
import re
from pixel_image import atkinson_dither
import numpy as np
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
############################
app.secret_key = "Social distancing"
UPLOAD_FOLDER = 'static/video_upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#############################social distancing and camera parameters############################################
CONFIDENCE_THRESHOLD = 0
NMS_THRESHOLD = 0.4
net = cv2.dnn.readNet('model/yolov4.weights', 'model/yolov4.cfg')
model = cv2.dnn_DetectionModel(net)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


######################################################################################################################
@app.route('/home')
@cross_origin()
def home():
    return render_template('index_social_distancing.html')


###################################WEBCAM###################################################
# @app.route('/webcam')
# def index():
#     return render_template('layout.html')

# @app.route('/process', methods=['POST'])
# def process():
#     input = request.json
#     image_data = re.sub('^data:image/.+;base64,', '', input['img'])
#     image_ascii = atkinson_dither(image_data)
#     # print(image_ascii)
#     return image_ascii
###################################WEBCAM-END###################################################

def detector(frame):
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return classes, scores, boxes


#############################################IMAGE TEST##############################################
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'jfif'])


def allowed_file(filename1):
    # print("orinhht1")
    dot_index = filename1.index('.')
    ext = filename1[(dot_index + 1):].lower()
    # ext=filename1[-3:].lower()
    # print(ext)
    if (ext in ALLOWED_EXTENSIONS):
        return True
    else:
        return False


@app.route("/image_upl")
def image_home():
    return render_template('image_upload_html.html')


def PIL_to_base64(pil_image):
    buffered = BytesIO()
    img1 = pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue())


@app.route('/image_upload', methods=['GET', 'POST'])
@cross_origin()
def upload_image():
    min_dist = 100  # 100 for uploaded images, 300 for webcam
    if 'file1' not in request.files:
        flash('No files part')
        return 'No image selected for uploading'
        # return redirect(request.url)
    file1 = request.files['file1']
    if (file1.filename == ''):
        flash('No image selected for uploading')
        # return redirect(request.url)
    if allowed_file(file1.filename):
        # print("check")
        filename1 = secure_filename(file1.filename)
        # file1.save(os.path.join(app.config['UPLOAD_FOLDER'], "img1.png"))
        # frame = cv2.imread("./static/uploads/img1.png")
        #####################################################################
        #############################################################
        # read image file string data
        filestr1 = file1.read()
        # convert string data to numpy array
        npimg1 = np.fromstring(filestr1, np.uint8)
        # convert numpy array to image
        frame = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
        # cv2.IMREAD_COLOR)
        # frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
        #############################################################
        #####################################################################
        classes, scores, boxes = detector(frame)
        
        draw.drawing(classes, scores, boxes, frame, min_dist)
        frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
        PIL_pfile1 = Image.fromarray(frame)
        processed_file1 = PIL_to_base64(PIL_pfile1)
        processed_file1 = processed_file1.decode("utf-8")
        # dictionary = processed_file1
        # json_object = json.dumps(dictionary)
        # json_object.headers.add("Access-Control-Allow-Origin", "*")
        # print(json_object)
        # print(type(json_object))
        #        flash('Image successfully uploaded and displayed below')
        #        flash('Image successfully uploaded and displayed below')
        # return distortion_new_model,file1,file2,processed_file1,processed_file2
        # return {"distortion_new_model":str(distortion_new_model),"processed_file1":PIL_pfile1,"processed_file2":PIL_pfile2}
        return processed_file1

        # cv2.imwrite("./static/uploads/processed_img1.png", frame)
        # cv2.imwrite("img2", frame)
        # en =cv2.imencode('.jpg', frame)[1].tobytes()
        # cv2.imshow(frame)
        return display_image("img1.png", "processed_img1.png")
        # return render_template('index.html', filename1=filename1,filename2=filename2)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return 'Allowed image types are - png, jpg, jpeg, gif'
        # return redirect(request.url)
    return render_template('image_upload_html.html')


@app.route('/display', methods=['post'])
def display_image(filename1, filename2):
    return render_template('image_upload_html.html', filename1=filename1, filename2=filename2)


#######################################IMAGE END ################################################
###################################VIDEO TEST##################################################################
ALLOWED_EXTENSIONS_VID = set(['mp4', 'avi', 'wmv'])


def allowed_file_vid(filename1):
    # print("yessssssssssss")
    ext = filename1[-3:].lower()
    if (ext in ALLOWED_EXTENSIONS_VID):
        return True
    else:
        return False


@app.route("/video_upl")
@cross_origin()
def video_home():
    return render_template('video_upload_html.html')


@app.route("/video_upload", methods=['GET', 'POST'])
@cross_origin()
def upload_video():
    if 'file2' not in request.files:
        flash('No files part')
        return 'No video selected for uploading'
    file2 = request.files['file2']
    if (file2.filename == ''):
        flash('No video selected for uploading')
        return 'No video selected for uploading'
    if allowed_file_vid(file2.filename):
        # print("check")
        filename1 = secure_filename(file2.filename)
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], "test_vid.mp4"))
    return gen_frame()


@cross_origin()
def gen_frame():
    min_dist = 100  # 100 for uploaded images and video, 300 for webcam
    cap = cv2.VideoCapture("./static/video_upload/test_vid.mp4")
    ############remove everything from folder for frames to save
    dir = './static/uploads/vid_frames'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    ###########################################################
    (grabbed, frame) = cap.read()
    # print(frame.shape)
    if not grabbed:
        # print("exit")
        exit()
    else:
        count = 0
        while grabbed:
            start = time.time()
            classes, scores, boxes = detector(frame)
            end = time.time()
            draw.drawing(classes, scores, boxes, frame, min_dist)
            path1 = f"./static/uploads/vid_frames/frame{count}.png"
            cv2.imwrite(path1, frame)
            grabbed, frame = cap.read()
            count += 1
    ###########################recreate the video#############################
    height, width, layers = (cv2.imread("./static/uploads/vid_frames/frame1.png")).shape
    video = cv2.VideoWriter('./static/to_download/result.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 24, (width, height))
    global dictionary
    dictionary = {}
    for j in range(0, count):
        path = "./static/uploads/vid_frames/frame{}.png".format(j)
        frame = cv2.imread(path)
        # print(type(frame),"ghghghghghghghgggg")
        video.write(frame)
        ###############################################################################
        # PIL_pfile1 = Image.fromarray(frame)
        # processed_file1 = PIL_to_base64(PIL_pfile1)
        # processed_file1 = processed_file1.decode("utf-8")
        jj = str(j)
        # dictionary[jj] = processed_file1
        dictionary[jj] = frame
    # json_object = json.dumps(dictionary)

    ###############################################################################
    # cv2.destroyAllWindows()
    video.release()
    #########################################################################
    # return "https://sdeiaiml.com:5014//video_feed_video"
    dicti={"url":"https://sdeiaiml.com:5014/video_feed_video"}
    json_object = json.dumps(dicti)
    return json_object
    # return render_template('video_download_html.html', video_file="result.mp4")


@app.route("/video_gen", methods=['GET', 'POST'])
def video_gen():
    # dir = './static/uploads/vid_frames'
    # count=len(os.listdir(dir))
    count=len(dictionary)
    for j in range(0, count):
        frame=dictionary[str(j)]
        en = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + en + b'\r\n')

#########################################video_feed for video#####################################
@app.route('/video_feed_video', methods=['GET','POST'])
def video_feed_video():
    return Response(
        video_gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def detector(frame):
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return classes, scores, boxes
    
###########################################################################################################

if __name__ == "__main__":
    # SERVER=Server(app.wsgi_app)
    # SERVER.serve()
    # app.config['TEMPLATES_AUTO_RELOAD'] = True
    # app.run(host="0.0.0.0",threaded=True,port=8010,debug=True)
    # app.run(host='0.0.0.0',threaded=True, port=5014,debug=True)
    app.run(host='0.0.0.0',ssl_context=('/SSL/server.crt', '/SSL/server.key'), threaded=True, port=5014,debug=True)


#############################
### Deep-Live-Cam Setting ###
#############################
import deeplivecam.modules.globals as dlc_glob
import deeplivecam.modules.core as dlc_core
from deeplivecam.modules.face_analyser import get_one_face


#########################
### Web & Cam Setting ###
#########################
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"]="1"
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import time
import random
import threading
import numpy as np

IMAGE_FOLDER = './web/static/sources/images'
BACKGROUND_FOLDER = './web/static/ui/bg'
BANNER_FOLDER = './web/static/ui/banner'

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'web', 'static'), template_folder=os.path.join(os.getcwd(), 'web', 'templates'))
socketio = SocketIO(app)

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
webcam.set(cv2.CAP_PROP_FPS, 60)

with open('./web/static/sources/userinfo.txt', 'r', encoding='utf-8') as file:
    users = [line.strip() for line in file]


##########################
### Mic & Chat Setting ###
##########################
from mic.chatgenerator import WhisperMic
from collections import deque

mic = WhisperMic(model="large-v3", device="cuda")
gpt_response = deque()
gpt_response_lock = threading.Lock()
stop_background_tasks = threading.Event()

##################################
### Frame Processing Functions ###
##################################
def load_select_image(image_path):
    if os.path.exists(image_path):
        print(f'Selected image path: {image_path}')
        img = cv2.imread(image_path)
        dlc_glob.user_select_image = get_one_face(img)
        if dlc_glob.user_select_image:
            print(f'Face detection succeed')
            return True
        else:
            print(f'Face detection failed')
            return False
    return False

def generate_frames():
    while True:
        success, frame = webcam.read()
        if not success:
            break
        temp_frame = frame.copy()
        #and dlc_glob.live_stream_state == 'live'
        if dlc_glob.user_select_image != None:
            for frame_processor in dlc_core.get_frame_processors_modules(dlc_glob.frame_processors):
                temp_frame = frame_processor.process_frame(dlc_glob.user_select_image, temp_frame)
        else:
            temp_frame = cv2.putText(temp_frame, "OFFLINE...", (50, 50), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', temp_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


###################
### Web Process ###
###################
@app.route('/')
def index():
    return render_template('zhzzk.html')


# image load
@app.route('/get-background', methods=['GET'])
def get_background():
    bg_image = [os.path.splitext(file)[0] for file in os.listdir(BACKGROUND_FOLDER) if file.endswith('.png')]
    return jsonify(bg_image[0])

@app.route('/get-banner', methods=['GET'])
def get_banner():
    banner_image = [os.path.splitext(file)[0] for file in os.listdir(BANNER_FOLDER) if file.endswith('.png')]
    return jsonify(banner_image)

@app.route('/get-images', methods=['GET'])
def get_images():
    images = [os.path.splitext(file)[0] for file in os.listdir(IMAGE_FOLDER) if file.endswith('.jpg')]
    return jsonify(images)

def watch_image_folder():
    previous_files = set()
    while True:
        current_files = set(os.listdir(IMAGE_FOLDER))
        if current_files != previous_files:
            images = sorted(
                [os.path.splitext(file)[0] for file in current_files if file.endswith('.jpg')],
                key=lambda x: int(x.split('_')[0])  # '1_aaa' -> 1 기준 정렬
            )
            socketio.emit('update_buttons', images)
            previous_files = current_files
        time.sleep(1)


# webcam frame load
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# process live/offline by image's face recognition result
@socketio.on("select_image")
def handle_image_selection(data):
    global gpt_response, gpt_response_lock, stop_background_tasks
    image_name = data.get("image")
    if not image_name:
        return
    image_path = os.path.join(IMAGE_FOLDER, f"{image_name}.jpg")
    if os.path.isfile(image_path):
        if dlc_glob.user_select_image_path != image_path:
            print(f"Image found: {image_path}")
            if load_select_image(image_path):
                dlc_glob.user_select_image_path = image_path
                stop_background_tasks.clear()
                socketio.start_background_task(target=handle_response)
                socketio.start_background_task(target=send_random_messages) 
                print("State: Live")
            else:
                dlc_glob.user_select_image_path = None
                stop_background_tasks.set()
                with gpt_response_lock:
                    gpt_response.clear()
                socketio.emit('clear_chat')
                print("State: Offline")
        else:
            print(f"Image duplicated: {image_path}")
    else:
        print(f"Image not found: {image_path}")


#####################
### Chat Function ###
#####################
def handle_response():
    global gpt_response, gpt_response_lock, mic, stop_background_tasks
    print('__1__')
    while dlc_glob.user_select_image_path != None:
        if stop_background_tasks.is_set():  # Check if we should stop
            break
        if not mic.mic_active:
            print("new input speech")
            mic.listen(phrase_time_limit=4, responses=gpt_response, lock=gpt_response_lock)
        socketio.sleep(1)
        
def send_random_messages():
    global users, gpt_response, gpt_response_lock, stop_background_tasks
    while dlc_glob.user_select_image_path != None:
        if stop_background_tasks.is_set():  # Check if we should stop
            break
        user = random.choice(users)
        with gpt_response_lock:
            if len(gpt_response) != 0:
                message = gpt_response[0]
                gpt_response.popleft()
                print(f'user: {user}, message: {message}')
                socketio.emit('message_response', {'user': user, 'message': message})
        socketio.sleep(random.choice([0.3, 0.7, 1]))


if __name__ == '__main__':
    dlc_core.run()
    socketio.start_background_task(watch_image_folder)
    socketio.run(app, debug=True)
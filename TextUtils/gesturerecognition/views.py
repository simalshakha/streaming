import cv2
import numpy as np
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import threading
import mediapipe as mp
from mediapipe.python.solutions import holistic
from keras.models import load_model


mp_drawing=mp.solutions.drawing_utils   #holistic model
mp_holistic=mp.solutions.holistic       #drawing utilities
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #color conversion BGR to RGB
    image.flags.writeable = False                   #Image is no longer writeable
    results = model.process(image)                  # make prediction
    image.flags.writeable = True                    #image is now writeable
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  #color conversion again RGB to BGR
    return image,results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Draw right hand connections


def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.y,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x,res.y,res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x,res.y,res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])
#path for exported data, numpy arrays

#action that we are creating and detect
actions = np.array(['Aausadi','Eklopan','Firstaid','Need','Sign','Sorry'])

#thirty videos worth of data
no_sequences = 50

#videos are going to be 60 frames in length
sequence_length = 60
label_map= {label:num for num, label in enumerate(actions)}
# Provide the path to your saved model
model_path = r'C:\Users\sshak\OneDrive\Desktop\codes\Hand-Gesture-Recognition\simal.h5'
# model_path = r'C:\Users\Dell\Downloads\1-7.h5'
# Load the model
loaded_model = load_model(model_path)
sequence = []
sentence = []
predictions = []
threshold = 0.9

a=False
# lock = threading.Lock()

@gzip.gzip_page
def Home(request):
    return render(request, 'home.html')

def start(request):
    global a
    a=True
    return render(request, 'start.html')

def stop(request):
    return render(request, 'stop.html')

def videogallery(request):
    return render(request, 'video.html')



class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.grabbed, self.frame = self.video.read()
        self.sequence = []  # Initialize the sequence variable here
        threading.Thread(target=self.update, args=()).start()
        
    def get_frame(self):
        image = self.frame.copy()  # Make a copy to avoid potential issues
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(image, holistic)
            # draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-60:]
            if len(self.sequence) == 60:
                res = loaded_model.predict(np.expand_dims(self.sequence, axis=0))[0]
                print("Predicted Action:", actions[np.argmax(res)])
                predictions.append(np.argmax(res))

            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()

    def update(self):
        global a
        while True:
                if a == True:
                    self.video.release()
                    break
                self.grabbed, self.frame = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video(request):
    global a
    a = False
    cam = VideoCamera()
    return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
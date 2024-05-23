from flask import Flask, render_template, request
import cv2
import numpy as np
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['facial_recognition']
collection = db['recognized_users']

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize faces using LBPH Face Recognizer
def recognize_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Train LBPH recognizer (for demo purposes, just use a placeholder ID)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([roi_gray], np.array([1]))
        
        # Predict the user ID (for demo purposes, always return 1)
        user_id, confidence = recognizer.predict(roi_gray)
        
        if confidence < 50:  # Adjust confidence threshold as needed
            # Recognized user, return user id or name
            return user_id
        else:
            # Face not recognized
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        image = request.form['photo']
        collection.insert_one({'username': username, 'password': password, 'image': image})
        return "Registration Successfull"
        # Add user registration logic
    return render_template('register.html')

@app.route('/vote', methods=['GET', 'POST'])
def vote():
    if request.method == 'POST':
        candidate_name = request.form['candidate']
        image = request.form['photo']
        collection.insert_one({'candidate': candidate_name,})
        id_to_check = image
        document = collection.find_one({"image": id_to_check})
        if document:
            return "voted successfully"
        else:
            return "Authentication Failed...Vote not Casted"
    return render_template('vote.html')



@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        # Get data from the HTML form
        username = request.form['username']

        # Capture image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        # Call function to recognize face
        user_id = recognize_face(frame)

        if user_id is not None:
            print("User ID:", user_id)
            # Store data in MongoDB
            collection.insert_one({'username': username, 'user_id': user_id})
            return 'Recognition successful. Data stored in MongoDB.'
        else:
            return 'Face not recognized.'

if __name__ == '__main__':
    app.run(debug=True)

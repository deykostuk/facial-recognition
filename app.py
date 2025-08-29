import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import load_model as keras_load_model
import logging

# Set TensorFlow logging to only show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Defining Flask App
app = Flask(__name__)

nimgs = 10
IMG_SIZE = 100
EMBEDDING_SIZE = 128

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object and face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Global variable for the CNN embedding model
embedding_model = None
cnn_model_path = 'static/cnn_embedding_model.h5'

def create_cnn_model():
    """Creates a simple CNN model for generating face embeddings."""
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(EMBEDDING_SIZE, activation='linear', name='embedding_layer')  # Linear activation for embeddings
    ])
    return model

# A function to extract a face and get its CNN embedding
def get_face_embedding(img):
    """
    Extracts a face from an image and returns its CNN embedding.
    Returns None if no face is detected.
    """
    global embedding_model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    if len(face_points) > 0:
        (x, y, w, h) = face_points[0]
        face_img = cv2.resize(img[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
        face_img = np.expand_dims(face_img, axis=0) # Add batch dimension
        embedding = embedding_model.predict(face_img, verbose=0)
        return embedding.flatten()
    return None

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# Identify face using ML model
def identify_face(facearray):
    """Identifies a person by their face embedding using a trained KNN model."""
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray.reshape(1, -1))[0]


# A function which trains the model on all the faces available in faces folder
def train_model():
    """
    Extracts CNN embeddings from all face images and trains a KNN classifier.
    Saves the trained KNN model.
    """
    global embedding_model
    faces_embeddings = []
    labels = []
    userlist = os.listdir('static/faces')
    
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            if img is not None:
                resized_face = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                embedding = embedding_model.predict(np.expand_dims(resized_face, axis=0), verbose=0).flatten()
                faces_embeddings.append(embedding)
                labels.append(user)

    if len(faces_embeddings) > 0:
        faces_embeddings = np.array(faces_embeddings)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces_embeddings, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        print("KNN model trained and saved successfully.")
    else:
        print("No face data found for training.")


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    """Extracts attendance data from the daily CSV file."""
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    """
    Adds a user's attendance record for the current day.
    Returns True if attendance was marked, False otherwise.
    """
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        return True
    return False


## A function to get names and rol numbers of all users
def getallusers():
    """Gets a list of all registered users from the file system."""
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder
def deletefolder(duser):
    """Deletes a user's face data folder."""
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


################## ROUTING FUNCTIONS #########################

@app.route('/')
def home():
    """Renders the main homepage with today's attendance."""
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/listusers')
def listusers():
    """Renders the page to list and manage all users."""
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    """Handles the deletion of a user."""
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    if not os.listdir('static/faces/'):
        if os.path.exists('static/face_recognition_model.pkl'):
            os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except Exception as e:
        print(f"Error during model retraining after deletion: {e}")
        pass

    return redirect(url_for('listusers'))


@app.route('/start', methods=['GET'])
def start():
    """Starts the real-time face recognition for attendance."""
    global embedding_model
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')
    
    # Use environment variable for video source, default to 0 (webcam)
    video_source = os.getenv('VIDEO_SOURCE_URL', 0)
    try:
        cap = cv2.VideoCapture(video_source)
    except Exception as e:
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=f'Error accessing video source: {e}. Check if the URL is correct or the camera is available.')

    message = 'Face not recognized.'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            message = "Failed to grab frame from video source."
            break
            
        faces = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.2, 5, minSize=(20, 20))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img_for_recog = cv2.resize(frame[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
            
            embedding = embedding_model.predict(np.expand_dims(face_img_for_recog, axis=0), verbose=0).flatten()
            
            identified_person = identify_face(embedding)
            
            # Check if attendance was just marked for this person
            if add_attendance(identified_person):
                message = f'Attendance marked for {identified_person.split("_")[0]}.'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle for success
                cv2.putText(frame, message, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Attendance', frame)
                cv2.waitKey(2000) # Wait 2 seconds to show confirmation
                break
            else:
                message = f'Attendance already marked for {identified_person.split("_")[0]}.'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue rectangle for already marked
                cv2.putText(frame, message, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Attendance', frame)
        
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27: # Press 'Esc' to exit
            message = "Attendance process manually stopped."
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=message)


@app.route('/add', methods=['GET', 'POST'])
def add():
    """Handles adding a new user and capturing face images."""
    global embedding_model
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        
        # Use environment variable for video source, default to 0 (webcam)
        video_source = os.getenv('VIDEO_SOURCE_URL', 0)
        try:
            cap = cv2.VideoCapture(video_source)
        except Exception as e:
            names, rolls, times, l = extract_attendance()
            return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=f'Error accessing video source: {e}. Check if the URL is correct or the camera is available.')
            
        i, j = 0, 0
        
        while i < nimgs:
            ret, frame = cap.read()
            if not ret:
                names, rolls, times, l = extract_attendance()
                return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess="Failed to grab frame from video source. Please try again.")

            faces = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.2, 5, minSize=(20, 20))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                
                if j % 5 == 0:
                    face_crop = frame[y:y+h, x:x+w]
                    cv2.imwrite(f'{userimagefolder}/{i}.jpg', face_crop)
                    i += 1
                j += 1
            
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        print('Training Model with new data...')
        train_model()
        
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=f'Successfully added new user: {newusername}.')
    else:
        return redirect(url_for('home'))


# Our main function which runs the Flask App
if __name__ == '__main__':
    # Load the CNN model only once at startup
    try:
        if not os.path.exists(cnn_model_path):
            print("Creating and saving a new CNN embedding model.")
            model = create_cnn_model()
            # Do not compile, as we only need it for inference
            model.save(cnn_model_path)
        embedding_model = keras_load_model(cnn_model_path, compile=False)
        print("CNN embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        embedding_model = None

    app.run(debug=True)

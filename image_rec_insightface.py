import cv2
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import urllib
import face_recognition
from datetime import datetime
import os
import requests
import pickle
from insightface.app import FaceAnalysis
import sklearn

# Initialize InsightFace detection and recognition model
app = FaceAnalysis(name="buffalo_l")  # 'buffalo_l' is a commonly used model.
app.prepare(ctx_id=0, det_size=(640, 640))  # GPU (ctx_id=0) or CPU (ctx_id=-1)

# Load your saved training database (face encodings)
# Assume 'face_database' is a dictionary {name: embedding}
with open("c:/Users/Osazuwa-Ojo Victory/augmented_face_database.pkl", "rb") as f:
    face_database = pickle.load(f)


def database(app):
    """Configures and returns the SQLAlchemy database object."""
    # Configure the SQLAlchemy connection string
    # params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-BJNEMLC;DATABASE=Flask_FaRec;Trusted_Connection=yes;')
    params = urllib.parse.quote_plus(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-P788JVN;DATABASE=Flask_FaRec;Trusted_Connection=yes;')
    app.config['SQLALCHEMY_DATABASE_URI'] = "mssql+pyodbc:///?odbc_connect=%s" % params
    # app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://<username>:<password>@<dsn>'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db = SQLAlchemy(app)
    return db

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(embedding, face_database, threshold=0.7):
    """Match face embedding to database."""
    best_match_name = "Unknown"
    best_match_score = 0
    best_match_not = "name"

    for name, db_embedding in face_database.items():
        similarity = cosine_similarity(embedding, db_embedding)
        if similarity > best_match_score and similarity >= threshold:
            best_match_score = similarity
            best_match_name = name
        elif similarity > best_match_score and similarity < threshold:
            best_match_not = name


    return best_match_name, best_match_score

def unknown_name(frame, face_encodings):
    if face_encodings is None:
        print(f'New Encodings: {type(face_encodings)} detected!')
        return  
      
    """Identifies and saves a new unknown face."""
    unknown_dir = '.\Faces_Dir\\Unknown_faces'
    largest = 0
    for folder in os.listdir(unknown_dir):
        no = int(folder.split('_')[-1])
        if no > largest:
            largest = no

    new_unknown = f'Unknown_{largest + 1}'
    output_dir = os.path.join(unknown_dir, new_unknown)

    # Load Previous unknowns
    new_dict = dict()
    try:
        with open('unknown_encodings.pkl', 'rb') as f:
            new_dict = pickle.load(f)
    except FileNotFoundError:
        print("Warning: 'unknown_encodings.pkl' not found. Create a new one.")

    # Compare previous with current unknown
    for key, value in new_dict.items():
        if type(value) is list:
            value = value[0]

        if value is None or (isinstance(value, list) and not value):
            print(f'{key}: {type(value)} detected!')
            continue

        else:
            known_np = np.array(value).reshape(1, -1)
            face_np = np.array(face_encodings).reshape(1, -1)
            matches = face_recognition.compare_faces(known_np, face_np)

            # If match, return just name
            if matches[0]:
                print(f'Match: {matches}, {key}')
                return key

    # Else create new unknown
    filename = f'{new_unknown}_{datetime.now().strftime("%Y-%m-%d %H_%M_%S")}'
    save_frame(filename, frame, output_dir)
    print(f'{new_unknown} saved!')

    # Add the most recent unknown to the dictionary and save all again as a pickle file.
    new_dict[new_unknown] = face_encodings

    with open('unknown_encodings.pkl', 'wb') as f:
        pickle.dump(new_dict, f)
        print('New encodings Saved!')

    return new_unknown

def crop_frame(frame, face_location):
    """Crops a frame around a detected face location."""
    expansion_amount = 300

    top, right, bottom, left = face_location
    # Expand the bounding box
    top = max(0, top - expansion_amount)
    right = min(frame.shape[1], right + expansion_amount)
    bottom = min(frame.shape[0], bottom + expansion_amount)
    left = max(0, left - expansion_amount)

    # Crop the face from the original frame
    face_image = frame[top:bottom, left:right]

    return face_image

def detect_objects(frame, face_location, face_encoding):
    """Detects faces using InsightFace and recognizes them with embeddings."""
    ret_dict = dict()

    # Perform face detection and recognition using InsightFace
    frame = crop_frame(frame, face_location)
    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)  # Bounding box coordinates
        embedding = face.embedding  # Face embedding vector

        # Match the embedding to the database
        name, confidence = recognize_face(embedding, face_database, threshold=0.15)

        if name == 'Unknown':
            name, confidence = unknown_name(frame, face_encoding), 0.4

        # Draw bounding box and label
        color = (0, 255, 0)  # Green color for bounding boxes
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f'{name} {confidence:.2f}',
                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Return processed results
    ret_dict = {
        'frame': frame,
        'label': name,
        'confidence': confidence + 0.3,
        'face_encodings': face_encoding
    }

    return ret_dict

def SVC_pred(frame, face_encodings, face_locations):
    """Predicts face identities using a trained SVC classifier."""
    try:
        with open('classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
    except FileNotFoundError:
        print("Warning: 'classifier.pkl' not found. Face recognition using SVC will not work.")
        return {'frame': frame, 'label': [], 'confidence': [], 'face_encodings': face_encodings}

    labels = []
    multi_conf = []

    for face_location, face_encoding in zip(face_locations, face_encodings):
        (top, right, bottom, left) = face_location
        pred_val = np.array(face_encoding).reshape(1, -1)
        predictions = classifier.predict(pred_val)
        confidence = classifier.predict_proba(pred_val).max()
        ret_dict = dict()

        name = ''

        if confidence > 0.79:
            name = predictions[0]
            labels.append(name)
            multi_conf.append(confidence)

        if name:
            if isinstance(name, np.ndarray):
                name = name.tolist()
            if isinstance(name, list):
                person = name[0]
                name = person
        
        else:
            labels.append(None)
            multi_conf.append(0)


            # Draw a box around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    ret_dict = {
        'frame': frame,
        'label': labels,
        'confidence': multi_conf,
        'face_encodings': face_encodings,
    }

    return ret_dict

def save_frame(filename, frame, output_dir="./static/Images and Icons/Captures/"):
    """Saves a given frame to a specified directory."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        cv2.imwrite(os.path.join(output_dir, f'{filename}.jpg'), frame)
        print('Saved Image:', f'{filename}.jpg')
    except Exception as e:
        print(f"Error saving image: {e}")

def call_post_function(frame):
    """Sends a POST request with the processed frame."""
    url = 'http://localhost:5000/upload_live'
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    data = {'frame': frame}

    response = requests.post(url, files=data)
    if response.status_code == 200:
        print('POST request successful')
    else:
        print(f'Error: {response.status_code}')

def process_faces(frame, face_locations, predicted, known_faces, img_id, current_face_encodings):
    for index, face_encoding in enumerate(current_face_encodings):
        if predicted and index < len(predicted):
            found_match = False
            for existing_id, face_data in known_faces.items():
                matches = face_recognition.compare_faces([face_data['encodings']], face_encoding)
                if matches[0]:
                    known_faces[existing_id]['prediction'].append(predicted[index])
                    if not known_faces[existing_id].get('datetime'):
                        known_faces[existing_id]['datetime'] = datetime.now()
                    found_match = True
                    break
            if not found_match:
                img_id += 1
                known_faces[img_id] = {
                    'encodings': face_encoding,
                    'prediction': [predicted[index]],
                    'datetime': datetime.now()
                }
    return known_faces, img_id

def average_prediction(predictions):
    """Averages predictions for each face."""
    name_comp = {}
    for name_prob in predictions:
        name, prob = name_prob
        name_comp[name] = name_comp.get(name, 0) + float(prob)

    highest = max(name_comp.items(), key=lambda x: x[1])  # Get the name with highest total probability
    avg_prob = highest[1] / len(predictions)  # Normalize probability sum

    return highest[0], f"{avg_prob:.3f}"

def face_upload(frame=None):
    """Processes a frame to detect, recognize, and track faces."""
    passed_frame = frame
    known_faces = dict()  # {img_id: {encodings:[], Predictions:[(name, prob)...5], date-time:first}}
    unknown_encodings = dict()

    face_locations = []
    predicted = []
    img_id = 1

    url = 0

    video_capture = cv2.VideoCapture(url, cv2.CAP_ANY)  # Local camera
    if not video_capture.isOpened():
        print("Error: Unable to open video source!")

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        
        if not ret:
            print('Unable to capture')
            break

        if passed_frame:
            frame = passed_frame

        # Convert Image from bgr to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            continue


        # Get encodings for the current faces
        current_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Load unknown encodings if they exist
        try:
            with open('unknown_encodings.pkl', 'rb') as f:
                unknown_face_encodings = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Error loading encodings: {e}")
            unknown_face_encodings = {}

        # Predict with SVC
        returned = SVC_pred(frame, current_face_encodings, face_locations)

        for i in range(len(returned['face_encodings'])):
            if not returned['label'][i]:
                insight_ret = detect_objects(frame, face_locations[i], current_face_encodings[i])
                returned['label'][i] = insight_ret['label']
                returned['confidence'][i] = insight_ret['confidence']
                returned['frame'] = insight_ret['frame']

        if returned:
            frame = returned['frame']
            current_face_encodings = returned['face_encodings']
            predicted = [list(pair) for pair in zip(returned['label'], returned['confidence'])]
            call_post_function(frame)

        # Process each detected face
        known_faces, img_id = process_faces(frame, face_locations, predicted, known_faces, img_id, current_face_encodings)

        for img_id in list(known_faces.keys()):
            if 'prediction' in known_faces[img_id] and len(known_faces[img_id]['prediction']) > 4:
                # Average the predictions for the face
                highest, prob = average_prediction(known_faces[img_id]['prediction'])

                if float(prob) < 0.65:
                    highest = unknown_name(frame, known_faces[img_id]['encodings'])

                date = known_faces[img_id].get('datetime')
                if date:
                    filename = f'{highest} {date.strftime("%Y-%m-%d %H_%M_%S")}'
                    save_frame(filename, frame)
                    print(f'The name: {highest}, The probability: {prob}, The time: {date}')

                # Clear prediction history after processing
                known_faces[img_id]['prediction'] = []
                known_faces[img_id]['datetime'] = None

                return [highest, prob, date]
            
            else:
                continue
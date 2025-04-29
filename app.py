from flask import Flask, render_template, request, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from image_rec_insightface import database, face_upload
import urllib.parse
# from threading import Thread, Event
from multiprocessing import Process, Event  # Import Process and Event from multiprocessing
import time
from datetime import datetime, date

app = Flask(__name__)
db = database(app)
video_access_event = Event()
frame = None

class Personnel(db.Model):
    # __tablename__ = 'existing_table'  # Specify the table name if it's different from the class name
    ID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(50))
    Sex = db.Column(db.String(10))
    Description = db.Column(db.Text)
    ImagePath = db.Column(db.String(255))

class Logs(db.Model):
    Log_ID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(50))
    Probability = db.Column(db.String(10))
    Time = db.Column(db.DateTime, default=datetime.utcnow)

def update_log(received_list = None):
    #Add new log to the db if the last log of this name is more than 5 mins from this one
    rec_list = received_list
    if not rec_list:
        rec_list = face_upload()
    if not rec_list:
        pass
    name = rec_list[0]
    prob = rec_list[1]
    time = rec_list[2]

    most_recent_log = Logs.query.filter_by(Name=name).order_by(Logs.Time.desc()).first()
    if most_recent_log:
        if (time - most_recent_log.Time).total_seconds() >= 300: 
            new_log = Logs(Name=name, Probability=prob, Time = time)
            db.session.add(new_log)
            db.session.commit()
        else:
            print(f'Last seen was {(time - most_recent_log.Time).total_seconds()} seconds ago!')
    
    else:
        new_log = Logs(Name=name, Probability=prob, Time = time)
        db.session.add(new_log)
        db.session.commit()

def continuous_function():
    with app.app_context():
        while True:
            update_log()
            print("Running continuous detection and recognition...")
            time.sleep(5)  # Sleep for 5 seconds before next iteration        

# Start the continuous function in a separate process
continuous_process = Process(target=continuous_function)
continuous_process.daemon = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Dashboard')
def dashboard():
    video_access_event.clear()
    # Get today's date
    today = date.today()

    # Calculate the start and end of today
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    # Query logs for today
    logs_today = Logs.query.filter(Logs.Time >= start_of_day, Logs.Time <= end_of_day).order_by(Logs.Time.desc()).all()

    # Calculate the number of logs for today
    num_logs_today = len(logs_today)

    # logs = Logs.query.limit(10).all()
    return render_template('DashBoard.html', title = 'Dashboard', no_logs = num_logs_today, logs = logs_today)

@app.route('/Logs')
def logs():
    video_access_event.clear()
    logs = Logs.query.order_by(Logs.Time.desc()).all()
    return render_template('Logs.html', title = 'Logs', logs = logs)

@app.route('/live')
def live():
    video_access_event.set()
    return render_template('Live.html', title = 'Live Feed')

@app.route('/ManageUsers')
def manageusers():
    video_access_event.clear()
    return render_template('Manageusers.html', title = 'Manage Users')

@app.route('/get_personnel_data', methods=['GET'])
def get_personnel_data():
    selected_option = request.args.get('selected_option')
    #Defining Global variables
    data = dict()
    times = []
    probs = []

    personnel_list = Personnel.query.all()
    logs = Logs.query.filter_by(Name=selected_option).order_by(Logs.Time.desc())
    for log in logs:
        times.append(log.Time)
        probs.append(log.Probability)
    print(times)

    for personnel in personnel_list:
        if personnel.Name == selected_option:
            data = {'name': personnel.Name, 
                    'sex': personnel.Sex,
                    'des': personnel.Description, 
                    'imgpath': personnel.ImagePath, 
                    'probs': probs, 
                    'times': times}
    return data

@app.route('/upload_live', methods=['POST'])
def upload_live():
    global frame
    try:
        # Access the data sent in the POST request
        data = request.files['frame'].read()  # Assuming JSON data is sent in the request body

        # Process the data (for example, print it)
        frame = data
        return jsonify({'message': 'Data received successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def frame_yield():
    while True:
        if frame:
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    video_access_event.set()
    # update_log()
    return Response(frame_yield(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    continuous_process.start()
    app.run(debug=True)
    # app.run(port=5000)

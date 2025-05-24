from flask import Flask, redirect, render_template, jsonify, Response, request, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin, LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import logging

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '12345'  # Needed for Flask-Login

db = SQLAlchemy(app)
migrate = Migrate(app, db)
logging.basicConfig(level=logging.DEBUG)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Assuming these imports are already defined in your environment
# Replace with your actual YOLO model initialization
model = YOLO('yolov8s.pt')

# Load the COCO class list
with open("coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Define parking areas
parking_areas = {
    1: [(52, 364), (30, 417), (73, 412), (88, 369)],
    2: [(105, 353), (86, 428), (137, 427), (146, 358)],
    3: [(159, 354), (150, 427), (204, 425), (203, 353)],
    4: [(217, 352), (219, 422), (273, 418), (261, 347)],
    5: [(274, 345), (286, 417), (338, 415), (321, 345)],
    6: [(336, 343), (357, 410), (409, 408), (382, 340)],
    7: [(396, 338), (426, 404), (479, 399), (439, 334)],
    8: [(458, 333), (494, 397), (543, 390), (495, 330)],
    9: [(511, 327), (557, 388), (603, 383), (549, 324)],
    10: [(564, 323), (615, 381), (654, 372), (596, 315)],
    11: [(616, 316), (666, 369), (703, 363), (642, 312)],
    12: [(674, 311), (730, 360), (764, 355), (707, 308)],
}

current_parking_status = {i: False for i in range(1, 13)}

# Define models using SQLAlchemy
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    bookings = db.relationship('Booking', backref='user', lazy=True)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class Booking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    parking_spot = db.Column(db.String(50), nullable=False)
    cost = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Flask-Login loader function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to check parking availability using YOLO model
def check_parking_availability(frame):
    global current_parking_status
    results = model.predict(frame)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    occupied_areas = {i: False for i in parking_areas.keys()}

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, cls = row
        cls = int(cls)
        if class_list[cls] == 'car':
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            for area_id, area_points in parking_areas.items():
                if cv2.pointPolygonTest(np.array(area_points, np.int32), (cx, cy), False) >= 0:
                    occupied_areas[area_id] = True

    current_parking_status = occupied_areas

# Generator function to stream video frames with parking status overlay
def generate_frames():
    cap = cv2.VideoCapture('parking1.mp4')
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (1020, 500))
        check_parking_availability(frame)

        for area_id, area_points in parking_areas.items():
            color = (0, 0, 255) if current_parking_status[area_id] else (0, 255, 0)
            cv2.polylines(frame, [np.array(area_points, np.int32)], True, color, 2)
            cv2.putText(frame, str(area_id), tuple(area_points[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Delay to slow down the frame rate
        time.sleep(0.5)
    cap.release()

# Routes for the Flask application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return redirect(url_for('parking_status'))
    else:
        return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            return render_template('register.html', message="Passwords do not match")

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create a new user
        new_user = User(username=username, email=email, password=hashed_password)

        # Add user to the database
        db.session.add(new_user)
        db.session.commit()

        # Redirect to login page after successful registration
        return redirect(url_for('index'))

    # Render register.html for GET requests
    return render_template('register.html')

@app.route('/parking')
@login_required
def parking_status():
    return render_template('parking.html', parking_areas=parking_areas, status=current_parking_status)

@app.route('/status')
def status():
    return jsonify(current_parking_status)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/payment', methods=['GET'])
@login_required
def payment():
    spot = request.args.get('spot')
    if spot:
        return render_template('payment.html', spot=spot)
    else:
        return redirect(url_for('index'))

@app.route('/process_payment', methods=['POST'])
@login_required
def process_payment():
    try:
        spot = request.form.get('spotNumber')
        hours = request.form.get('hours')
        total_amount = request.form.get('totalAmount')

        logging.debug(f"Spot: {spot}")
        logging.debug(f"Hours: {hours}")
        logging.debug(f"Total Amount: {total_amount}")

        if not spot or not hours or not total_amount:
            logging.error("Missing form data")
            return "Missing form data", 400

        cost = float(total_amount.replace(' INR', ''))
        logging.debug(f"Cost: {cost}")
        user_id = current_user.id
        logging.debug(f"User ID: {user_id}")

        new_booking = Booking(user_id=user_id, parking_spot=spot, cost=cost)
        db.session.add(new_booking)
        db.session.commit()

        logging.debug("Booking added successfully!")
        return jsonify(success=True, message="Booking added successfully!")
        #return redirect(url_for('parking_status'))

    except ValueError as ve:
        logging.error(f"Value Error: {ve}")
        return jsonify(success=False, message="Invalid input data!"), 400

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error adding booking: {e}")
        return jsonify(success=False, message="Error processing payment!"), 500


if __name__ == '__main__':
    app.run(debug=True)

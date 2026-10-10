import os
import cv2
import numpy as np
from datetime import datetime, timezone
from flask import Flask, render_template_string, Response, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

app = Flask(__name__)
app.config['SECRET_KEY'] = 'whithack-sisyphus-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///workout_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    sessions = db.relationship('ExerciseSession', backref='user', lazy=True)

class ExerciseSession(db.Model):
    __tablename__ = 'exercise_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)
    rep_count = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Initialize MediaPipe Pose Tasks Landmarker
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_lite.task')
if not os.path.exists(MODEL_PATH):
    import urllib.request
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        MODEL_PATH
    )

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360.0 - angle if angle > 180.0 else angle

global_reps = 0
global_stage = "up"

def generate_frames():
    global global_reps, global_stage
    demo_video = 'squat_demo.mp4'
    cap = cv2.VideoCapture(0)
    if not cap.isOpened() and os.path.exists(demo_video):
        cap = cv2.VideoCapture(demo_video)

    while True:
        success, frame = cap.read()
        if not success:
            if os.path.exists(demo_video):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
            if not success or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Webcam / Demo Feed Active", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 198), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        try:
            results = landmarker.detect(mp_image)
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                landmarks = results.pose_landmarks[0]
                hip = [landmarks[23].x * w, landmarks[23].y * h]
                knee = [landmarks[25].x * w, landmarks[25].y * h]
                ankle = [landmarks[27].x * w, landmarks[27].y * h]

                angle = calculate_angle(hip, knee, ankle)
                if angle > 160:
                    global_stage = "up"
                if angle < 100 and global_stage == 'up':
                    global_stage = "down"
                    global_reps += 1

                pts = [
                    (int(hip[0]), int(hip[1])),
                    (int(knee[0]), int(knee[1])),
                    (int(ankle[0]), int(ankle[1]))
                ]
                cv2.line(frame, pts[0], pts[1], (0, 255, 198), 3, cv2.LINE_AA)
                cv2.line(frame, pts[1], pts[2], (0, 255, 198), 3, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(frame, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 8, (0, 255, 198), 2, cv2.LINE_AA)

                cv2.putText(frame, f"KNEE: {int(angle)} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

        cv2.putText(frame, f'REPS: {global_reps}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 198), 3, cv2.LINE_AA)
        stage_color = (0, 255, 198) if global_stage == "down" else (255, 200, 0)
        cv2.putText(frame, f'STAGE: {global_stage.upper()}', (240, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, stage_color, 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

HTML_LAYOUT = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sisyphus AI Coach</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --bg: #090d16; --card-bg: #111827; --accent: #00ffc6; --text: #f3f4f6; --subtext: #9ca3af; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 0; }
        .navbar { display: flex; justify-content: space-between; align-items: center; background: #030712; padding: 16px 32px; border-bottom: 1px solid #1f2937; }
        .logo { font-weight: 800; font-size: 1.25rem; color: var(--accent); letter-spacing: 0.05em; }
        .nav-links a { color: var(--subtext); text-decoration: none; margin-left: 20px; font-weight: 500; transition: color 0.2s; }
        .nav-links a:hover { color: var(--accent); }
        .container { max-width: 1100px; margin: 40px auto; padding: 0 20px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
        .card { background: var(--card-bg); border-radius: 12px; padding: 24px; border: 1px solid #1f2937; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); }
        .btn { background: var(--accent); color: #000; border: none; padding: 12px 20px; border-radius: 8px; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-block; width: 100%; box-sizing: border-box; text-align: center; }
        .btn:hover { opacity: 0.9; }
        input[type="text"], input[type="password"] { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #374151; background: #030712; color: #fff; margin-bottom: 16px; box-sizing: border-box; }
        .stat-val { font-size: 2.25rem; font-weight: 800; color: var(--accent); margin: 8px 0; }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="logo">SISYPHUS AI</div>
        <div class="nav-links">
            {% if current_user.is_authenticated %}
                <a href="{{ url_for('dashboard') }}">Dashboard</a>
                <a href="{{ url_for('workout') }}">Live Workout</a>
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </div>
    <div class="container">
        <!--PAGE_CONTENT-->
    </div>
</body>
</html>
'''

def render_page(content, **kwargs):
    template = HTML_LAYOUT.replace('<!--PAGE_CONTENT-->', content)
    return render_template_string(template, **kwargs)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username and password:
            existing = User.query.filter_by(username=username).first()
            if not existing:
                user = User(username=username, password_hash=generate_password_hash(password))
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_page('''
    <div class="card" style="max-width: 400px; margin: 40px auto;">
        <h2>Create Account</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit" class="btn">Sign Up</button>
        </form>
    </div>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_page('''
    <div class="card" style="max-width: 400px; margin: 40px auto;">
        <h2>Welcome Back</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit" class="btn">Sign In</button>
        </form>
    </div>
    ''')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    sessions = ExerciseSession.query.filter_by(user_id=current_user.id).order_by(ExerciseSession.timestamp.desc()).all()
    total_reps = sum(s.rep_count for s in sessions)
    labels = [s.timestamp.strftime('%m/%d %H:%M') for s in reversed(sessions)]
    data = [s.rep_count for s in reversed(sessions)]
    
    return render_page('''
    <div class="grid">
        <div class="card">
            <h3>Repetition Volume</h3>
            <canvas id="progressChart"></canvas>
        </div>
        <div>
            <div class="card" style="margin-bottom: 24px;">
                <span style="color: var(--subtext);">Total Reps Logged</span>
                <div class="stat-val">{{ total_reps }}</div>
                <a href="{{ url_for('workout') }}" class="btn">Start Workout</a>
            </div>
            <div class="card">
                <h3>Recent Activity</h3>
                <ul style="padding-left: 20px; color: var(--subtext);">
                    {% for s in sessions %}
                        <li><strong>{{ s.exercise_type }}</strong>: {{ s.rep_count }} reps ({{ s.timestamp.strftime('%b %d') }})</li>
                    {% else %}
                        <li>No workout history recorded yet.</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    </div>
    <script>
        const ctx = document.getElementById('progressChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: {{ labels|tojson }},
                datasets: [{
                    label: 'Reps per Session',
                    data: {{ data|tojson }},
                    borderColor: '#00ffc6',
                    backgroundColor: 'rgba(0, 255, 198, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });
    </script>
    ''', sessions=sessions, total_reps=total_reps, labels=labels, data=data)

@app.route('/workout')
@login_required
def workout():
    return render_page('''
    <div class="card" style="text-align: center;">
        <h2>Live Tracking Feed</h2>
        <div style="margin: 20px 0;">
            <img src="{{ url_for('video_feed') }}" style="max-width: 100%; width: 640px; border-radius: 8px; border: 2px solid #1f2937;">
        </div>
        <form action="{{ url_for('save_workout') }}" method="POST" style="max-width: 300px; margin: 0 auto;">
            <input type="hidden" name="exercise_type" value="Squats">
            <button type="submit" class="btn">Save Workout Session</button>
        </form>
    </div>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_workout', methods=['POST'])
@login_required
def save_workout():
    global global_reps
    exercise_type = request.form.get('exercise_type', 'Squats')
    if global_reps > 0:
        session_rec = ExerciseSession(user_id=current_user.id, exercise_type=exercise_type, rep_count=global_reps)
        db.session.add(session_rec)
        db.session.commit()
        global_reps = 0
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=8080)

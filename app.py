import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string, Response, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'whithack-sisyphus-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///workout_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    sessions = db.relationship('ExerciseSession', backref='user', lazy=True)

class ExerciseSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)
    rep_count = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

mp_pose = mp.solutions.pose
pose = mp.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360.0 - angle if angle > 180.0 else angle

global_reps = 0
global_stage = None

def generate_frames():
    global global_reps, global_stage
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam / Demo Feed Active", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 198), 2)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)
                if angle > 160:
                    global_stage = "up"
                if angle < 90 and global_stage == 'up':
                    global_stage = "down"
                    global_reps += 1

            except Exception:
                pass

            cv2.putText(image, f'REPS: {global_reps}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 198), 3)

        ret, buffer = cv2.imencode('.jpg', frame if not success else image)
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
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = User(username=request.form.get('username'), password_hash=generate_password_hash(request.form.get('password')))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template_string(HTML_LAYOUT + '''
    {% block content %}
    <div class="card" style="max-width: 400px; margin: 40px auto;">
        <h2>Create Account</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit" class="btn">Sign Up</button>
        </form>
    </div>
    {% endblock %}
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template_string(HTML_LAYOUT + '''
    {% block content %}
    <div class="card" style="max-width: 400px; margin: 40px auto;">
        <h2>Welcome Back</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit" class="btn">Sign In</button>
        </form>
    </div>
    {% endblock %}
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
    
    return render_template_string(HTML_LAYOUT + '''
    {% block content %}
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
    {% endblock %}
    ''', sessions=sessions, total_reps=total_reps, labels=labels, data=data)

@app.route('/workout')
@login_required
def workout():
    return render_template_string(HTML_LAYOUT + '''
    {% block content %}
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
    {% endblock %}
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

import os
import sys

# Transparently delegate to local virtual environment if run via system python3
if __name__ == '__main__' and sys.prefix == sys.base_prefix:
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'bin', 'python')
    if os.path.isfile(_venv_python) and os.access(_venv_python, os.X_OK) and sys.executable != _venv_python:
        os.execv(_venv_python, [_venv_python] + sys.argv)

import time
import math
import json
import urllib.parse
import threading
from datetime import datetime, timezone, timedelta
from functools import wraps
from collections import defaultdict

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    render_template_string,
    Response,
    jsonify,
    request,
    redirect,
    url_for,
    flash,
    session as flask_session,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash

# MediaPipe Tasks imports
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# ---------------------------------------------------------
# PATHS & MODEL ASSETS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')
SQUAT_DEMO_PATH = os.path.join(BASE_DIR, 'squat_demo.mp4')
PUSHUP_DEMO_PATH = os.path.join(BASE_DIR, 'pushup_demo.mp4')
DUMBBELL_DEMO_PATH = os.path.join(BASE_DIR, 'dumbbell_demo.mp4')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'


def ensure_model_exists():
    """Ensure pose landmarker model exists locally."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print("[*] Downloading pose landmarker model...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Pose landmarker model ready.")


ensure_model_exists()

# ---------------------------------------------------------
# FLASK & DATABASE INITIALIZATION
# ---------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'whithack-sisyphus-secret-key-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', f"sqlite:///{os.path.join(BASE_DIR, 'workout_data.db')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# ---------------------------------------------------------
# DATABASE MODELS WITH EXP, LEVELING & PERSISTENT BADGES
# ---------------------------------------------------------
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    exp = db.Column(db.Integer, default=0, nullable=False)
    level = db.Column(db.Integer, default=1, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    sessions = db.relationship('ExerciseSession', backref='user', lazy=True, cascade='all, delete-orphan')
    badges = db.relationship('Badge', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ExerciseSession(db.Model):
    __tablename__ = 'exercise_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    exercise_type = db.Column(db.String(50), nullable=False)
    rep_count = db.Column(db.Integer, nullable=False, default=0)
    total_duration_seconds = db.Column(db.Float, nullable=False, default=0.0)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    avg_form_accuracy = db.Column(db.Float, nullable=False, default=100.0)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "exercise_type": self.exercise_type,
            "rep_count": self.rep_count,
            "total_duration_seconds": round(self.total_duration_seconds, 1),
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else "",
            "avg_form_accuracy": round(self.avg_form_accuracy, 1)
        }


class Badge(db.Model):
    __tablename__ = 'badges'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    icon = db.Column(db.String(50), nullable=False, default='🏆')
    unlocked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "unlocked_at": self.unlocked_at.strftime("%Y-%m-%d %H:%M:%S") if self.unlocked_at else ""
        }


AVAILABLE_BADGES = [
    {
        "id": "first_step",
        "name": "First Step",
        "icon": "🌱",
        "description": "Completed your first workout session",
        "requirement": "Complete 1 workout"
    },
    {
        "id": "century_club",
        "name": "Century Club",
        "icon": "💯",
        "description": "Completed 100 total repetitions across workouts",
        "requirement": "Reach 100 total reps"
    },
    {
        "id": "form_perfectionist",
        "name": "Form Perfectionist",
        "icon": "🎯",
        "description": "Achieved 90%+ average form accuracy",
        "requirement": "Average form ≥ 90%"
    },
    {
        "id": "iron_legs",
        "name": "Iron Legs",
        "icon": "🦵",
        "description": "Completed 25+ Squats in total",
        "requirement": "Log 25+ Squats"
    },
    {
        "id": "pushup_pro",
        "name": "Push Prodigy",
        "icon": "💪",
        "description": "Completed 25+ Push-Ups in total",
        "requirement": "Log 25+ Push-Ups"
    },
    {
        "id": "bicep_beast",
        "name": "Bicep Beast",
        "icon": "⚡",
        "description": "Completed 25+ Dumbbell Curls in total",
        "requirement": "Log 25+ Curls"
    },
    {
        "id": "dedicated_athlete",
        "name": "Dedicated Athlete",
        "icon": "🔥",
        "description": "Logged 5 or more workout sessions",
        "requirement": "Complete 5 workouts"
    },
    {
        "id": "level_5_champion",
        "name": "Level 5 Champion",
        "icon": "👑",
        "description": "Reached Level 5 in Sisyphus AI",
        "requirement": "Reach Level 5"
    }
]


def check_and_award_badges(user):
    """Evaluate and award persistent SQLite badges for user."""
    user_sessions = ExerciseSession.query.filter_by(user_id=user.id).all()
    total_reps = sum(s.rep_count for s in user_sessions)
    total_workouts = len(user_sessions)
    avg_form = (
        round(sum(s.avg_form_accuracy for s in user_sessions) / total_workouts, 1)
        if total_workouts > 0
        else 100.0
    )

    squat_reps = sum(s.rep_count for s in user_sessions if "squat" in s.exercise_type.lower())
    pushup_reps = sum(s.rep_count for s in user_sessions if "push" in s.exercise_type.lower())
    curl_reps = sum(s.rep_count for s in user_sessions if "curl" in s.exercise_type.lower())

    existing_badge_names = {b.name for b in user.badges}
    new_badges = []

    badge_rules = [
        ("First Step", "Completed your first workout session", "🌱", total_workouts >= 1),
        ("Century Club", "Completed 100 total repetitions across workouts", "💯", total_reps >= 100),
        ("Form Perfectionist", "Achieved 90%+ average form accuracy", "🎯", total_workouts >= 1 and avg_form >= 90.0),
        ("Iron Legs", "Completed 25+ Squats in total", "🦵", squat_reps >= 25),
        ("Push Prodigy", "Completed 25+ Push-Ups in total", "💪", pushup_reps >= 25),
        ("Bicep Beast", "Completed 25+ Dumbbell Curls in total", "⚡", curl_reps >= 25),
        ("Dedicated Athlete", "Logged 5 or more workout sessions", "🔥", total_workouts >= 5),
        ("Level 5 Champion", "Reached Level 5 in Sisyphus AI", "👑", user.level >= 5),
    ]

    for name, desc, icon, condition in badge_rules:
        if name not in existing_badge_names and condition:
            b = Badge(user_id=user.id, name=name, description=desc, icon=icon)
            db.session.add(b)
            new_badges.append(b)

    if new_badges:
        db.session.commit()
    return new_badges


def init_db():
    """Ensure database schema has required columns and tables."""
    with app.app_context():
        try:
            from sqlalchemy import text
            with db.engine.connect() as conn:
                res = conn.execute(text("PRAGMA table_info(users)"))
                cols = [row[1] for row in res.fetchall()]
                if cols:
                    if 'exp' not in cols:
                        conn.execute(text("ALTER TABLE users ADD COLUMN exp INTEGER DEFAULT 0 NOT NULL"))
                    if 'level' not in cols:
                        conn.execute(text("ALTER TABLE users ADD COLUMN level INTEGER DEFAULT 1 NOT NULL"))
                    conn.commit()
        except Exception as e:
            print(f"[!] DB migration notice: {e}")
        db.create_all()


init_db()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.template_filter('format_duration')
def format_duration(seconds):
    if not seconds:
        return "0s"
    seconds = int(round(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


# ---------------------------------------------------------
# TRIGONOMETRY & SIGNAL PROCESSING
# ---------------------------------------------------------
def calculate_angle(a, b, c):
    """
    Calculate 2D planar angle at joint b (in degrees) given points a, b, c.
    Returns angle in range [0, 180].
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    if ang > 180.0:
        ang = 360.0 - ang
    return float(ang)


def calculate_elbow_flare(shoulder, hip, elbow):
    """
    Calculate elbow flare angle relative to torso axis (Shoulder -> Hip).
    0 deg: elbow along ribs/torso; 45 deg: ideal form; 90 deg: flared T-shape.
    """
    v_torso = np.array(hip, dtype=np.float32) - np.array(shoulder, dtype=np.float32)
    v_arm = np.array(elbow, dtype=np.float32) - np.array(shoulder, dtype=np.float32)

    norms = np.linalg.norm(v_torso) * np.linalg.norm(v_arm)
    if norms == 0:
        return 0.0
    cos_ang = np.clip(np.dot(v_torso, v_arm) / norms, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def ema_filter(current_val, prev_val, alpha=0.35):
    """Exponential Moving Average (EMA) smoothing across consecutive frames."""
    if prev_val is None:
        return current_val
    return float(alpha * current_val + (1.0 - alpha) * prev_val)


class MockLandmark:
    """Mock landmark wrapper for synthetic animation frames."""
    def __init__(self, x, y, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.visibility = float(visibility)


# ---------------------------------------------------------
# EXERCISE FORM & REPETITION TRACKER
# ---------------------------------------------------------
class ExerciseTracker:
    def __init__(self, model_path=MODEL_PATH):
        self.lock = threading.RLock()
        self.exercise = "squat"  # "squat", "pushup", or "curl"
        self.counter = 0
        self.stage = "up"        # for squat/pushup: "up"/"down"; for curl: "down" (extended) / "up" (curled)

        # Filtered Joint Angles
        self.smoothed_primary_angle = 180.0
        self.smoothed_secondary_angle = 180.0
        self.smoothed_flare_angle = 0.0

        # Form Quality & Analytics
        self.form_score = 100
        self.rep_score_deductions = set()
        self.form_scores_history = []
        self.session_start_time = time.time()
        self.total_pose_frames = 0
        self.warning_pose_frames = 0

        self.feedback = "Ready - Position body in frame"
        self.warning_flags = []

        # Smart Audio Feedback: ONLY set on form errors
        # Silent on movement updates, starting positions, and successful reps
        self.audio_event = None
        self.vocal_message = ""
        self.latest_audio_alert = ""
        self.latest_alert_time = 0.0
        self.last_audio_trigger_time = 0.0

        self.min_angle_in_rep = 180.0
        self.max_angle_in_rep = 180.0
        self.active_side = "right"
        self.last_rep_time = 0
        self.last_landmarks = None

        # MediaPipe Pose Task
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def set_exercise(self, ex):
        with self.lock:
            if ex in ["squat", "pushup", "curl"] and ex != self.exercise:
                self.exercise = ex
                self.counter = 0
                self.stage = "down" if ex == "curl" else "up"
                self.form_score = 100
                self.form_scores_history.clear()
                self.session_start_time = time.time()
                self.total_pose_frames = 0
                self.warning_pose_frames = 0
                self.rep_score_deductions.clear()
                self.smoothed_primary_angle = 180.0
                self.smoothed_secondary_angle = 180.0
                self.smoothed_flare_angle = 0.0
                self.min_angle_in_rep = 180.0
                self.max_angle_in_rep = 180.0
                self.warning_flags.clear()
                self.feedback = f"{ex.capitalize()} mode activated"
                # SILENT: No vocal alerts on mode switch
                self.audio_event = None
                self.vocal_message = ""
                self.latest_audio_alert = ""
                self.latest_alert_time = 0.0
                self.last_landmarks = None

    def reset_reps(self):
        with self.lock:
            self.counter = 0
            self.stage = "down" if self.exercise == "curl" else "up"
            self.form_score = 100
            self.form_scores_history.clear()
            self.session_start_time = time.time()
            self.total_pose_frames = 0
            self.warning_pose_frames = 0
            self.rep_score_deductions.clear()
            self.min_angle_in_rep = 180.0
            self.max_angle_in_rep = 180.0
            self.warning_flags.clear()
            self.feedback = "Counter reset to 0."
            # SILENT: No vocal alerts on reset
            self.audio_event = None
            self.vocal_message = ""
            self.latest_audio_alert = ""
            self.latest_alert_time = 0.0
            self.last_landmarks = None

    def get_duration(self):
        return max(0.0, time.time() - self.session_start_time)

    def get_avg_accuracy(self):
        if self.form_scores_history:
            return float(np.mean(self.form_scores_history))
        return float(self.form_score)

    def get_status(self):
        with self.lock:
            elapsed = self.get_duration()
            avg_acc = self.get_avg_accuracy()
            return {
                "exercise": self.exercise,
                "reps": self.counter,
                "stage": self.stage,
                "primary_angle": round(self.smoothed_primary_angle, 1),
                "secondary_angle": round(self.smoothed_secondary_angle, 1),
                "flare_angle": round(self.smoothed_flare_angle, 1),
                "form_score": int(self.form_score),
                "avg_form_accuracy": round(avg_acc, 1),
                "duration_seconds": round(elapsed, 1),
                "feedback": self.feedback,
                "warnings": list(self.warning_flags),
                "audio_event": self.audio_event,
                "vocal_message": self.vocal_message,
                "active_side": self.active_side
            }

    def process_frame(self, frame, custom_landmarks=None, skip_pose=False):
        h, w, _ = frame.shape
        landmarks = custom_landmarks

        if landmarks is None:
            if skip_pose and self.last_landmarks is not None:
                # Frame skipping optimization: reuse cached pose tracking results
                landmarks = self.last_landmarks
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect(mp_image)
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]
                    self.last_landmarks = landmarks
                else:
                    landmarks = None
                    self.last_landmarks = None

        with self.lock:
            audio_ev = None
            vocal_msg = None

            if landmarks is not None:
                self.warning_flags = []
                self.total_pose_frames += 1

                if self.exercise == "squat":
                    audio_ev, vocal_msg = self._process_squat(landmarks, w, h)
                elif self.exercise == "pushup":
                    audio_ev, vocal_msg = self._process_pushup(landmarks, w, h)
                else:
                    audio_ev, vocal_msg = self._process_curl(landmarks, w, h)

                if self.warning_flags or (audio_ev and audio_ev.startswith("warn_")):
                    self.warning_pose_frames += 1

                self._draw_skeleton(frame, landmarks, w, h)
            else:
                self.feedback = "No person detected. Position body in frame."
                self.warning_flags = ["no_person"]

            # 4-second throttle check for backend alerts
            now = time.time()
            if audio_ev and audio_ev.startswith("warn_") and vocal_msg:
                if now - self.last_audio_trigger_time >= 4.0:
                    self.audio_event = audio_ev
                    self.vocal_message = vocal_msg
                    self.latest_audio_alert = vocal_msg
                    self.latest_alert_time = now
                    self.last_audio_trigger_time = now
                else:
                    self.audio_event = None
                    self.vocal_message = ""
            else:
                self.audio_event = None
                self.vocal_message = ""

            self._draw_hud(frame)

        return frame

    def _process_squat(self, landmarks, w, h):
        """
        Squat tracking:
        Speaks ONLY on specific form errors:
          - Depth fail: 'Squat deeper!'
          - Knee cave: 'Keep your knees out!'
        Silent on standard movement, descent, and rep counts.
        """
        l_hip = [landmarks[23].x * w, landmarks[23].y * h]
        l_knee = [landmarks[25].x * w, landmarks[25].y * h]
        l_ankle = [landmarks[27].x * w, landmarks[27].y * h]

        r_hip = [landmarks[24].x * w, landmarks[24].y * h]
        r_knee = [landmarks[26].x * w, landmarks[26].y * h]
        r_ankle = [landmarks[28].x * w, landmarks[28].y * h]

        l_ang = calculate_angle(l_hip, l_knee, l_ankle)
        r_ang = calculate_angle(r_hip, r_knee, r_ankle)

        r_vis = getattr(landmarks[26], 'visibility', 1.0)
        l_vis = getattr(landmarks[25], 'visibility', 1.0)
        if r_vis >= l_vis:
            raw_knee_angle = r_ang
            self.active_side = "right"
        else:
            raw_knee_angle = l_ang
            self.active_side = "left"

        self.smoothed_primary_angle = ema_filter(raw_knee_angle, self.smoothed_primary_angle, alpha=0.35)
        knee_angle = self.smoothed_primary_angle

        # Knee Cave (Valgus) check
        knee_dist = abs(landmarks[25].x - landmarks[26].x)
        ankle_dist = abs(landmarks[27].x - landmarks[28].x)
        hip_dist = abs(landmarks[23].x - landmarks[24].x)

        knee_cave = False
        if knee_angle < 140.0 and ankle_dist > 0.08:
            if knee_dist < 0.72 * ankle_dist or knee_dist < 0.80 * hip_dist:
                knee_cave = True
                self.warning_flags.append("Knees caving inward! Push knees out.")
                self.rep_score_deductions.add("knee_cave")

        audio_ev = None
        vocal_msg = None

        DOWN_THRESH = 100.0
        UP_THRESH = 160.0

        if self.stage == "up":
            if knee_angle < 140.0:
                self.stage = "down"
                self.min_angle_in_rep = knee_angle
                self.rep_score_deductions.clear()
                self.feedback = "Descending... Lower hips to 100°"
            else:
                self.feedback = "Standing: Begin squat"
        elif self.stage == "down":
            self.min_angle_in_rep = min(self.min_angle_in_rep, knee_angle)

            if knee_angle <= DOWN_THRESH:
                self.feedback = "Good Depth! Now push up"
            elif knee_angle > 115.0 and self.min_angle_in_rep > DOWN_THRESH:
                self.feedback = "Squat deeper! Knee angle must reach < 100°"
                self.warning_flags.append("Shallow squat depth (< 100° required)")
                self.rep_score_deductions.add("shallow_depth")

            if knee_cave:
                self.feedback = "Knees caving! Keep knees wide"

            # Rep Completion: Visual update only (SILENT: no audio)
            if knee_angle >= UP_THRESH:
                self.stage = "up"
                self.counter += 1
                self.last_rep_time = time.time()

                deductions = 0
                if "shallow_depth" in self.rep_score_deductions:
                    deductions += 30
                if "knee_cave" in self.rep_score_deductions:
                    deductions += 25

                rep_score = max(30, 100 - deductions)
                self.form_scores_history.append(rep_score)
                self.form_score = int(0.7 * self.form_score + 0.3 * rep_score)

                if deductions == 0:
                    self.feedback = f"Rep #{self.counter} Complete! Perfect Depth (100%)"
                else:
                    critique = "Squat deeper next time." if "shallow_depth" in self.rep_score_deductions else "Keep knees out."
                    self.feedback = f"Rep #{self.counter} Complete ({rep_score}% Form) - {critique}"

                # ZERO speech on rep completion
                audio_ev = None
                vocal_msg = None

        # Form Errors: Trigger speech ONLY on specific form errors
        if audio_ev is None:
            if knee_cave:
                audio_ev = "warn_kneecave"
                vocal_msg = "Keep your knees out!"
            elif self.stage == "down" and knee_angle < 120.0 and self.min_angle_in_rep > 105.0:
                audio_ev = "warn_depth"
                vocal_msg = "Squat deeper!"

        return audio_ev, vocal_msg

    def _process_pushup(self, landmarks, w, h):
        """
        Push-up tracking:
        Speaks ONLY on specific form errors:
          - Sagging back / hip alignment: 'Raise your hips!'
          - Chest depth fail: 'Lower your chest!'
        Silent on standard movement and rep counts.
        """
        l_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        l_elbow = [landmarks[13].x * w, landmarks[13].y * h]
        l_wrist = [landmarks[15].x * w, landmarks[15].y * h]
        l_hip = [landmarks[23].x * w, landmarks[23].y * h]
        l_ankle = [landmarks[27].x * w, landmarks[27].y * h]

        r_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        r_elbow = [landmarks[14].x * w, landmarks[14].y * h]
        r_wrist = [landmarks[16].x * w, landmarks[16].y * h]
        r_hip = [landmarks[24].x * w, landmarks[24].y * h]
        r_ankle = [landmarks[28].x * w, landmarks[28].y * h]

        l_vis = getattr(landmarks[13], 'visibility', 1.0)
        r_vis = getattr(landmarks[14], 'visibility', 1.0)

        if r_vis >= l_vis:
            shoulder, elbow, wrist, hip, ankle = r_shoulder, r_elbow, r_wrist, r_hip, r_ankle
            self.active_side = "right"
        else:
            shoulder, elbow, wrist, hip, ankle = l_shoulder, l_elbow, l_wrist, l_hip, l_ankle
            self.active_side = "left"

        raw_elbow_angle = calculate_angle(shoulder, elbow, wrist)
        raw_spine_angle = calculate_angle(shoulder, hip, ankle)
        raw_flare_angle = calculate_elbow_flare(shoulder, hip, elbow)

        self.smoothed_primary_angle = ema_filter(raw_elbow_angle, self.smoothed_primary_angle, alpha=0.35)
        self.smoothed_secondary_angle = ema_filter(raw_spine_angle, self.smoothed_secondary_angle, alpha=0.35)
        self.smoothed_flare_angle = ema_filter(raw_flare_angle, self.smoothed_flare_angle, alpha=0.35)

        elbow_angle = self.smoothed_primary_angle
        spine_angle = self.smoothed_secondary_angle
        flare_angle = self.smoothed_flare_angle

        # Sagging back check: Hip-Shoulder-Ankle angle too low (< 160 deg)
        back_sags = False
        if spine_angle < 160.0:
            back_sags = True
            self.warning_flags.append("Hips sagging! Raise hips to keep straight plank.")
            self.rep_score_deductions.add("spine_sag")

        elbow_flare = False
        if elbow_angle < 140.0 and flare_angle > 75.0:
            elbow_flare = True
            self.warning_flags.append("Elbows flaring! Tuck elbows to 45°.")
            self.rep_score_deductions.add("elbow_flare")

        audio_ev = None
        vocal_msg = None

        DOWN_THRESH = 100.0
        UP_THRESH = 160.0

        if self.stage == "up":
            if elbow_angle < 135.0:
                self.stage = "down"
                self.min_angle_in_rep = elbow_angle
                self.rep_score_deductions.clear()
                self.feedback = "Descending... Lower chest to 100°"
            else:
                self.feedback = "Plank: Begin push-up"
        elif self.stage == "down":
            self.min_angle_in_rep = min(self.min_angle_in_rep, elbow_angle)

            if elbow_angle <= DOWN_THRESH:
                self.feedback = "Great Depth! Now press up"
            elif elbow_angle > 115.0 and self.min_angle_in_rep > DOWN_THRESH:
                self.feedback = "Chest lower! Elbow angle must reach < 100°"
                self.warning_flags.append("Insufficient chest depth (< 100° required)")
                self.rep_score_deductions.add("shallow_chest")

            if back_sags:
                self.feedback = "Hips sagging! Raise hips to keep straight plank"

            # Rep Completion: Visual update only (SILENT: no audio)
            if elbow_angle >= UP_THRESH:
                self.stage = "up"
                self.counter += 1
                self.last_rep_time = time.time()

                deductions = 0
                if "shallow_chest" in self.rep_score_deductions:
                    deductions += 30
                if "spine_sag" in self.rep_score_deductions:
                    deductions += 25
                if "elbow_flare" in self.rep_score_deductions:
                    deductions += 20

                rep_score = max(25, 100 - deductions)
                self.form_scores_history.append(rep_score)
                self.form_score = int(0.7 * self.form_score + 0.3 * rep_score)

                if deductions == 0:
                    self.feedback = f"Rep #{self.counter} Complete! Solid Form (100%)"
                else:
                    critique = "Raise hips next time." if "spine_sag" in self.rep_score_deductions else "Lower chest next time."
                    self.feedback = f"Rep #{self.counter} Complete ({rep_score}% Form) - {critique}"

                # ZERO speech on rep completion
                audio_ev = None
                vocal_msg = None

        # Form Errors: Trigger speech ONLY on specific form errors
        if audio_ev is None:
            if back_sags:
                audio_ev = "warn_back"
                vocal_msg = "Raise your hips!"
            elif self.stage == "down" and elbow_angle < 120.0 and self.min_angle_in_rep > 105.0:
                audio_ev = "warn_chest"
                vocal_msg = "Lower your chest!"

        return audio_ev, vocal_msg

    def _process_curl(self, landmarks, w, h):
        """
        Dumbbell Curls tracking:
        - Track arm angle (Shoulder-Elbow-Wrist).
        - Increment rep count ONLY with strict hysteresis:
          Moving from extension (>150 deg) to full curl (<40 deg), then returning to extension (>150 deg).
        - Form Error: If elbow moves excessively away from torso (>35 deg), speak 'Keep your elbows tucked!'.
        """
        l_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        l_elbow = [landmarks[13].x * w, landmarks[13].y * h]
        l_wrist = [landmarks[15].x * w, landmarks[15].y * h]
        l_hip = [landmarks[23].x * w, landmarks[23].y * h]

        r_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        r_elbow = [landmarks[14].x * w, landmarks[14].y * h]
        r_wrist = [landmarks[16].x * w, landmarks[16].y * h]
        r_hip = [landmarks[24].x * w, landmarks[24].y * h]

        l_vis = getattr(landmarks[13], 'visibility', 1.0)
        r_vis = getattr(landmarks[14], 'visibility', 1.0)

        if r_vis >= l_vis:
            shoulder, elbow, wrist, hip = r_shoulder, r_elbow, r_wrist, r_hip
            self.active_side = "right"
        else:
            shoulder, elbow, wrist, hip = l_shoulder, l_elbow, l_wrist, l_hip
            self.active_side = "left"

        raw_arm_angle = calculate_angle(shoulder, elbow, wrist)
        raw_flare_angle = calculate_elbow_flare(shoulder, hip, elbow)

        self.smoothed_primary_angle = ema_filter(raw_arm_angle, self.smoothed_primary_angle, alpha=0.35)
        self.smoothed_flare_angle = ema_filter(raw_flare_angle, self.smoothed_flare_angle, alpha=0.35)
        self.smoothed_secondary_angle = 40.0

        arm_angle = self.smoothed_primary_angle
        flare_angle = self.smoothed_flare_angle

        # Form error check: Elbow flared (> 35 deg lateral angle from torso)
        elbow_flared = False
        if flare_angle > 35.0:
            elbow_flared = True
            self.warning_flags.append("Elbow flaring outward! Keep elbows tucked.")
            self.rep_score_deductions.add("elbow_flare")

        audio_ev = None
        vocal_msg = None

        EXT_THRESH = 150.0
        CURL_THRESH = 40.0

        if self.stage == "down":
            self.max_angle_in_rep = max(self.max_angle_in_rep, arm_angle)

            # Reached full curl contraction (< 40 deg)
            if arm_angle <= CURL_THRESH:
                self.stage = "up"
                self.min_angle_in_rep = arm_angle
                self.feedback = "Peak Curl Reached (<40°)! Now lower slowly"
            elif arm_angle < 90.0:
                self.feedback = "Curling up... Target < 40°"
            else:
                self.feedback = "Arm extended (>150°). Begin curl"

        elif self.stage == "up":
            # Moving down towards full extension
            if arm_angle >= EXT_THRESH:
                # Rep Complete upon full extension (> 150 deg)
                self.stage = "down"
                self.counter += 1
                self.last_rep_time = time.time()
                self.max_angle_in_rep = arm_angle

                deductions = 0
                if "elbow_flare" in self.rep_score_deductions:
                    deductions += 30

                rep_score = max(40, 100 - deductions)
                self.form_scores_history.append(rep_score)
                self.form_score = int(0.7 * self.form_score + 0.3 * rep_score)
                self.rep_score_deductions.clear()

                if deductions == 0:
                    self.feedback = f"Rep #{self.counter} Complete! Full Extension & Curl (100%)"
                else:
                    self.feedback = f"Rep #{self.counter} Complete ({rep_score}% Form) - Keep elbows tucked next time"

                # ZERO speech on rep completion
                audio_ev = None
                vocal_msg = None
            elif arm_angle > 80.0:
                self.feedback = "Extending arm... Lower past 150°"
            else:
                self.feedback = "Lower arm with control"

        # Form Errors: Trigger speech ONLY on specific form errors
        if audio_ev is None:
            if elbow_flared:
                audio_ev = "warn_elbow_tuck"
                vocal_msg = "Keep your elbows tucked!"

        return audio_ev, vocal_msg

    def _draw_skeleton(self, frame, landmarks, w, h):
        """Minimal keypoint skeleton overlay with thinner lines and clean joint highlights."""
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        color = (0, 255, 198) if not self.warning_flags else (0, 165, 255)
        for p1, p2 in connections:
            if p1 < len(landmarks) and p2 < len(landmarks):
                pt1 = (int(landmarks[p1].x * w), int(landmarks[p1].y * h))
                pt2 = (int(landmarks[p2].x * w), int(landmarks[p2].y * h))
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(landmarks):
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame):
        """
        Minimal HUD pass-through:
        Heavy dark solid OpenCV rectangles are replaced by modern sleek
        floating glassmorphism CSS UI over the frame container in the browser.
        """
        pass


# ---------------------------------------------------------
# SYNTHETIC DEMO GENERATOR ENGINE
# ---------------------------------------------------------
class SyntheticDemoEngine:
    """
    Generates synthetic realistic animated pose frames for Dumbbell Curls,
    Squats, and Push-ups if physical webcam or video assets are unavailable.
    """
    def __init__(self, width=640, height=480):
        self.w = width
        self.h = height

    def generate(self, exercise, t):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Background gym styling
        cv2.rectangle(frame, (0, 0), (self.w, self.h), (12, 17, 28), -1)
        cv2.line(frame, (0, int(self.h * 0.88)), (self.w, int(self.h * 0.88)), (25, 35, 55), 2)

        # 33 MediaPipe pose landmarks list initialization
        landmarks = [MockLandmark(0.5, 0.5) for _ in range(33)]

        if exercise == "curl":
            cycle = (math.sin(t * 2.1) + 1.0) / 2.0
            arm_angle_deg = 160.0 - cycle * 125.0
            arm_rad = math.radians(arm_angle_deg)

            sh_x, sh_y = 350, 170
            el_x, el_y = 355, 270
            wr_x = el_x - int(85 * math.sin(arm_rad))
            wr_y = el_y - int(85 * math.cos(arm_rad))

            landmarks[11] = MockLandmark(290 / self.w, 170 / self.h)
            landmarks[12] = MockLandmark(sh_x / self.w, sh_y / self.h)
            landmarks[13] = MockLandmark(285 / self.w, 270 / self.h)
            landmarks[14] = MockLandmark(el_x / self.w, el_y / self.h)
            landmarks[15] = MockLandmark(285 / self.w, 350 / self.h)
            landmarks[16] = MockLandmark(wr_x / self.w, wr_y / self.h)
            landmarks[23] = MockLandmark(300 / self.w, 290 / self.h)
            landmarks[24] = MockLandmark(340 / self.w, 290 / self.h)
            landmarks[25] = MockLandmark(300 / self.w, 380 / self.h)
            landmarks[26] = MockLandmark(340 / self.w, 380 / self.h)
            landmarks[27] = MockLandmark(300 / self.w, 440 / self.h)
            landmarks[28] = MockLandmark(340 / self.w, 440 / self.h)

            cv2.circle(frame, (wr_x - 12, wr_y), 10, (140, 150, 160), -1)
            cv2.circle(frame, (wr_x + 12, wr_y), 10, (140, 150, 160), -1)
            cv2.line(frame, (wr_x - 12, wr_y), (wr_x + 12, wr_y), (200, 210, 220), 4)

        elif exercise == "squat":
            cycle = (math.sin(t * 2.0) + 1.0) / 2.0
            hip_y = int(240 + cycle * 85)
            knee_y = 360
            knee_x = int(320 - cycle * 35)

            landmarks[11] = MockLandmark(310 / self.w, (hip_y - 100) / self.h)
            landmarks[12] = MockLandmark(330 / self.w, (hip_y - 100) / self.h)
            landmarks[23] = MockLandmark(310 / self.w, hip_y / self.h)
            landmarks[24] = MockLandmark(330 / self.w, hip_y / self.h)
            landmarks[25] = MockLandmark(knee_x / self.w, knee_y / self.h)
            landmarks[26] = MockLandmark((knee_x + 20) / self.w, knee_y / self.h)
            landmarks[27] = MockLandmark(320 / self.w, 440 / self.h)
            landmarks[28] = MockLandmark(340 / self.w, 440 / self.h)

        else:  # pushup
            cycle = (math.sin(t * 2.0) + 1.0) / 2.0
            sh_y = int(230 + cycle * 70)
            hip_y = int(250 + cycle * 50)

            landmarks[11] = MockLandmark(230 / self.w, sh_y / self.h)
            landmarks[12] = MockLandmark(240 / self.w, sh_y / self.h)
            landmarks[13] = MockLandmark(210 / self.w, (sh_y + 40) / self.h)
            landmarks[14] = MockLandmark(220 / self.w, (sh_y + 40) / self.h)
            landmarks[15] = MockLandmark(240 / self.w, 360 / self.h)
            landmarks[16] = MockLandmark(250 / self.w, 360 / self.h)
            landmarks[23] = MockLandmark(360 / self.w, hip_y / self.h)
            landmarks[24] = MockLandmark(370 / self.w, hip_y / self.h)
            landmarks[27] = MockLandmark(480 / self.w, 340 / self.h)
            landmarks[28] = MockLandmark(490 / self.w, 340 / self.h)

        return frame, landmarks


# ---------------------------------------------------------
# VIDEO STREAM & RESOURCE LIFECYCLE MANAGER
# ---------------------------------------------------------
class VideoStreamManager:
    def __init__(self, tracker):
        self.tracker = tracker
        self.cap = None
        self.source_mode = "demo"  # "demo", "webcam", or "browser_cam"
        self.latest_frame_bytes = None
        self.lock = threading.RLock()
        self.running = True
        self.synthetic_engine = SyntheticDemoEngine(640, 480)
        self.synthetic_time = 0.0
        self.active_viewers = 0
        self.is_paused = False
        self.frame_count = 0

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _get_demo_path(self):
        if self.tracker.exercise == "pushup" and os.path.exists(PUSHUP_DEMO_PATH):
            return PUSHUP_DEMO_PATH
        if self.tracker.exercise == "squat" and os.path.exists(SQUAT_DEMO_PATH):
            return SQUAT_DEMO_PATH
        if self.tracker.exercise == "curl" and os.path.exists(DUMBBELL_DEMO_PATH):
            return DUMBBELL_DEMO_PATH
        return None

    def _open_source(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.source_mode == "webcam":
            print("[*] Opening physical webcam cv2.VideoCapture(0)...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[!] Physical webcam device not found, falling back to demo engine.")
                self.source_mode = "demo"
                demo_path = self._get_demo_path()
                if demo_path:
                    self.cap = cv2.VideoCapture(demo_path)
        elif self.source_mode == "demo":
            demo_path = self._get_demo_path()
            if demo_path:
                print(f"[*] Opening demo video source: {demo_path}")
                self.cap = cv2.VideoCapture(demo_path)

    def release_camera(self):
        """Cleanly release VideoCapture hardware/file descriptor resources."""
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                print("[+] VideoCapture resource successfully released.")

    def add_viewer(self):
        """Track active stream viewer and start camera if needed."""
        with self.lock:
            self.active_viewers += 1
            self.is_paused = False
            if self.cap is None and self.source_mode in ["demo", "webcam"]:
                self._open_source()

    def remove_viewer(self):
        """Untrack viewer; automatically release camera when no clients are viewing."""
        with self.lock:
            self.active_viewers = max(0, self.active_viewers - 1)
            if self.active_viewers == 0:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    print("[+] All stream clients disconnected. Camera VideoCapture released.")

    def pause(self):
        """Pause stream and release camera."""
        self.is_paused = True
        self.release_camera()

    def resume(self):
        """Resume stream and re-open video source."""
        with self.lock:
            self.is_paused = False
            if self.active_viewers > 0 and self.cap is None:
                self._open_source()

    def set_source(self, mode):
        with self.lock:
            self.source_mode = mode
            if self.active_viewers > 0 and not self.is_paused:
                self._open_source()
            elif mode == "browser_cam":
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
            self.tracker.reset_reps()

    def update_demo_for_exercise(self):
        with self.lock:
            if self.source_mode == "demo" and self.active_viewers > 0 and not self.is_paused:
                self._open_source()

    def _capture_loop(self):
        target_fps = 30.0
        frame_interval = 1.0 / target_fps

        while self.running:
            # If in browser camera mode, or paused, or no viewers connected, sleep and conserve CPU
            if self.source_mode == "browser_cam" or self.is_paused or self.active_viewers == 0:
                time.sleep(0.05)
                continue

            start_t = time.time()
            self.synthetic_time += frame_interval
            self.frame_count += 1

            frame = None
            custom_landmarks = None

            with self.lock:
                if self.cap is None and self.source_mode in ["webcam", "demo"]:
                    self._open_source()

                # 1. Try physical/demo capture if available
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()

                # 2. Seamless Synthetic Demo Fallback if no camera/video exists (e.g. for Dumbbell Curls)
                if frame is None:
                    frame, custom_landmarks = self.synthetic_engine.generate(
                        self.tracker.exercise, self.synthetic_time
                    )

            # Performance Optimization: Resize incoming frame to 480x360 before MediaPipe processing to reduce CPU/GPU load
            h, w = frame.shape[:2]
            if w != 480 or h != 360:
                frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)

            # Performance Optimization: Process pose tracking every 2nd frame (frame skipping) to maintain smooth 30+ FPS stream playback
            skip_pose = (self.frame_count % 2 != 0)
            processed = self.tracker.process_frame(
                frame,
                custom_landmarks=custom_landmarks,
                skip_pose=skip_pose
            )

            # Bandwidth Optimization: Set JPEG encoding compression parameter to quality 65
            ret, buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                with self.lock:
                    self.latest_frame_bytes = buf.tobytes()

            elapsed = time.time() - start_t
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def generate_frames(self):
        """Yield multipart JPEG frames and properly release camera resource when disconnected."""
        self.add_viewer()
        last_yielded = None
        try:
            while self.running and not self.is_paused:
                with self.lock:
                    f_bytes = self.latest_frame_bytes

                if f_bytes and f_bytes != last_yielded:
                    last_yielded = f_bytes
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + f_bytes + b'\r\n')
                time.sleep(0.02)
        finally:
            self.remove_viewer()


tracker = ExerciseTracker()
stream_mgr = VideoStreamManager(tracker)


def generate_frames():
    """Global video stream generator."""
    return stream_mgr.generate_frames()


# ---------------------------------------------------------
# BENTO BOX MODERN UI TEMPLATES (CREXTIO INSPIRED)
# ---------------------------------------------------------
LAYOUT_SHELL = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sisyphus AI Coach - Smart Form Tracker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-main: #f7f7f8;
      --card-bg: #ffffff;
      --card-border: #e5e7eb;
      --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.04), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
      --card-shadow-hover: 0 20px 30px -10px rgba(0, 0, 0, 0.07);
      --yellow: #facc15;
      --yellow-hover: #eab308;
      --yellow-light: #fef08a;
      --yellow-subtle: rgba(250, 204, 21, 0.16);
      --charcoal: #18181b;
      --charcoal-hover: #27272a;
      --charcoal-border: #27272a;
      --text-primary: #18181b;
      --text-secondary: #71717a;
      --text-muted: #a1a1aa;
      --accent-green: #10b981;
      --accent-red: #ef4444;
      --accent-blue: #3b82f6;
      --radius-bento: 20px;
      --radius-pill: 9999px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at top right, rgba(250, 204, 21, 0.16), transparent 45%), var(--bg-main);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
    }
    .navbar-container {
      padding: 16px 24px;
      max-width: 1360px;
      margin: 0 auto;
      width: 100%;
    }
    .navbar-floating {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.9);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-pill);
      padding: 10px 24px;
      box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);
      flex-wrap: wrap;
      gap: 12px;
    }
    .logo-text {
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--text-primary);
      letter-spacing: -0.02em;
      text-decoration: none;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .logo-dot {
      width: 8px;
      height: 8px;
      background: var(--yellow);
      border-radius: 50%;
      display: inline-block;
    }
    .logo-badge {
      background: var(--yellow-subtle);
      border: 1px solid rgba(250, 204, 21, 0.5);
      color: #854d0e;
      font-size: 0.72rem;
      padding: 2px 10px;
      border-radius: var(--radius-pill);
      font-weight: 700;
    }
    .nav-pills {
      display: flex;
      align-items: center;
      gap: 6px;
      background: #f4f4f5;
      padding: 4px;
      border-radius: var(--radius-pill);
    }
    .nav-pill {
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 8px 18px;
      border-radius: var(--radius-pill);
      color: var(--text-secondary);
      transition: all 0.15s ease;
    }
    .nav-pill:hover {
      color: var(--text-primary);
    }
    .nav-pill.active {
      background: var(--charcoal);
      color: #ffffff;
      box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .user-badge-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .user-avatar {
      width: 36px;
      height: 36px;
      background: var(--charcoal);
      color: #ffffff;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.88rem;
      font-weight: 700;
    }
    .container {
      width: 100%;
      max-width: 1360px;
      margin: 8px auto 32px auto;
      padding: 0 24px;
      flex: 1;
      box-sizing: border-box;
    }
    .bento-card {
      background: var(--card-bg);
      border-radius: var(--radius-bento);
      border: 1px solid var(--card-border);
      box-shadow: var(--card-shadow);
      padding: 24px;
      box-sizing: border-box;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .bento-card:hover {
      box-shadow: var(--card-shadow-hover);
    }
    .bento-dark-card {
      background: var(--charcoal);
      color: #ffffff;
      border-radius: var(--radius-bento);
      border: 1px solid var(--charcoal-border);
      box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.15);
      padding: 24px;
      box-sizing: border-box;
    }
    .btn-yellow {
      background: var(--yellow);
      color: var(--charcoal);
      font-weight: 700;
      font-size: 0.88rem;
      border: none;
      border-radius: var(--radius-pill);
      padding: 10px 20px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
      text-decoration: none;
      box-shadow: 0 2px 8px rgba(250, 204, 21, 0.35);
    }
    .btn-yellow:hover {
      background: var(--yellow-hover);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(250, 204, 21, 0.45);
    }
    .btn-charcoal {
      background: var(--charcoal);
      color: #ffffff;
      font-weight: 700;
      font-size: 0.88rem;
      border: none;
      border-radius: var(--radius-pill);
      padding: 10px 20px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .btn-charcoal:hover {
      background: var(--charcoal-hover);
      transform: translateY(-1px);
    }
    .btn-ghost {
      background: #f4f4f5;
      color: var(--text-primary);
      font-weight: 600;
      font-size: 0.86rem;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-pill);
      padding: 9px 18px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .btn-ghost:hover {
      background: #e4e4e7;
    }
    .btn-mode-pill {
      background: #f4f4f5;
      color: var(--text-primary);
      font-weight: 700;
      font-size: 0.84rem;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-pill);
      padding: 8px 18px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn-mode-pill:hover {
      background: #e4e4e7;
    }
    .btn-mode-pill.active {
      background: var(--yellow);
      color: var(--charcoal);
      border-color: var(--yellow);
      box-shadow: 0 2px 8px rgba(250, 204, 21, 0.35);
    }
    .stat-figure {
      font-size: 2.5rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: var(--text-primary);
      line-height: 1.1;
      margin: 8px 0 4px 0;
    }
    .stat-label {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
    }
    .pill-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: var(--radius-pill);
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .pill-yellow {
      background: var(--yellow-subtle);
      color: #854d0e;
      border: 1px solid rgba(250, 204, 21, 0.4);
    }
    .pill-charcoal {
      background: var(--charcoal);
      color: #ffffff;
    }
    .pill-gray {
      background: #f4f4f5;
      color: var(--text-secondary);
      border: 1px solid var(--card-border);
    }
    .alert-box {
      padding: 14px 18px;
      border-radius: 14px;
      margin-bottom: 20px;
      font-size: 0.88rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .alert-success {
      background: rgba(16, 185, 129, 0.1);
      color: #065f46;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .alert-danger {
      background: rgba(239, 68, 68, 0.1);
      color: #991b1b;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    input[type="text"], input[type="password"] {
      width: 100%;
      padding: 13px 16px;
      border-radius: 14px;
      border: 1px solid var(--card-border);
      background: #ffffff;
      color: var(--text-primary);
      margin-bottom: 16px;
      box-sizing: border-box;
      font-family: inherit;
      font-size: 0.92rem;
      transition: all 0.15s ease;
    }
    input[type="text"]:focus, input[type="password"]:focus {
      outline: none;
      border-color: var(--yellow);
      box-shadow: 0 0 0 3px rgba(250, 204, 21, 0.25);
    }
  </style>
</head>
<body>
  <div class="navbar-container">
    <div class="navbar-floating">
      <a href="/" class="logo-text">
        <span>Sisyphus</span>
        <span class="logo-dot"></span>
        <span class="logo-badge">Smart Form</span>
      </a>

      <div class="nav-pills">
        {% if current_user.is_authenticated %}
          <a href="/dashboard" class="nav-pill active">📊 Dashboard</a>
          <a href="/workout" class="nav-pill">🏋️ Live Workout</a>
          <a href="/logout" class="nav-pill" style="color: var(--accent-red);">Logout</a>
        {% else %}
          <a href="/login" class="nav-pill active">Login</a>
          <a href="/register" class="nav-pill">Register</a>
        {% endif %}
      </div>

      <div class="user-badge-wrap">
        {% if current_user.is_authenticated %}
          <div class="user-avatar">{{ current_user.username[0]|upper }}</div>
          <div>
            <div style="font-size: 0.86rem; font-weight: 700; color: var(--text-primary); line-height: 1.2;">
              {{ current_user.username }}
            </div>
            <div style="font-size: 0.72rem; color: var(--text-secondary); font-weight: 600;">
              Level {{ current_user.level }}
            </div>
          </div>
        {% else %}
          <a href="/register" class="btn-yellow" style="padding: 7px 16px; font-size: 0.8rem;">Join Sisyphus</a>
        {% endif %}
      </div>
    </div>
  </div>

  <div class="container">
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
          <div class="alert-box alert-{{ category }}">
            <span>⚡</span>
            <span>{{ message }}</span>
          </div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <!--BODY_CONTENT-->
  </div>
</body>
</html>
"""


def render_layout(content, **kwargs):
    html = LAYOUT_SHELL.replace("<!--BODY_CONTENT-->", content)
    return render_template_string(html, **kwargs)


WORKOUT_BODY = """
<div style="display: grid; grid-template-columns: 1fr 390px; gap: 24px;">
  <!-- Video Feed & Bento Setup Card -->
  <div class="bento-card" style="padding: 24px; display: flex; flex-direction: column; gap: 18px;">
    <!-- Encapsulated Video Feed with Modern Floating Glassmorphism Camera HUD -->
    <div style="position: relative; width: 100%; aspect-ratio: 16 / 9; background: #09090b; border-radius: 16px; overflow: hidden; border: 1px solid var(--card-border); box-shadow: inset 0 2px 8px rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center;">
      <img id="video-feed-img" src="/video_feed" alt="MediaPipe Tracking Feed" style="width: 100%; height: 100%; object-fit: contain; display: block;" />

      <!-- Top Floating Glassmorphism HUD Bar -->
      <div style="position: absolute; top: 14px; left: 14px; right: 14px; display: flex; justify-content: space-between; align-items: center; pointer-events: none; z-index: 10;">
        <!-- Top Left: Exercise Mode & Phase Pill Badge -->
        <div style="display: flex; align-items: center; gap: 10px; background: rgba(24, 24, 27, 0.75); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 6px 14px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);">
          <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
          <span id="hud-exercise-mode" style="color: #ffffff; font-weight: 700; font-size: 0.84rem; letter-spacing: 0.04em; text-transform: uppercase;">Squats</span>
          <span id="hud-stage-pill" style="font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 8px; background: rgba(250, 204, 21, 0.2); color: #facc15; font-family: 'JetBrains Mono', monospace; transition: all 0.2s ease;">UP</span>
        </div>

        <!-- Top Right: Rep Counter Pill Badge with Warm Yellow Typography -->
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(24, 24, 27, 0.75); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 6px 16px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);">
          <span style="font-size: 0.7rem; font-weight: 700; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em;">REPS</span>
          <span id="hud-rep-count" style="color: #facc15; font-size: 1.45rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; transition: transform 0.18s ease;">0</span>
        </div>
      </div>

      <!-- Bottom Floating Glassmorphism HUD Bar: Form Status & Precision Metrics -->
      <div style="position: absolute; bottom: 14px; left: 14px; right: 14px; display: flex; justify-content: space-between; align-items: center; pointer-events: none; z-index: 10; background: rgba(24, 24, 27, 0.75); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 8px 16px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);">
        <div style="display: flex; align-items: center; gap: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          <span id="hud-status-icon" style="font-size: 0.95rem;">⚡</span>
          <span id="hud-form-status" style="color: #ffffff; font-size: 0.86rem; font-weight: 600; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">Ready - Position body in frame</span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px; flex-shrink: 0; margin-left: 12px;">
          <span id="hud-angle-pill" style="font-size: 0.78rem; font-weight: 700; color: #d1d5db; font-family: 'JetBrains Mono', monospace;">180°</span>
          <span id="hud-score-pill" style="font-size: 0.78rem; font-weight: 800; color: #facc15; background: rgba(250, 204, 21, 0.18); padding: 3px 8px; border-radius: 8px; font-family: 'JetBrains Mono', monospace;">100%</span>
        </div>
      </div>
    </div>

    <!-- Mode Selector Bar (Warm Yellow Buttons & Pills) -->
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; background: #fbfbfb; padding: 14px 18px; border-radius: 16px; border: 1px solid var(--card-border);">
      <div>
        <div style="font-size: 0.74rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Exercise Mode</div>
        <div class="exercise-pill-group" style="display: flex; gap: 8px; flex-wrap: wrap;">
          <button type="button" class="btn-mode-pill active" data-exercise="squat">🏋️ Squats</button>
          <button type="button" class="btn-mode-pill" data-exercise="pushup">💪 Push-ups</button>
          <button type="button" class="btn-mode-pill" data-exercise="curl">⚡ Dumbbell Curls</button>
        </div>
        <!-- Hidden select for compatibility -->
        <select id="exercise-select" style="display:none;">
          <option value="squat" selected>Squats</option>
          <option value="pushup">Push-Ups</option>
          <option value="curl">Dumbbell Curls</option>
        </select>
      </div>

      <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
        <div>
          <div style="font-size: 0.74rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Video Source</div>
          <select id="source-select" class="btn-ghost" style="padding: 7px 14px; font-weight: 600; cursor: pointer; outline: none;">
            <option value="demo" selected>📹 Demo Video Engine</option>
            <option value="webcam">🎥 Live Webcam (cv2.VideoCapture(0))</option>
            <option value="browser_cam">💻 Browser Camera (getUserMedia)</option>
          </select>
        </div>

        <div style="align-self: flex-end;">
          <button id="btn-pause-camera" class="btn-ghost" style="padding: 8px 16px;">
            <span id="pause-icon">⏸️</span> <span id="pause-text">Pause Camera</span>
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- Live Stats Bento Panel -->
  <div style="display: flex; flex-direction: column; gap: 18px;">
    <!-- Reps Counter Bento Hero Card -->
    <div class="bento-card" style="background: linear-gradient(135deg, #18181b, #09090b); color: #ffffff; border: 1px solid #27272a; position: relative; overflow: hidden;">
      <div style="position: absolute; top: -20px; right: -20px; width: 100px; height: 100px; background: rgba(250, 204, 21, 0.15); border-radius: 50%; pointer-events: none;"></div>
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.78rem; font-weight: 800; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.05em;">Repetitions</span>
        <span id="phase-badge" class="pill-tag pill-yellow">UP</span>
      </div>
      <div class="stat-figure" id="reps-num" style="color: #facc15; font-size: 4.6rem; margin: 12px 0; font-family: 'JetBrains Mono', monospace; line-height: 1;">0</div>
      <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #a1a1aa; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
        <span>Session Time: <strong id="session-timer-text" style="color: #ffffff; font-family: 'JetBrains Mono', monospace;">00:00</strong></span>
        <span id="active-side-text">Right Leg</span>
      </div>
    </div>

    <!-- Joint Angles Bento Grid -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
      <div class="bento-card" style="padding: 16px;">
        <span class="stat-label" id="primary-angle-label">Knee Angle</span>
        <div class="stat-figure" id="primary-angle-val" style="font-size: 1.8rem; margin: 4px 0 0 0;">180°</div>
      </div>
      <div class="bento-card" style="padding: 16px;">
        <span class="stat-label" id="secondary-angle-label">Target Angle</span>
        <div class="stat-figure" id="secondary-angle-val" style="font-size: 1.8rem; margin: 4px 0 0 0; color: #854d0e;">≤ 100°</div>
      </div>
    </div>

    <!-- Form Score Card -->
    <div class="bento-card" style="padding: 18px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="stat-label">Form Quality Score</span>
        <span class="stat-figure" id="score-val" style="font-size: 1.6rem; margin: 0; color: var(--text-primary);">100%</span>
      </div>
      <div style="width: 100%; height: 8px; background: #f4f4f5; border-radius: 9999px; overflow: hidden; margin-top: 10px; border: 1px solid #e5e7eb;">
        <div id="score-bar" style="width: 100%; height: 100%; background: #facc15; border-radius: 9999px; transition: width 0.3s ease;"></div>
      </div>
    </div>

    <!-- Smart Vocal Feedback Bento Banner -->
    <div class="bento-card" id="feedback-banner" style="background: rgba(250, 204, 21, 0.12); border-color: rgba(250, 204, 21, 0.4); display: flex; align-items: center; gap: 14px; padding: 16px;">
      <div id="feedback-avatar" style="width: 40px; height: 40px; border-radius: 50%; background: #facc15; display: flex; align-items: center; justify-content: center; font-size: 1.15rem; color: #18181b; flex-shrink: 0; box-shadow: 0 2px 6px rgba(250, 204, 21, 0.3);">🗣️</div>
      <div>
        <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--text-secondary); font-weight: 700; letter-spacing: 0.04em;">Smart Form Coach</div>
        <div id="feedback-message" style="font-size: 0.92rem; font-weight: 700; color: var(--text-primary); margin-top: 2px;">Ready - Position body in frame</div>
      </div>
    </div>

    <!-- Action Buttons -->
    <div style="display: flex; gap: 10px; flex-direction: column;">
      <!-- Initial user interaction trigger to unlock browser audio autoplay restrictions -->
      <button type="button" id="btn-start-session" class="btn-yellow" style="width: 100%; padding: 12px; font-size: 0.9rem; display: flex; align-items: center; justify-content: center; gap: 8px;">
        <span>▶️</span> <span>Start Session / Enable Audio</span>
      </button>

      <form action="/save_workout" method="POST" id="workout-save-form">
        <input type="hidden" name="exercise_type" id="hidden-exercise-type" value="Squats">
        <input type="hidden" name="rep_count" id="hidden-rep-count" value="0">
        <input type="hidden" name="total_duration_seconds" id="hidden-duration" value="0">
        <input type="hidden" name="avg_form_accuracy" id="hidden-accuracy" value="100">
        <button type="submit" class="btn-charcoal" style="width: 100%; padding: 13px; font-size: 0.92rem; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
          💾 Finish & Save Workout
        </button>
      </form>

      <div style="display: flex; gap: 10px;">
        <button id="btn-voice-toggle" class="btn-ghost" style="flex: 1;">
          <span id="voice-icon">🎙️</span> <span id="voice-label">Vocal: ON (Errors Only)</span>
        </button>
        <button id="btn-reset" class="btn-ghost" style="flex: 1;">
          🔄 Reset Reps
        </button>
      </div>
    </div>
  </div>
</div>

<video id="client-webcam" playsinline autoplay muted style="display:none;"></video>
<canvas id="capture-canvas" style="display:none;"></canvas>

<script>
  let voiceEnabled = true;
  let lastSpokenTime = 0;
  let lastRepCount = null;
  let currentDuration = 0;
  let currentAccuracy = 100;
  let lastAlertTimestamp = Date.now() / 1000;
  let audioUnlocked = false;
  let currentInputSource = "demo";
  let isCapturing = false;
  let webcamStream = null;
  let currentBlobUrl = null;
  let isCameraPaused = false;
  let audioCtx = null;
  let selectedVoice = null;

  const repsEl = document.getElementById("reps-num");
  const phaseBadgeEl = document.getElementById("phase-badge");
  const sessionTimerEl = document.getElementById("session-timer-text");
  const primaryAngleLabel = document.getElementById("primary-angle-label");
  const primaryAngleVal = document.getElementById("primary-angle-val");
  const secondaryAngleLabel = document.getElementById("secondary-angle-label");
  const secondaryAngleVal = document.getElementById("secondary-angle-val");
  const scoreValEl = document.getElementById("score-val");
  const scoreBarEl = document.getElementById("score-bar");
  const feedbackBanner = document.getElementById("feedback-banner");
  const feedbackAvatar = document.getElementById("feedback-avatar");
  const feedbackMsg = document.getElementById("feedback-message");
  const activeSideText = document.getElementById("active-side-text");
  const exerciseSelect = document.getElementById("exercise-select");
  const sourceSelect = document.getElementById("source-select");
  const btnVoiceToggle = document.getElementById("btn-voice-toggle");
  const voiceLabel = document.getElementById("voice-label");
  const btnReset = document.getElementById("btn-reset");
  const videoFeedImg = document.getElementById("video-feed-img");
  const clientWebcam = document.getElementById("client-webcam");
  const captureCanvas = document.getElementById("capture-canvas");
  const captureCtx = captureCanvas.getContext("2d");
  const hiddenExerciseType = document.getElementById("hidden-exercise-type");
  const hiddenRepCount = document.getElementById("hidden-rep-count");
  const hiddenDuration = document.getElementById("hidden-duration");
  const hiddenAccuracy = document.getElementById("hidden-accuracy");
  const btnPauseCamera = document.getElementById("btn-pause-camera");
  const pauseIcon = document.getElementById("pause-icon");
  const pauseText = document.getElementById("pause-text");
  const modePills = document.querySelectorAll(".btn-mode-pill");
  const btnStartSession = document.getElementById("btn-start-session");
  const workoutSaveForm = document.getElementById("workout-save-form");

  // Modern Floating Glassmorphism Camera HUD Elements
  const hudExerciseMode = document.getElementById("hud-exercise-mode");
  const hudStagePill = document.getElementById("hud-stage-pill");
  const hudRepCount = document.getElementById("hud-rep-count");
  const hudStatusIcon = document.getElementById("hud-status-icon");
  const hudFormStatus = document.getElementById("hud-form-status");
  const hudAnglePill = document.getElementById("hud-angle-pill");
  const hudScorePill = document.getElementById("hud-score-pill");

  function formatTime(totalSec) {
    const s = Math.floor(totalSec || 0);
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return (mins < 10 ? "0" : "") + mins + ":" + (secs < 10 ? "0" : "") + secs;
  }

  // Web Audio API for Crisp Rep Chime Feedback
  function getAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    return audioCtx;
  }

  function playRepChime() {
    try {
      const ctx = getAudioContext();
      if (!ctx) return;
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      // Crisp 880Hz high-frequency synthesized sine wave chime/ding
      osc.frequency.setValueAtTime(880, now);

      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.3, now + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.28);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.28);
    } catch (e) {
      console.warn("Web Audio chime error:", e);
    }
  }

  // Query browser voices and prioritize natural/premium human-like voices
  function loadBestVoice() {
    if (!window.speechSynthesis) return;
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return;

    // Prioritize natural/premium voices: search for voices containing 'Google', 'Natural', 'Samantha', 'Karen', or 'Daniel'
    const priorityKeywords = ['Google', 'Natural', 'Samantha', 'Karen', 'Daniel'];
    for (const kw of priorityKeywords) {
      const match = voices.find(v => v.lang && v.lang.startsWith("en") && v.name.includes(kw));
      if (match) {
        selectedVoice = match;
        return;
      }
    }
    for (const kw of priorityKeywords) {
      const match = voices.find(v => v.name.includes(kw));
      if (match) {
        selectedVoice = match;
        return;
      }
    }
    const defaultEn = voices.find(v => v.lang === "en-US" || v.lang === "en_US") ||
                      voices.find(v => v.lang && v.lang.startsWith("en"));
    selectedVoice = defaultEn || voices[0];
  }

  if (window.speechSynthesis) {
    loadBestVoice();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadBestVoice;
    }
  }

  // Unlock browser audio context and Web Speech API on user interaction
  function unlockAudio() {
    getAudioContext();
    if (!audioUnlocked && window.speechSynthesis) {
      try {
        const dummy = new SpeechSynthesisUtterance("");
        dummy.volume = 0;
        window.speechSynthesis.speak(dummy);
        audioUnlocked = true;
        if (btnStartSession) {
          btnStartSession.innerHTML = "<span>🔊</span> <span>Audio Active</span>";
          btnStartSession.className = "btn-ghost";
          btnStartSession.style.color = "#18181b";
        }
      } catch (err) {
        console.warn("SpeechSynthesis unlock note:", err);
      }
    }
  }

  if (btnStartSession) {
    btnStartSession.addEventListener("click", () => {
      unlockAudio();
      speakFormError("Session started. Audio feedback enabled.");
    });
  }
  document.addEventListener("click", unlockAudio);

  // Client-side Web Speech API playback with natural tuning and 4-second delay throttle
  function speakFormError(errorMessage) {
    if (!voiceEnabled || !window.speechSynthesis || !errorMessage) return;
    const now = Date.now();
    if (now - lastSpokenTime < 4000) return;

    try {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(errorMessage);
      if (!selectedVoice) loadBestVoice();
      if (selectedVoice) {
        utter.voice = selectedVoice;
        utter.lang = selectedVoice.lang || "en-US";
      } else {
        utter.lang = "en-US";
      }
      // Tune speech parameters for human-like cadence: pitch 1.0, rate 1.05, volume 1.0
      utter.pitch = 1.0;
      utter.rate = 1.05;
      utter.volume = 1.0;
      window.speechSynthesis.speak(utter);
      lastSpokenTime = now;
    } catch (e) {
      console.warn("SpeechSynthesis error:", e);
    }
  }

  // Poll lightweight endpoint /api/audio_alert every 500ms
  async function pollAudioAlert() {
    try {
      const res = await fetch("/api/audio_alert");
      if (res.ok) {
        const data = await res.json();
        if (data && data.alert && data.timestamp && data.timestamp > lastAlertTimestamp) {
          lastAlertTimestamp = data.timestamp;
          speakFormError(data.alert);
        }
      }
    } catch (e) {}
  }
  setInterval(pollAudioAlert, 500);

  if (workoutSaveForm) {
    workoutSaveForm.addEventListener("submit", () => {
      if (hiddenRepCount) hiddenRepCount.value = lastRepCount;
      if (hiddenDuration) hiddenDuration.value = currentDuration;
      if (hiddenAccuracy) hiddenAccuracy.value = currentAccuracy;
    });
  }

  btnVoiceToggle.addEventListener("click", () => {
    voiceEnabled = !voiceEnabled;
    if (voiceEnabled) {
      btnVoiceToggle.className = "btn-ghost";
      voiceLabel.textContent = "Vocal: ON (Errors Only)";
    } else {
      btnVoiceToggle.className = "btn-ghost";
      btnVoiceToggle.style.color = "var(--accent-red)";
      voiceLabel.textContent = "Vocal: MUTED";
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
  });

  // Pause / Resume camera resource handling
  btnPauseCamera.addEventListener("click", async () => {
    isCameraPaused = !isCameraPaused;
    if (isCameraPaused) {
      pauseIcon.textContent = "▶️";
      pauseText.textContent = "Resume Camera";
      btnPauseCamera.className = "btn-ghost";
      await fetch("/pause_feed", { method: "POST" });
      videoFeedImg.src = "";
    } else {
      pauseIcon.textContent = "⏸️";
      pauseText.textContent = "Pause Camera";
      btnPauseCamera.className = "btn-ghost";
      await fetch("/resume_feed", { method: "POST" });
      videoFeedImg.src = "/video_feed?t=" + Date.now();
    }
  });

  // Ensure camera VideoCapture resource releases properly when leaving or pausing /workout
  window.addEventListener("pagehide", () => {
    navigator.sendBeacon("/pause_feed");
  });
  window.addEventListener("beforeunload", () => {
    navigator.sendBeacon("/pause_feed");
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      navigator.sendBeacon("/pause_feed");
    }
  });

  function updateTelemetry(data) {
    if (!data) return;

    if (lastRepCount === null) {
      lastRepCount = data.reps;
      repsEl.textContent = data.reps;
      if (hudRepCount) hudRepCount.textContent = data.reps;
    } else if (data.reps !== lastRepCount) {
      if (data.reps > lastRepCount) {
        playRepChime();
      }
      lastRepCount = data.reps;
      repsEl.textContent = data.reps;
      if (hudRepCount) hudRepCount.textContent = data.reps;
      repsEl.style.transform = "scale(1.15)";
      if (hudRepCount) hudRepCount.style.transform = "scale(1.2)";
      setTimeout(() => {
        repsEl.style.transform = "scale(1)";
        if (hudRepCount) hudRepCount.style.transform = "scale(1)";
      }, 180);
    } else {
      repsEl.textContent = data.reps;
      if (hudRepCount) hudRepCount.textContent = data.reps;
    }

    if (data.duration_seconds !== undefined) {
      currentDuration = data.duration_seconds;
      sessionTimerEl.textContent = formatTime(data.duration_seconds);
    }
    if (data.avg_form_accuracy !== undefined) {
      currentAccuracy = data.avg_form_accuracy;
    } else if (data.form_score !== undefined) {
      currentAccuracy = data.form_score;
    }

    const isDown = data.stage === "down";
    phaseBadgeEl.textContent = isDown ? "DOWN" : "UP";
    phaseBadgeEl.className = isDown ? "pill-tag pill-yellow" : "pill-tag pill-gray";
    if (hudStagePill) {
      hudStagePill.textContent = isDown ? "DOWN" : "UP";
      hudStagePill.style.color = isDown ? "#facc15" : "#9ca3af";
      hudStagePill.style.background = isDown ? "rgba(250, 204, 21, 0.22)" : "rgba(255, 255, 255, 0.08)";
    }

    let exerciseDisplayName = "Squats";
    if (data.exercise === "squat") {
      exerciseDisplayName = "Squats";
      primaryAngleLabel.textContent = "Knee Angle";
      primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
      secondaryAngleLabel.textContent = "Target Depth";
      secondaryAngleVal.textContent = "≤ 100°";
      activeSideText.textContent = `Active: ${data.active_side === 'right' ? 'Right Leg' : 'Left Leg'}`;
      if (hudAnglePill) hudAnglePill.textContent = Math.round(data.primary_angle) + "°";
    } else if (data.exercise === "pushup") {
      exerciseDisplayName = "Push-ups";
      primaryAngleLabel.textContent = "Elbow Angle";
      primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
      secondaryAngleLabel.textContent = "Spine Angle";
      secondaryAngleVal.textContent = Math.round(data.secondary_angle) + "°";
      activeSideText.textContent = `Flare: ${Math.round(data.flare_angle)}° (≤ 75°)`;
      if (hudAnglePill) hudAnglePill.textContent = Math.round(data.primary_angle) + "°";
    } else {
      exerciseDisplayName = "Dumbbell Curls";
      primaryAngleLabel.textContent = "Arm Angle";
      primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
      secondaryAngleLabel.textContent = "Curl Range";
      secondaryAngleVal.textContent = "< 40° / > 150°";
      activeSideText.textContent = `Active: ${data.active_side === 'right' ? 'Right Arm' : 'Left Arm'}`;
      if (hudAnglePill) hudAnglePill.textContent = Math.round(data.primary_angle) + "°";
    }
    if (hudExerciseMode) hudExerciseMode.textContent = exerciseDisplayName;

    const score = Math.max(0, Math.min(100, data.form_score));
    scoreValEl.textContent = score + "%";
    scoreBarEl.style.width = score + "%";
    if (hudScorePill) hudScorePill.textContent = score + "%";

    feedbackMsg.textContent = data.feedback;
    if (hudFormStatus) hudFormStatus.textContent = data.feedback;

    const hasWarnings = data.warnings && data.warnings.length > 0;
    if (hasWarnings) {
      feedbackBanner.style.background = "rgba(239, 68, 68, 0.1)";
      feedbackBanner.style.borderColor = "rgba(239, 68, 68, 0.3)";
      feedbackAvatar.style.background = "var(--accent-red)";
      feedbackAvatar.style.color = "#ffffff";
      if (hudStatusIcon) hudStatusIcon.textContent = "⚠️";
      if (hudFormStatus) hudFormStatus.style.color = "#f87171";
      if (hudScorePill) {
        hudScorePill.style.color = "#f87171";
        hudScorePill.style.background = "rgba(239, 68, 68, 0.18)";
      }
    } else {
      feedbackBanner.style.background = "rgba(250, 204, 21, 0.12)";
      feedbackBanner.style.borderColor = "rgba(250, 204, 21, 0.4)";
      feedbackAvatar.style.background = "#facc15";
      feedbackAvatar.style.color = "#18181b";
      if (hudStatusIcon) hudStatusIcon.textContent = "⚡";
      if (hudFormStatus) hudFormStatus.style.color = "#ffffff";
      if (hudScorePill) {
        hudScorePill.style.color = "#facc15";
        hudScorePill.style.background = "rgba(250, 204, 21, 0.18)";
      }
    }

    // Trigger speech ONLY on specific form errors with 4s debounce
    if (data.audio_event && data.audio_event.startsWith("warn_") && data.vocal_message) {
      speakFormError(data.vocal_message);
    }
  }

  async function pollStatus() {
    if (currentInputSource === "browser_cam" || isCameraPaused) return;
    try {
      const res = await fetch("/status");
      if (res.ok) {
        const data = await res.json();
        updateTelemetry(data);
      }
    } catch (e) {}
  }
  setInterval(pollStatus, 250);

  // Exercise Mode switching using Bento Pills
  modePills.forEach(pill => {
    pill.addEventListener("click", async () => {
      modePills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const selected = pill.getAttribute("data-exercise");
      exerciseSelect.value = selected;

      if (selected === 'squat') {
        hiddenExerciseType.value = 'Squats';
      } else if (selected === 'pushup') {
        hiddenExerciseType.value = 'Push-Ups';
      } else {
        hiddenExerciseType.value = 'Dumbbell Curls';
      }

      await fetch("/set_exercise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exercise: selected })
      });
      if (!isCameraPaused) {
        videoFeedImg.src = "/video_feed?t=" + Date.now();
      }
    });
  });

  sourceSelect.addEventListener("change", async () => {
    const selected = sourceSelect.value;
    currentInputSource = selected;

    if (selected !== "browser_cam" && webcamStream) {
      isCapturing = false;
      webcamStream.getTracks().forEach(t => t.stop());
      webcamStream = null;
      clientWebcam.srcObject = null;
      if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
        currentBlobUrl = null;
      }
    }

    if (selected === "browser_cam") {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Webcam access not supported in this browser.");
        sourceSelect.value = "demo";
        currentInputSource = "demo";
        return;
      }
      try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false
        });
        clientWebcam.srcObject = webcamStream;
        await clientWebcam.play();

        await fetch("/set_source", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: "browser_cam" })
        });

        isCapturing = true;
        runCameraCaptureLoop();
      } catch(err) {
        alert("Could not access browser camera: " + err.message);
        sourceSelect.value = "demo";
        currentInputSource = "demo";
      }
    } else {
      const res = await fetch("/set_source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: selected })
      });
      const data = await res.json();
      if (!isCameraPaused) {
        videoFeedImg.src = "/video_feed?t=" + Date.now();
      }
      if (selected === "webcam" && data.source === "demo") {
        alert("Physical webcam device 0 is unavailable. Falling back to demo video.");
        sourceSelect.value = "demo";
      }
    }
  });

  btnReset.addEventListener("click", async () => {
    await fetch("/reset", { method: "POST" });
    lastRepCount = 0;
    repsEl.textContent = "0";
    if (hudRepCount) hudRepCount.textContent = "0";
    sessionTimerEl.textContent = "00:00";
  });

  function runCameraCaptureLoop() {
    if (!isCapturing || currentInputSource !== "browser_cam") return;

    if (clientWebcam.readyState >= 2) {
      const vw = clientWebcam.videoWidth || 640;
      const vh = clientWebcam.videoHeight || 480;
      captureCanvas.width = 640;
      captureCanvas.height = 480;

      captureCtx.save();
      captureCtx.translate(640, 0);
      captureCtx.scale(-1, 1);
      captureCtx.drawImage(clientWebcam, 0, 0, 640, 480);
      captureCtx.restore();

      captureCanvas.toBlob(async (blob) => {
        if (!blob || !isCapturing || currentInputSource !== "browser_cam") {
          if (isCapturing) requestAnimationFrame(runCameraCaptureLoop);
          return;
        }

        try {
          const res = await fetch("/process_frame", {
            method: "POST",
            body: blob,
            headers: { "Content-Type": "image/jpeg" }
          });

          if (res.ok) {
            const metricsHeader = res.headers.get("X-Metrics");
            if (metricsHeader) {
              try {
                const metrics = JSON.parse(decodeURIComponent(metricsHeader));
                updateTelemetry(metrics);
              } catch(e) {}
            }

            const processedBlob = await res.blob();
            const newUrl = URL.createObjectURL(processedBlob);
            videoFeedImg.src = newUrl;
            if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
            currentBlobUrl = newUrl;
          }
        } catch(e) {}

        if (isCapturing && currentInputSource === "browser_cam") {
          requestAnimationFrame(runCameraCaptureLoop);
        }
      }, "image/jpeg", 0.65);

    } else {
      requestAnimationFrame(runCameraCaptureLoop);
    }
  }
</script>
"""

DASHBOARD_BODY = """
<!-- Greeting & Header Layout -->
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 24px; flex-wrap: wrap; gap: 16px;">
  <div>
    <h1 style="font-size: 2.1rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em; margin-bottom: 8px;">
      Welcome back, {{ current_user.username }} 👋
    </h1>
    <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
      <span class="pill-tag pill-charcoal">⚡ Level {{ user_level }} Athlete</span>
      <span class="pill-tag pill-yellow">Squats {{ squat_pct }}%</span>
      <span class="pill-tag pill-gray">Push-ups {{ pushup_pct }}%</span>
      <span class="pill-tag pill-gray">Curls {{ curl_pct }}%</span>
    </div>
  </div>

  <!-- Top-Right Key Metrics -->
  <div style="display: flex; gap: 24px; align-items: center; background: #ffffff; padding: 12px 24px; border-radius: var(--radius-bento); border: 1px solid var(--card-border); box-shadow: var(--card-shadow);">
    <div>
      <div class="stat-label">Level</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: var(--text-primary); font-family: 'JetBrains Mono', monospace;">{{ user_level }}</div>
    </div>
    <div style="width: 1px; height: 32px; background: #e5e7eb;"></div>
    <div>
      <div class="stat-label">Total Reps</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: #854d0e; font-family: 'JetBrains Mono', monospace;">{{ total_reps }}</div>
    </div>
    <div style="width: 1px; height: 32px; background: #e5e7eb;"></div>
    <div>
      <div class="stat-label">Experience</div>
      <div style="font-size: 1.6rem; font-weight: 800; color: var(--text-primary); font-family: 'JetBrains Mono', monospace;">{{ user_exp }} <span style="font-size: 0.9rem; color: var(--text-secondary);">XP</span></div>
    </div>
  </div>
</div>

<!-- Bento Grid Row 1: Profile & 4 Key Stat Cards -->
<div style="display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; margin-bottom: 20px;">
  <!-- User Gamer Profile Card (4 Columns) -->
  <div class="bento-card" style="grid-column: span 4; display: flex; flex-direction: column; justify-content: space-between;">
    <div>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <span class="pill-tag pill-yellow">Gamer Profile</span>
        <span style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary);">Tier {{ user_level }}</span>
      </div>
      <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 18px;">
        <div style="width: 52px; height: 52px; border-radius: 16px; background: #18181b; color: #facc15; font-size: 1.5rem; font-weight: 800; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(24, 24, 27, 0.2);">
          {{ current_user.username[0]|upper }}
        </div>
        <div>
          <div style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary);">{{ current_user.username }}</div>
          <div style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 600;">Sisyphus Form Master</div>
        </div>
      </div>

      <!-- EXP Progress Bar -->
      <div style="margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 700; margin-bottom: 6px;">
          <span style="color: var(--text-secondary);">Level Progress</span>
          <span style="color: #854d0e; font-family: 'JetBrains Mono', monospace;">{{ exp_in_level }} / 100 XP</span>
        </div>
        <div style="width: 100%; height: 10px; background: #f4f4f5; border-radius: 9999px; overflow: hidden; border: 1px solid #e5e7eb;">
          <div style="width: {{ exp_progress_pct }}%; height: 100%; background: linear-gradient(90deg, #facc15, #18181b); border-radius: 9999px; transition: width 0.4s ease;"></div>
        </div>
        <div style="font-size: 0.74rem; color: var(--text-muted); margin-top: 5px;">{{ 100 - exp_in_level }} XP needed for Level {{ user_level + 1 }}</div>
      </div>
    </div>

    <div>
      <a href="/workout" class="btn-yellow" style="width: 100%; box-sizing: border-box;">
        🏋️ Start Workout Now
      </a>
    </div>
  </div>

  <!-- 4 Stat Metric Cards (8 Columns) -->
  <div style="grid-column: span 8; display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
    <div class="bento-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="stat-label">TOTAL REPS COMPLETED</div>
        <div style="width: 32px; height: 32px; border-radius: 50%; background: var(--yellow-subtle); display: flex; align-items: center; justify-content: center; font-size: 0.95rem;">⚡</div>
      </div>
      <div class="stat-figure">{{ total_reps }}</div>
      <div style="font-size: 0.8rem; color: var(--text-secondary);">Across all exercises</div>
    </div>

    <div class="bento-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="stat-label">TOTAL WORKOUTS</div>
        <div style="width: 32px; height: 32px; border-radius: 50%; background: #f4f4f5; display: flex; align-items: center; justify-content: center; font-size: 0.95rem;">📅</div>
      </div>
      <div class="stat-figure">{{ total_workouts }}</div>
      <div style="font-size: 0.8rem; color: var(--text-secondary);">Completed sessions</div>
    </div>

    <div class="bento-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="stat-label">ACTIVE TIME</div>
        <div style="width: 32px; height: 32px; border-radius: 50%; background: #eff6ff; display: flex; align-items: center; justify-content: center; font-size: 0.95rem;">⏱️</div>
      </div>
      <div class="stat-figure" style="font-size: 2.1rem;">{{ total_duration | format_duration }}</div>
      <div style="font-size: 0.8rem; color: var(--text-secondary);">Under exercise tracking</div>
    </div>

    <div class="bento-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="stat-label">AVERAGE FORM</div>
        <div style="width: 32px; height: 32px; border-radius: 50%; background: #fefce8; display: flex; align-items: center; justify-content: center; font-size: 0.95rem;">🎯</div>
      </div>
      <div class="stat-figure" style="color: #854d0e;">{{ avg_form }}%</div>
      <div style="font-size: 0.8rem; color: var(--text-secondary);">Movement accuracy score</div>
    </div>
  </div>
</div>

<!-- Bento Grid Row 2: Analytics Bar Chart & Dark Quests Widget -->
<div style="display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; margin-bottom: 20px;">
  <!-- Progress Analytics Card (7 Columns) -->
  <div class="bento-card" style="grid-column: span 7;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
      <div>
        <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">Daily Rep Volume Over Time</h3>
        <p style="font-size: 0.8rem; color: var(--text-secondary);">Minimalist volume analytics across sessions</p>
      </div>
      <span class="pill-tag pill-charcoal">Activity</span>
    </div>
    <div style="height: 290px; width: 100%;">
      <canvas id="progressChart"></canvas>
    </div>
  </div>

  <!-- Quests & Badges Widget: High Contrast Dark Charcoal Card (5 Columns) -->
  <div class="bento-dark-card" style="grid-column: span 5; display: flex; flex-direction: column; justify-content: space-between;">
    <div>
      <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
        <div>
          <h3 style="font-size: 1.15rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">🏆 Badges & Achievements</h3>
          <p style="font-size: 0.78rem; color: #a1a1aa; margin-top: 2px;">Active milestones & training quests</p>
        </div>
        <span class="pill-tag pill-yellow" style="font-size: 0.7rem;">
          {{ unlocked_count }}/{{ total_badge_count }} Completed
        </span>
      </div>

      <div style="display: flex; flex-direction: column; gap: 10px; max-height: 270px; overflow-y: auto; padding-right: 4px;">
        {% for b in badges %}
          <div style="background: {% if b.unlocked %}rgba(250, 204, 21, 0.08){% else %}rgba(39, 39, 42, 0.5){% endif %}; border: 1px solid {% if b.unlocked %}rgba(250, 204, 21, 0.3){% else %}#27272a{% endif %}; border-radius: 14px; padding: 12px 14px; display: flex; align-items: center; justify-content: space-between; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
              <div style="font-size: 1.35rem; width: 36px; height: 36px; border-radius: 10px; background: {% if b.unlocked %}rgba(250, 204, 21, 0.2){% else %}#27272a{% endif %}; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                {{ b.icon }}
              </div>
              <div>
                <div style="font-size: 0.88rem; font-weight: 700; color: {% if b.unlocked %}#ffffff{% else %}#a1a1aa{% endif %};">{{ b.name }}</div>
                <div style="font-size: 0.74rem; color: #71717a;">{{ b.description }}</div>
              </div>
            </div>
            <div>
              {% if b.unlocked %}
                <span class="pill-tag" style="background: #facc15; color: #18181b; padding: 3px 8px; font-size: 0.68rem;">✓ UNLOCKED</span>
              {% else %}
                <span class="pill-tag" style="background: #27272a; color: #71717a; padding: 3px 8px; font-size: 0.68rem;">🔒 {{ b.requirement }}</span>
              {% endif %}
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  </div>
</div>

<!-- Bento Grid Row 3: Recent Sessions Table -->
<div class="bento-card">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
    <div>
      <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">Recent Exercise Sessions</h3>
      <p style="font-size: 0.8rem; color: var(--text-secondary);">Historical log of performed workouts</p>
    </div>
    <a href="/workout" class="btn-yellow" style="padding: 8px 18px; font-size: 0.84rem;">Start New Workout</a>
  </div>

  {% if sessions %}
    <div style="overflow-x: auto;">
      <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem;">
        <thead>
          <tr style="border-bottom: 2px solid #f4f4f5; color: var(--text-secondary); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em;">
            <th style="padding: 12px 14px;">Date & Time</th>
            <th style="padding: 12px 14px;">Exercise</th>
            <th style="padding: 12px 14px;">Reps</th>
            <th style="padding: 12px 14px;">Duration</th>
            <th style="padding: 12px 14px;">Form Score</th>
          </tr>
        </thead>
        <tbody>
          {% for s in sessions %}
            <tr style="border-bottom: 1px solid #f4f4f5;">
              <td style="padding: 14px; font-family: 'JetBrains Mono', monospace; color: var(--text-secondary); font-size: 0.82rem;">
                {{ s.timestamp.strftime('%b %d, %Y - %H:%M') if s.timestamp else 'N/A' }}
              </td>
              <td style="padding: 14px;">
                {% if 'squat' in s.exercise_type.lower() %}
                  <span class="pill-tag pill-yellow">{{ s.exercise_type }}</span>
                {% elif 'push' in s.exercise_type.lower() %}
                  <span class="pill-tag" style="background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe;">{{ s.exercise_type }}</span>
                {% else %}
                  <span class="pill-tag pill-charcoal">{{ s.exercise_type }}</span>
                {% endif %}
              </td>
              <td style="padding: 14px; font-family: 'JetBrains Mono', monospace; font-weight: 800; color: var(--text-primary); font-size: 1rem;">
                {{ s.rep_count }}
              </td>
              <td style="padding: 14px; font-family: 'JetBrains Mono', monospace; color: var(--text-secondary); font-size: 0.84rem;">
                {{ s.total_duration_seconds | format_duration }}
              </td>
              <td style="padding: 14px;">
                <span class="pill-tag" style="{% if s.avg_form_accuracy >= 80 %}background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;{% elif s.avg_form_accuracy >= 60 %}background: #fffbeb; color: #92400e; border: 1px solid #fde68a;{% else %}background: #fef2f2; color: #991b1b; border: 1px solid #fecaca;{% endif %}">
                  {{ s.avg_form_accuracy }}%
                </span>
              </td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  {% else %}
    <div style="text-align: center; padding: 48px; color: var(--text-secondary);">
      <div style="font-size: 2.2rem; margin-bottom: 8px;">🏋️</div>
      <p style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">No workout sessions recorded yet.</p>
      <p style="font-size: 0.84rem; margin-bottom: 18px;">Perform your first workout to start tracking your daily progress and earning badges.</p>
      <a href="/workout" class="btn-yellow">Start Your First Workout</a>
    </div>
  {% endif %}
</div>

<script>
  const labels = {{ chart_labels | tojson }};
  const data = {{ chart_data | tojson }};
  const ctx = document.getElementById('progressChart').getContext('2d');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.length > 0 ? labels : ['No Data'],
      datasets: [{
        label: 'Repetitions',
        data: data.length > 0 ? data : [0],
        backgroundColor: '#facc15',
        hoverBackgroundColor: '#eab308',
        borderRadius: 6,
        barThickness: 16,
        maxBarThickness: 24
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#18181b',
          titleColor: '#ffffff',
          bodyColor: '#facc15',
          cornerRadius: 8,
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#71717a', font: { family: 'Plus Jakarta Sans', weight: '600' } }
        },
        y: {
          beginAtZero: true,
          grid: { color: '#f3f4f6' },
          ticks: { color: '#71717a', precision: 0, font: { family: 'Plus Jakarta Sans', weight: '600' } }
        }
      }
    }
  });
</script>
"""

AUTH_BODY = """
<div class="bento-card" style="max-width: 440px; margin: 40px auto; padding: 36px 32px; border-radius: 24px;">
  <div style="text-align: center; margin-bottom: 24px;">
    <div style="width: 52px; height: 52px; border-radius: 16px; background: #facc15; color: #18181b; font-size: 1.6rem; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto; box-shadow: 0 4px 12px rgba(250, 204, 21, 0.35);">
      ⚡
    </div>
    <h2 style="font-size: 1.5rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">{{ title }}</h2>
    <p style="font-size: 0.84rem; color: var(--text-secondary); margin-top: 4px;">{{ subtitle }}</p>
  </div>

  <form method="POST" action="{{ action }}">
    <label style="font-size: 0.78rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.04em; display: block; margin-bottom: 6px;">Username</label>
    <input type="text" name="username" placeholder="e.g. alex_fitness" required autocomplete="username">

    <label style="font-size: 0.78rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.04em; display: block; margin-bottom: 6px;">Password</label>
    <input type="password" name="password" placeholder="••••••••" required autocomplete="current-password">

    <button type="submit" class="btn-yellow" style="width: 100%; padding: 13px; font-size: 0.92rem; margin-top: 6px;">
      {{ button_label }}
    </button>
  </form>

  <div style="text-align: center; font-size: 0.85rem; color: var(--text-secondary); margin-top: 20px;">
    {% if is_register %}
      Already have an account? <a href="/login" style="color: var(--text-primary); font-weight: 800; text-decoration: underline;">Log In</a>
    {% else %}
      Don't have an account? <a href="/register" style="color: var(--text-primary); font-weight: 800; text-decoration: underline;">Register</a>
    {% endif %}
  </div>
</div>
"""

SUMMARY_BODY = """
<style>
  @keyframes checkPop {
    0% { transform: scale(0.6); opacity: 0; }
    60% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(1); }
  }
  .summary-hero-badge {
    width: 72px;
    height: 72px;
    border-radius: 22px;
    background: #facc15;
    color: #18181b;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 16px;
    box-shadow: 0 10px 25px -5px rgba(250, 204, 21, 0.45);
    animation: checkPop 0.5s ease-out;
  }
</style>

<!-- Hero Header -->
<div style="text-align: center; margin-bottom: 32px; margin-top: 10px;">
  <div class="summary-hero-badge">✓</div>
  <h1 style="font-size: 2.3rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em;">
    Workout Completed!
  </h1>
  <p style="font-size: 0.95rem; color: var(--text-secondary); margin-top: 6px;">
    Outstanding performance! Your repetitions and movement mechanics have been evaluated and rewarded.
  </p>
</div>

<!-- Stats Row: 4 Bento Cards -->
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;">
  <div class="bento-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span class="stat-label">Exercise Completed</span>
      <span style="font-size: 1.25rem;">
        {% if 'squat' in session.exercise_type|lower %}🏋️{% elif 'push' in session.exercise_type|lower %}💪{% else %}⚡{% endif %}
      </span>
    </div>
    <div class="stat-figure" style="font-size: 1.8rem; margin-top: 8px;">{{ session.exercise_type }}</div>
    <span class="pill-tag pill-charcoal" style="margin-top: 8px;">Logged Session</span>
  </div>

  <div class="bento-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span class="stat-label">Total Reps</span>
      <span style="font-size: 1.25rem;">🔥</span>
    </div>
    <div class="stat-figure" style="font-size: 2.3rem; margin-top: 8px; color: #854d0e; font-family: 'JetBrains Mono', monospace;">
      {{ session.rep_count }}
    </div>
    <span class="pill-tag pill-yellow" style="margin-top: 8px;">Completed Reps</span>
  </div>

  <div class="bento-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span class="stat-label">EXP Earned</span>
      <span style="font-size: 1.25rem;">⚡</span>
    </div>
    <div class="stat-figure" style="font-size: 2.3rem; margin-top: 8px; color: var(--text-primary); font-family: 'JetBrains Mono', monospace;">
      +{{ exp_earned }} <span style="font-size: 1rem; color: var(--text-secondary);">XP</span>
    </div>
    <span class="pill-tag pill-yellow" style="margin-top: 8px;">
      {% if 'push' in session.exercise_type|lower %}+12 EXP / rep{% elif 'curl' in session.exercise_type|lower %}+8 EXP / rep{% else %}+10 EXP / rep{% endif %}
    </span>
  </div>

  <div class="bento-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span class="stat-label">Average Form</span>
      <span style="font-size: 1.25rem;">🎯</span>
    </div>
    <div class="stat-figure" style="font-size: 2.3rem; margin-top: 8px; color: #854d0e; font-family: 'JetBrains Mono', monospace;">
      {{ session.avg_form_accuracy }}%
    </div>
    <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 8px;">
      Time: <strong>{{ session.total_duration_seconds | format_duration }}</strong>
    </div>
  </div>
</div>

<!-- Bento Row 2: Level Progress Widget & New Badges Unlocked Section -->
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px;">
  <!-- Level Progress Widget -->
  <div class="bento-card" style="display: flex; flex-direction: column; justify-content: space-between;">
    <div>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <span class="stat-label">Athlete Progression</span>
        <span class="pill-tag pill-charcoal">Level {{ user_level }}</span>
      </div>

      <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 20px;">
        <div style="width: 52px; height: 52px; border-radius: 16px; background: #facc15; color: #18181b; font-size: 1.5rem; font-weight: 800; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(250, 204, 21, 0.35);">
          ⚡
        </div>
        <div>
          <div style="font-size: 1.2rem; font-weight: 800; color: var(--text-primary);">Level {{ user_level }} Sisyphus Athlete</div>
          <div style="font-size: 0.82rem; color: var(--text-secondary); font-weight: 600;">{{ user_exp }} Total Lifetime EXP</div>
        </div>
      </div>

      <div style="margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 700; margin-bottom: 8px;">
          <span style="color: var(--text-secondary);">Level {{ user_level }} Progress</span>
          <span style="color: #854d0e; font-family: 'JetBrains Mono', monospace;">{{ exp_in_level }} / 100 XP</span>
        </div>
        <div style="width: 100%; height: 12px; background: #f4f4f5; border-radius: 9999px; overflow: hidden; border: 1px solid #e5e7eb;">
          <div style="width: {{ exp_progress_pct }}%; height: 100%; background: linear-gradient(90deg, #facc15, #18181b); border-radius: 9999px; transition: width 0.5s ease;"></div>
        </div>
        <div style="font-size: 0.76rem; color: var(--text-muted); margin-top: 6px;">
          {{ exp_needed }} XP needed to reach Level {{ user_level + 1 }}
        </div>
      </div>
    </div>

    <div style="background: #fbfbfb; border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 16px; font-size: 0.82rem; color: var(--text-secondary);">
      💡 Earn bonus XP by keeping your posture steady and hitting full joint depth on every rep!
    </div>
  </div>

  <!-- New Badges Unlocked Section: Dark Bento Card Widget -->
  <div class="bento-dark-card" style="display: flex; flex-direction: column; justify-content: space-between;">
    <div>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div>
          <h3 style="font-size: 1.15rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">🏆 Badges Unlocked</h3>
          <p style="font-size: 0.78rem; color: #a1a1aa; margin-top: 2px;">Achievements earned during this session</p>
        </div>
        <span class="pill-tag pill-yellow">{{ new_badges|length }} New</span>
      </div>

      {% if new_badges %}
        <div style="display: flex; flex-direction: column; gap: 10px;">
          {% for b in new_badges %}
            <div style="background: rgba(250, 204, 21, 0.12); border: 1px solid rgba(250, 204, 21, 0.4); border-radius: 14px; padding: 12px 14px; display: flex; align-items: center; justify-content: space-between; gap: 12px;">
              <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 1.4rem; width: 40px; height: 40px; border-radius: 12px; background: rgba(250, 204, 21, 0.25); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                  {{ b.icon }}
                </div>
                <div>
                  <div style="font-size: 0.92rem; font-weight: 700; color: #ffffff;">{{ b.name }}</div>
                  <div style="font-size: 0.76rem; color: #d4d4d8;">{{ b.description }}</div>
                </div>
              </div>
              <span class="pill-tag" style="background: #facc15; color: #18181b; padding: 4px 10px; font-size: 0.72rem;">🎉 UNLOCKED!</span>
            </div>
          {% endfor %}
        </div>
      {% else %}
        <div style="background: rgba(39, 39, 42, 0.5); border: 1px dashed #3f3f46; border-radius: 14px; padding: 28px; text-align: center;">
          <div style="font-size: 2rem; margin-bottom: 8px;">🛡️</div>
          <div style="font-size: 0.94rem; font-weight: 700; color: #ffffff;">No new badges unlocked this session</div>
          <div style="font-size: 0.78rem; color: #a1a1aa; margin-top: 4px; max-width: 320px; margin-left: auto; margin-right: auto;">
            Keep pushing! You're making progress toward Century Club, Iron Legs, and Level 5 Champion.
          </div>
        </div>
      {% endif %}
    </div>

    <div style="margin-top: 16px;">
      <span style="font-size: 0.76rem; color: #a1a1aa;">View all quests & badge requirements on your dashboard</span>
    </div>
  </div>
</div>

<!-- Action Buttons -->
<div style="display: flex; gap: 14px; justify-content: center; align-items: center; margin-top: 8px;">
  <a href="/" class="btn-yellow" style="padding: 13px 30px; font-size: 0.92rem;">
    📊 Back to Dashboard
  </a>
  <a href="/workout" class="btn-charcoal" style="padding: 13px 30px; font-size: 0.92rem;">
    🏋️ Start Another Workout
  </a>
</div>
"""


# ---------------------------------------------------------
# FLASK ROUTE HANDLERS
# ---------------------------------------------------------
@app.route('/')
def index():
    """
    Primary Landing Route:
    - Serves the Dashboard directly for logged-in users instead of redirecting to workout feed.
    - For unauthenticated users visiting '/', redirects directly to '/login'.
    """
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return dashboard()


@app.route('/dashboard')
@login_required
def dashboard():
    """User analytics dashboard with Bento Box aesthetic, EXP/Leveling & SQLite Badge achievements."""
    check_and_award_badges(current_user)

    user_sessions = (
        ExerciseSession.query.filter_by(user_id=current_user.id)
        .order_by(ExerciseSession.timestamp.desc())
        .all()
    )

    total_reps = sum(s.rep_count for s in user_sessions)
    total_workouts = len(user_sessions)
    total_duration = sum(s.total_duration_seconds for s in user_sessions)
    avg_form = (
        round(sum(s.avg_form_accuracy for s in user_sessions) / total_workouts, 1)
        if total_workouts > 0
        else 100.0
    )

    # Adaptive New vs. Returning User Welcome Messaging
    is_new_user = (total_workouts == 0)
    if is_new_user:
        welcome_title = f"Welcome to Sisyphus, {current_user.username}! Ready to start your fitness journey?"
        workout_streak = 0
        last_exercise_stats = "No previous workouts"
    else:
        welcome_title = f"Welcome back, {current_user.username}! Ready to crush today's goals?"
        # Calculate workout streak from consecutive session days
        dates = sorted({s.timestamp.date() for s in user_sessions if s.timestamp}, reverse=True)
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        streak = 0
        if dates and (dates[0] == today or dates[0] == yesterday):
            curr = dates[0]
            for d in dates:
                if d == curr:
                    streak += 1
                    curr = curr - timedelta(days=1)
                else:
                    break
        else:
            streak = 1 if len(dates) > 0 else 0
        workout_streak = max(1, streak)
        latest = user_sessions[0]
        last_exercise_stats = f"{latest.rep_count} {latest.exercise_type} ({int(round(latest.avg_form_accuracy))}% Form)"

    daily_volume = defaultdict(int)
    for s in sorted(user_sessions, key=lambda x: x.timestamp):
        if s.timestamp:
            date_key = s.timestamp.strftime('%b %d')
            daily_volume[date_key] += s.rep_count

    chart_labels = list(daily_volume.keys())
    chart_data = list(daily_volume.values())

    user_exp = current_user.exp or 0
    user_level = current_user.level or 1
    exp_in_level = user_exp % 100
    exp_progress_pct = exp_in_level

    # Exercise distribution breakdown & Skill Mastery Reps
    squat_reps = sum(s.rep_count for s in user_sessions if "squat" in s.exercise_type.lower())
    pushup_reps = sum(s.rep_count for s in user_sessions if "push" in s.exercise_type.lower())
    curl_reps = sum(s.rep_count for s in user_sessions if "curl" in s.exercise_type.lower())
    total_tracked = max(1, squat_reps + pushup_reps + curl_reps)
    squat_pct = int(round((squat_reps / total_tracked) * 100)) if total_reps > 0 else 0
    pushup_pct = int(round((pushup_reps / total_tracked) * 100)) if total_reps > 0 else 0
    curl_pct = int(round((curl_reps / total_tracked) * 100)) if total_reps > 0 else 0

    unlocked_names = {
        b.name: (b.unlocked_at.strftime('%b %d, %Y') if b.unlocked_at else 'Unlocked')
        for b in current_user.badges
    }
    badge_cards = []
    for b_info in AVAILABLE_BADGES:
        unlocked = b_info["name"] in unlocked_names
        badge_cards.append({
            "name": b_info["name"],
            "icon": b_info["icon"],
            "description": b_info["description"],
            "requirement": b_info["requirement"],
            "unlocked": unlocked,
            "unlocked_date": unlocked_names.get(b_info["name"], "")
        })

    unlocked_count = len(unlocked_names)
    total_badge_count = len(AVAILABLE_BADGES)

    return render_template(
        'index.html',
        sessions=user_sessions,
        total_reps=total_reps,
        total_workouts=total_workouts,
        total_duration=total_duration,
        avg_form=avg_form,
        chart_labels=chart_labels,
        chart_data=chart_data,
        user_level=user_level,
        user_exp=user_exp,
        exp_in_level=exp_in_level,
        exp_progress_pct=exp_progress_pct,
        badges=badge_cards,
        squat_reps=squat_reps,
        pushup_reps=pushup_reps,
        curl_reps=curl_reps,
        squat_pct=squat_pct,
        pushup_pct=pushup_pct,
        curl_pct=curl_pct,
        unlocked_count=unlocked_count,
        total_badge_count=total_badge_count,
        is_new_user=is_new_user,
        welcome_title=welcome_title,
        workout_streak=workout_streak,
        last_exercise_stats=last_exercise_stats
    )


@app.route('/workout')
def workout():
    """Live workout tracking view with Bento Card encapsulation."""
    return render_template('workout.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template('register.html')

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("Username already exists. Please pick another.", "danger")
            return render_template('register.html')

        new_user = User(username=username, exp=0, level=1)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        flash(f"Welcome, {new_user.username}!", "success")
        return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")
            next_url = request.args.get('next')
            return redirect(next_url or url_for('dashboard'))
        else:
            flash("Invalid username or password.", "danger")

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """User logout route."""
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))


@app.route('/save_workout', methods=['POST'])
@login_required
def save_workout():
    """Save active workout session into SQLite for the logged-in user with EXP & badge evaluation."""
    data = request.get_json(silent=True) or request.form
    exercise_type = data.get('exercise_type')
    if not exercise_type:
        if tracker.exercise == 'squat':
            exercise_type = 'Squats'
        elif tracker.exercise == 'pushup':
            exercise_type = 'Push-Ups'
        else:
            exercise_type = 'Dumbbell Curls'

    with tracker.lock:
        reps = tracker.counter
        duration = max(1.0, tracker.get_duration())

        # Form Accuracy Percentage computation based on valid rep frame ratios vs form error warnings
        valid_frames = data.get("valid_frames")
        error_warnings = data.get("error_warnings") or data.get("warning_count")
        if valid_frames is not None:
            total_f = max(1, int(valid_frames) + int(error_warnings or 0))
            avg_acc = round((int(valid_frames) / total_f) * 100.0, 1)
        elif tracker.total_pose_frames > 0:
            valid_f = max(0, tracker.total_pose_frames - tracker.warning_pose_frames)
            avg_acc = round((valid_f / max(1, tracker.total_pose_frames)) * 100.0, 1)
        elif "avg_form_accuracy" in data and data["avg_form_accuracy"] is not None:
            avg_acc = float(data["avg_form_accuracy"])
        else:
            avg_acc = tracker.get_avg_accuracy()

        if "rep_count" in data and data["rep_count"] is not None:
            reps = int(data["rep_count"])
        if "total_duration_seconds" in data and data["total_duration_seconds"] is not None:
            duration = float(data["total_duration_seconds"])

    # Reset active session counter
    tracker.reset_reps()

    new_session = ExerciseSession(
        user_id=current_user.id,
        exercise_type=exercise_type,
        rep_count=reps,
        total_duration_seconds=round(duration, 1),
        avg_form_accuracy=round(avg_acc, 1),
        timestamp=datetime.now(timezone.utc)
    )
    db.session.add(new_session)

    # Gamification: EXP Calculation & Leveling
    base_exp = max(10, reps * 10)
    form_bonus = 50 if avg_acc >= 90 else (25 if avg_acc >= 75 else 0)
    exp_gained = base_exp + form_bonus

    current_user.exp = (current_user.exp or 0) + exp_gained
    new_level = 1 + (current_user.exp // 100)
    leveled_up = new_level > current_user.level
    current_user.level = new_level

    db.session.commit()

    # Persistent SQLite Badges evaluation
    new_badges = check_and_award_badges(current_user)

    if leveled_up:
        flash(f"🎉 Level Up! You reached Level {current_user.level}! (+{exp_gained} EXP)", "success")
    for nb in new_badges:
        flash(f"🏆 Achievement Unlocked: {nb.icon} {nb.name} - {nb.description}", "success")

    summary_url = url_for('summary', session_id=new_session.id)

    if request.is_json:
        return jsonify({
            "success": True,
            "message": "Workout saved successfully!",
            "session": new_session.to_dict(),
            "session_id": new_session.id,
            "summary_url": summary_url,
            "exp_gained": exp_gained,
            "level": current_user.level,
            "new_badges": [b.to_dict() for b in new_badges]
        })

    flash(f"Workout saved! {reps} {exercise_type} logged (+{exp_gained} EXP).", "success")
    return redirect(summary_url)


@app.route('/summary/<int:session_id>')
@login_required
def summary(session_id):
    """Post-workout summary & form analytics screen."""
    session_record = ExerciseSession.query.filter_by(id=session_id, user_id=current_user.id).first_or_404()
    score = session_record.avg_form_accuracy

    # Status badge evaluation: Optimal (>=85%), Moderate (70-84%), Needs Adjustment (<70%)
    if score >= 85:
        form_status = "Optimal"
        badge_class = "badge-optimal"
        badge_color = "#10b981"
    elif score >= 70:
        form_status = "Moderate"
        badge_class = "badge-moderate"
        badge_color = "#f59e0b"
    else:
        form_status = "Needs Adjustment"
        badge_class = "badge-adjustment"
        badge_color = "#ef4444"

    # Contextual Form Review Feedback Insights
    insights = []
    ex_lower = session_record.exercise_type.lower()
    score_int = int(round(score))
    if "squat" in ex_lower:
        insights.append({
            "title": f"Squat depth consistent on {score_int}% of reps",
            "detail": "Knee flexion and hip crease reached parallel with the knee joint to engage quads and glutes.",
            "tag": "Depth Consistency"
        })
        insights.append({
            "title": "Knee alignment maintained through eccentric descent" if score >= 75 else "Keep knees driven outward over toes",
            "detail": "No significant knee valgus collapse detected during active rep cycles." if score >= 75 else "Drive knees outward to avoid excessive medial stress on ligaments.",
            "tag": "Joint Alignment"
        })
        insights.append({
            "title": "Controlled descent tempo maintained",
            "detail": "Smooth eccentric descent prevented bounce at bottom of rep range.",
            "tag": "Cadence & Tempo"
        })
    elif "push" in ex_lower:
        insights.append({
            "title": "Keep hips aligned on push-ups",
            "detail": "Core engaged to maintain a straight line from shoulders through hips and heels.",
            "tag": "Core Alignment"
        })
        insights.append({
            "title": f"Elbow flare angle kept tucked within safe zone on {score_int}% of reps",
            "detail": "Elbows held at roughly 45° to 60° to protect anterior shoulder capsules.",
            "tag": "Elbow Safety"
        })
        insights.append({
            "title": "Full chest descent and arm extension lockout",
            "detail": "Achieved full range of motion on each completed repetition.",
            "tag": "Range of Motion"
        })
    else:
        insights.append({
            "title": f"Bicep peak contraction sustained on {score_int}% of reps",
            "detail": "Arm flexion reached top apex for maximum bicep muscle tension.",
            "tag": "Peak Flexion"
        })
        insights.append({
            "title": "Minimal torso swing and body momentum detected",
            "detail": "Upper body kept stationary to isolate bicep heads without hip thrust.",
            "tag": "Strict Form"
        })
        insights.append({
            "title": "Smooth eccentric lowering phase",
            "detail": "Full elbow extension reached at bottom before commencing next rep.",
            "tag": "Controlled Tempo"
        })

    exp_earned = max(10, session_record.rep_count * 10) + (50 if score >= 90 else (25 if score >= 75 else 0))
    user_exp = current_user.exp or 0
    user_level = current_user.level or 1
    exp_in_level = user_exp % 100
    exp_progress_pct = exp_in_level
    exp_needed = 100 - exp_in_level

    # Determine badges unlocked recently
    now_utc = datetime.now(timezone.utc)
    new_badges = []
    for b in current_user.badges:
        if b.unlocked_at:
            b_dt = b.unlocked_at
            if b_dt.tzinfo is None:
                b_dt = b_dt.replace(tzinfo=timezone.utc)
            if abs((now_utc - b_dt).total_seconds()) < 120:
                new_badges.append(b)

    return render_template(
        'summary.html',
        session=session_record,
        form_status=form_status,
        badge_class=badge_class,
        badge_color=badge_color,
        insights=insights,
        exp_earned=exp_earned,
        user_level=user_level,
        user_exp=user_exp,
        exp_in_level=exp_in_level,
        exp_progress_pct=exp_progress_pct,
        exp_needed=exp_needed,
        new_badges=new_badges
    )


@app.route('/video_feed')
def video_feed():
    """Video feed endpoint streaming multipart JPEG."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/pause_feed', methods=['POST', 'GET'])
def pause_feed():
    """Explicitly release camera VideoCapture when leaving or pausing workout."""
    stream_mgr.pause()
    return jsonify({"success": True, "paused": True})


@app.route('/resume_feed', methods=['POST', 'GET'])
def resume_feed():
    """Resume video feed and re-acquire camera."""
    stream_mgr.resume()
    return jsonify({"success": True, "paused": False})


@app.route('/api/audio_alert')
def api_audio_alert():
    """Lightweight endpoint for throttled form warning vocal cues."""
    return jsonify({
        "alert": tracker.latest_audio_alert,
        "timestamp": tracker.latest_alert_time
    })


@app.route('/status')
def status():
    """Real-time status endpoint."""
    data = tracker.get_status()
    data['source'] = stream_mgr.source_mode
    data['is_paused'] = stream_mgr.is_paused
    return jsonify(data)


@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    """Switch active exercise between 'squat', 'pushup', and 'curl'."""
    data = request.get_json(silent=True) or request.form
    ex = data.get('exercise', 'squat')
    tracker.set_exercise(ex)
    stream_mgr.update_demo_for_exercise()
    return jsonify({"success": True, "exercise": tracker.exercise})


@app.route('/reset', methods=['POST', 'GET'])
def reset():
    """Reset repetitions counter and timers."""
    tracker.reset_reps()
    return jsonify({"success": True, "reps": 0})


@app.route('/set_source', methods=['POST'])
def set_source():
    """Switch video source between 'demo', 'webcam', and 'browser_cam'."""
    data = request.get_json(silent=True) or request.form
    src = data.get('source', 'demo')
    if src in ['demo', 'webcam', 'browser_cam']:
        stream_mgr.set_source(src)
        return jsonify({"success": True, "source": stream_mgr.source_mode})
    return jsonify({"success": False, "error": "Invalid source"}), 400


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frames from in-browser camera."""
    try:
        if stream_mgr.source_mode != "browser_cam":
            stream_mgr.source_mode = "browser_cam"

        raw_data = request.get_data()
        if not raw_data:
            return jsonify({"error": "No frame received"}), 400

        file_bytes = np.frombuffer(raw_data, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        # Performance Optimization: Resize incoming frame to 480x360 before MediaPipe processing to reduce CPU/GPU load
        h, w = frame.shape[:2]
        if w != 480 or h != 360:
            frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)

        # Performance Optimization: Process pose tracking every 2nd frame (frame skipping)
        stream_mgr.frame_count += 1
        skip_pose = (stream_mgr.frame_count % 2 != 0)
        processed = tracker.process_frame(frame, skip_pose=skip_pose)

        # Bandwidth Optimization: Set JPEG encoding compression to quality 65
        ret, buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ret:
            return jsonify({"error": "Failed to encode frame"}), 500

        metrics = tracker.get_status()
        metrics_json = json.dumps(metrics)

        resp = Response(buf.tobytes(), mimetype='image/jpeg')
        resp.headers['X-Metrics'] = urllib.parse.quote(metrics_json)
        resp.headers['Access-Control-Expose-Headers'] = 'X-Metrics'
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# DATABASE CREATION & APP STARTUP
# ---------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"[*] Starting Sisyphus AI Exercise Tracker on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

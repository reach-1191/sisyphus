# pip install flask flask-sqlalchemy flask-login werkzeug opencv-python-headless mediapipe numpy
import os
import sys
import time
import math
import json
import urllib.parse
import threading
from datetime import datetime, timezone
from functools import wraps
from collections import defaultdict

import cv2
import numpy as np
from flask import (
    Flask,
    render_template_string,
    Response,
    jsonify,
    request,
    redirect,
    url_for,
    flash,
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
# DATABASE MODELS
# ---------------------------------------------------------
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    sessions = db.relationship('ExerciseSession', backref='user', lazy=True, cascade='all, delete-orphan')

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
        self.lock = threading.Lock()
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

        self.feedback = "Ready - Position body in frame"
        self.warning_flags = []

        # Smart Audio Feedback: ONLY set on form errors
        # Silent on movement updates, starting positions, and successful reps
        self.audio_event = None
        self.vocal_message = ""
        self.last_audio_trigger_time = 0.0

        self.min_angle_in_rep = 180.0
        self.max_angle_in_rep = 180.0
        self.active_side = "right"
        self.last_rep_time = 0

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

    def reset_reps(self):
        with self.lock:
            self.counter = 0
            self.stage = "down" if self.exercise == "curl" else "up"
            self.form_score = 100
            self.form_scores_history.clear()
            self.session_start_time = time.time()
            self.rep_score_deductions.clear()
            self.min_angle_in_rep = 180.0
            self.max_angle_in_rep = 180.0
            self.warning_flags.clear()
            self.feedback = "Counter reset to 0."
            # SILENT: No vocal alerts on reset
            self.audio_event = None
            self.vocal_message = ""

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

    def process_frame(self, frame, custom_landmarks=None):
        h, w, _ = frame.shape
        landmarks = custom_landmarks

        if landmarks is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

        with self.lock:
            audio_ev = None
            vocal_msg = None

            if landmarks is not None:
                self.warning_flags = []

                if self.exercise == "squat":
                    audio_ev, vocal_msg = self._process_squat(landmarks, w, h)
                elif self.exercise == "pushup":
                    audio_ev, vocal_msg = self._process_pushup(landmarks, w, h)
                else:
                    audio_ev, vocal_msg = self._process_curl(landmarks, w, h)

                self._draw_skeleton(frame, landmarks, w, h)
            else:
                self.feedback = "No person detected. Position body in frame."
                self.warning_flags = ["no_person"]

            # 4-second throttle check for backend alerts
            now = time.time()
            if audio_ev and audio_ev.startswith("warn_"):
                if now - self.last_audio_trigger_time >= 4.0:
                    self.audio_event = audio_ev
                    self.vocal_message = vocal_msg or ""
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
        # Track alignment between Shoulder, Hip, and Ankle
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

        # Hysteresis state machine:
        # stage == "down": Arm in extension (>150 deg). Waiting to reach full curl (<40 deg).
        # stage == "up": Full curl reached (<40 deg). Waiting to return to full extension (>150 deg) to complete rep.
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
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(landmarks):
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 98), (9, 13, 22), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.line(frame, (0, 98), (w, 98), (31, 41, 55), 1, cv2.LINE_AA)

        elapsed_sec = int(self.get_duration())
        mins, secs = divmod(elapsed_sec, 60)
        time_str = f"{mins:02d}:{secs:02d}"

        cv2.putText(frame, f"SISYPHUS AI  |  EXERCISE: {self.exercise.upper()}  |  TIME: {time_str}",
                    (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 198), 1, cv2.LINE_AA)

        cv2.putText(frame, "REPS", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)
        rep_col = (0, 255, 198) if self.stage == "down" else (255, 255, 255)
        cv2.putText(frame, str(self.counter), (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.25, rep_col, 3, cv2.LINE_AA)

        cv2.putText(frame, "STAGE", (115, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)
        stage_col = (0, 255, 198) if self.stage == "down" else (245, 158, 11)
        cv2.putText(frame, self.stage.upper(), (115, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.05, stage_col, 2, cv2.LINE_AA)

        if self.exercise == "squat":
            angle_title = "KNEE ANGLE"
        elif self.exercise == "pushup":
            angle_title = "ELBOW ANGLE"
        else:
            angle_title = "ARM ANGLE"

        cv2.putText(frame, angle_title, (230, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(self.smoothed_primary_angle)} deg", (230, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

        score_x = max(390, w - 200)
        cv2.putText(frame, "FORM SCORE", (score_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 163, 175), 1, cv2.LINE_AA)
        score_col = (0, 255, 198) if self.form_score >= 80 else ((245, 158, 11) if self.form_score >= 60 else (239, 68, 68))
        cv2.putText(frame, f"{self.form_score}%", (score_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.05, score_col, 2, cv2.LINE_AA)

        fb_h = 42
        fb_overlay = frame.copy()
        cv2.rectangle(fb_overlay, (0, h - fb_h), (w, h), (9, 13, 22), -1)
        cv2.addWeighted(fb_overlay, 0.8, frame, 0.2, 0, frame)

        msg_col = (239, 68, 68) if self.warning_flags else ((0, 255, 198) if "Complete" in self.feedback or "Great" in self.feedback else (243, 244, 246))
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, msg_col, 2, cv2.LINE_AA)


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
        # Background subtle gym styling
        cv2.rectangle(frame, (0, 0), (self.w, self.h), (12, 17, 28), -1)
        cv2.line(frame, (0, int(self.h * 0.88)), (self.w, int(self.h * 0.88)), (25, 35, 55), 2)

        # 33 MediaPipe pose landmarks list initialization
        landmarks = [MockLandmark(0.5, 0.5) for _ in range(33)]

        if exercise == "curl":
            # Bicep curl sinusoidal animation (period ~ 3.0s)
            cycle = (math.sin(t * 2.1) + 1.0) / 2.0  # 0.0 (extended) to 1.0 (curled)
            arm_angle_deg = 160.0 - cycle * 125.0    # 160° down to 35°
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

            # Draw dumbbell at hand position
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

        else: # pushup
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
# VIDEO STREAM & MULTI-SOURCE MANAGER
# ---------------------------------------------------------
class VideoStreamManager:
    def __init__(self, tracker):
        self.tracker = tracker
        self.cap = None
        self.source_mode = "demo"  # "demo", "webcam", or "browser_cam"
        self.latest_frame_bytes = None
        self.lock = threading.Lock()
        self.running = True
        self.synthetic_engine = SyntheticDemoEngine(640, 480)
        self.synthetic_time = 0.0
        self._open_source()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _get_demo_path(self):
        if self.tracker.exercise == "pushup" and os.path.exists(PUSHUP_DEMO_PATH):
            return PUSHUP_DEMO_PATH
        if self.tracker.exercise == "squat" and os.path.exists(SQUAT_DEMO_PATH):
            return SQUAT_DEMO_PATH
        return None

    def _open_source(self):
        if self.cap is not None:
            self.cap.release()
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

    def set_source(self, mode):
        with self.lock:
            self.source_mode = mode
            self._open_source()
            self.tracker.reset_reps()

    def update_demo_for_exercise(self):
        with self.lock:
            if self.source_mode == "demo":
                self._open_source()

    def _capture_loop(self):
        target_fps = 25.0
        frame_interval = 1.0 / target_fps

        while self.running:
            if self.source_mode == "browser_cam":
                time.sleep(0.1)
                continue

            start_t = time.time()
            self.synthetic_time += frame_interval

            frame = None
            custom_landmarks = None

            with self.lock:
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

            h, w = frame.shape[:2]
            if w > 720:
                scale = 720.0 / w
                frame = cv2.resize(frame, (720, int(h * scale)), interpolation=cv2.INTER_AREA)

            processed = self.tracker.process_frame(frame, custom_landmarks=custom_landmarks)
            ret, buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                with self.lock:
                    self.latest_frame_bytes = buf.tobytes()

            elapsed = time.time() - start_t
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def generate_frames(self):
        last_yielded = None
        while self.running:
            with self.lock:
                f_bytes = self.latest_frame_bytes

            if f_bytes and f_bytes != last_yielded:
                last_yielded = f_bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + f_bytes + b'\r\n')
            time.sleep(0.02)


tracker = ExerciseTracker()
stream_mgr = VideoStreamManager(tracker)


# ---------------------------------------------------------
# EMBEDDED HIGH-PERFORMANCE DARK-MODE UI
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
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --accent: #00ffc6;
      --accent-dim: rgba(0, 255, 198, 0.15);
      --accent-blue: #3b82f6;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text: #f3f4f6;
      --subtext: #9ca3af;
      --font: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .navbar {
      display: flex; justify-content: space-between; align-items: center;
      background: #030712; padding: 16px 32px; border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap; gap: 12px;
    }
    .logo {
      display: flex; align-items: center; gap: 10px;
      font-weight: 800; font-size: 1.25rem; color: var(--accent); letter-spacing: 0.05em;
      text-decoration: none;
    }
    .logo-badge {
      background: var(--accent-dim); border: 1px solid var(--accent);
      color: var(--accent); font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; font-weight: 700;
    }
    .nav-links { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .nav-links a {
      color: var(--subtext); text-decoration: none; font-size: 0.88rem; font-weight: 600;
      padding: 6px 12px; border-radius: 8px; transition: all 0.15s ease;
    }
    .nav-links a:hover, .nav-links a.active { color: #fff; background: rgba(255, 255, 255, 0.05); }
    .container { width: 100%; max-width: 1280px; margin: 24px auto; padding: 0 20px; flex: 1; }
    .card {
      background: var(--card-bg); border-radius: 14px; padding: 22px;
      border: 1px solid var(--card-border); box-shadow: 0 8px 24px -4px rgba(0,0,0,0.5);
    }
    .btn {
      background: var(--accent); color: #000; border: none; padding: 10px 18px;
      border-radius: 8px; font-weight: 700; font-size: 0.86rem; cursor: pointer;
      text-decoration: none; display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      transition: all 0.15s ease;
    }
    .btn:hover { opacity: 0.92; transform: translateY(-1px); }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08); color: var(--text); border: 1px solid var(--card-border);
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.15); color: #fff; }
    .btn-voice {
      background: rgba(0, 255, 198, 0.12); color: var(--accent); border: 1px solid rgba(0, 255, 198, 0.3);
    }
    .btn-voice.muted {
      background: rgba(239, 68, 68, 0.12); color: var(--accent-red); border-color: rgba(239, 68, 68, 0.3);
    }
    .stat-val { font-size: 2.3rem; font-weight: 800; font-family: var(--mono); color: var(--accent); margin: 6px 0; }
    .stat-sub { font-size: 0.8rem; color: var(--subtext); }
    select.custom-select {
      background: #030712; color: #fff; border: 1px solid var(--accent);
      border-radius: 8px; padding: 8px 12px; font-size: 0.88rem; font-weight: 700;
      cursor: pointer; outline: none;
    }
    input[type="text"], input[type="password"] {
      width: 100%; padding: 12px; border-radius: 8px; border: 1px solid var(--card-border);
      background: #030712; color: #fff; margin-bottom: 16px; box-sizing: border-box; font-family: var(--font);
    }
    input[type="text"]:focus, input[type="password"]:focus { outline: none; border-color: var(--accent); }
    .badge-tag {
      padding: 4px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    }
    .alert-box {
      padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 0.88rem; font-weight: 600;
    }
    .alert-success { background: rgba(0, 255, 198, 0.15); color: var(--accent); border: 1px solid var(--accent); }
    .alert-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid var(--accent-red); }
  </style>
</head>
<body>
  <div class="navbar">
    <a href="/" class="logo">
      <span>⚡ SISYPHUS AI</span>
      <span class="logo-badge">Smart Form Coach</span>
    </a>
    <div class="nav-links">
      {% if current_user.is_authenticated %}
        <span style="font-size: 0.84rem; color: var(--accent); font-weight: 700;">👤 {{ current_user.username }}</span>
        <a href="/workout">🏋️ Live Workout</a>
        <a href="/dashboard">📊 Dashboard</a>
        <a href="/logout" style="color: var(--accent-red);">Logout</a>
      {% else %}
        <a href="/workout">🏋️ Live Workout</a>
        <a href="/login">Login</a>
        <a href="/register" style="background: var(--accent); color: #000; font-weight: 700;">Register</a>
      {% endif %}
    </div>
  </div>

  <div class="container">
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
          <div class="alert-box alert-{{ category }}">{{ message }}</div>
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
<div style="display: grid; grid-template-columns: 1fr 380px; gap: 20px;">
  <!-- Video Feed & Setup -->
  <div class="card" style="padding: 16px; display: flex; flex-direction: column; gap: 14px;">
    <div style="position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 10px; overflow: hidden; border: 1px solid var(--card-border); display: flex; align-items: center; justify-content: center;">
      <img id="video-feed-img" src="/video_feed" alt="MediaPipe Tracking Feed" style="width: 100%; height: 100%; object-fit: contain; display: block;" />
    </div>

    <!-- Toolbar -->
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
      <div style="display: flex; align-items: center; gap: 10px;">
        <label for="exercise-select" style="font-size: 0.82rem; font-weight: 700; color: var(--subtext);">Exercise:</label>
        <select id="exercise-select" class="custom-select">
          <option value="squat" selected>Squats (Hip-Knee-Ankle)</option>
          <option value="pushup">Push-Ups (Chest & Spine)</option>
          <option value="curl">Dumbbell Curls (Shoulder-Elbow-Wrist)</option>
        </select>
      </div>

      <div style="display: flex; align-items: center; gap: 10px;">
        <label for="source-select" style="font-size: 0.82rem; font-weight: 700; color: var(--subtext);">Input:</label>
        <select id="source-select" class="custom-select">
          <option value="demo" selected>📹 Demo Video Engine</option>
          <option value="webcam">🎥 Live Webcam (cv2.VideoCapture(0))</option>
          <option value="browser_cam">💻 Browser Camera (getUserMedia)</option>
        </select>
      </div>
    </div>
  </div>

  <!-- Live Stats & Smart Coach Feedback -->
  <div style="display: flex; flex-direction: column; gap: 16px;">
    <!-- Reps Counter Card -->
    <div class="card" style="background: linear-gradient(145deg, #131b2e, #0f1624); border: 1px solid #1f293d;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.78rem; font-weight: 800; color: var(--subtext); text-transform: uppercase;">Repetitions</span>
        <span id="phase-badge" class="badge-tag" style="background: var(--accent-dim); color: var(--accent); border: 1px solid var(--accent);">UP</span>
      </div>
      <div class="stat-val" id="reps-num" style="font-size: 4.2rem; line-height: 1; margin: 12px 0;">0</div>
      <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: var(--subtext); border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
        <span>Session Time: <strong id="session-timer-text" style="color: var(--accent); font-family: var(--mono);">00:00</strong></span>
        <span id="active-side-text">Right Side</span>
      </div>
    </div>

    <!-- Joint Angles -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
      <div class="card" style="padding: 16px;">
        <span style="font-size: 0.74rem; font-weight: 800; color: var(--subtext); text-transform: uppercase;" id="primary-angle-label">Knee Angle</span>
        <div class="stat-val" id="primary-angle-val" style="font-size: 1.8rem;">180°</div>
      </div>
      <div class="card" style="padding: 16px;">
        <span style="font-size: 0.74rem; font-weight: 800; color: var(--subtext); text-transform: uppercase;" id="secondary-angle-label">Target Angle</span>
        <div class="stat-val" id="secondary-angle-val" style="font-size: 1.8rem;">≤ 100°</div>
      </div>
    </div>

    <!-- Form Score Card -->
    <div class="card" style="padding: 16px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.76rem; font-weight: 800; color: var(--subtext); text-transform: uppercase;">Form Quality Score</span>
        <span class="stat-val" id="score-val" style="font-size: 1.6rem; margin: 0;">100%</span>
      </div>
      <div style="width: 100%; height: 7px; background: rgba(0,0,0,0.5); border-radius: 999px; overflow: hidden; margin-top: 10px;">
        <div id="score-bar" style="width: 100%; height: 100%; background: var(--accent); border-radius: 999px; transition: width 0.3s ease;"></div>
      </div>
    </div>

    <!-- Smart Vocal Feedback Banner -->
    <div class="card" id="feedback-banner" style="background: rgba(0, 255, 198, 0.08); border-color: rgba(0, 255, 198, 0.3); display: flex; align-items: center; gap: 12px; padding: 14px;">
      <div id="feedback-avatar" style="width: 38px; height: 38px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #000; flex-shrink: 0;">🗣️</div>
      <div>
        <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--subtext); font-weight: 700;">Smart Form Coach</div>
        <div id="feedback-message" style="font-size: 0.92rem; font-weight: 700; color: #fff; margin-top: 2px;">Ready - Position body in frame</div>
      </div>
    </div>

    <!-- Action Buttons -->
    <div style="display: flex; gap: 10px; flex-direction: column;">
      <form action="/save_workout" method="POST" id="workout-save-form">
        <input type="hidden" name="exercise_type" id="hidden-exercise-type" value="Squats">
        <button type="submit" class="btn" style="width: 100%; padding: 12px;">
          💾 Finish & Save Workout
        </button>
      </form>

      <div style="display: flex; gap: 10px;">
        <button id="btn-voice-toggle" class="btn btn-voice" style="flex: 1;">
          <span id="voice-icon">🎙️</span> <span id="voice-label">Vocal: ON (Errors Only)</span>
        </button>
        <button id="btn-reset" class="btn btn-secondary" style="flex: 1;">
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
  let lastRepCount = 0;
  let currentInputSource = "demo";
  let isCapturing = false;
  let webcamStream = null;
  let currentBlobUrl = null;

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

  function formatTime(totalSec) {
    const s = Math.floor(totalSec || 0);
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return (mins < 10 ? "0" : "") + mins + ":" + (secs < 10 ? "0" : "") + secs;
  }

  // ----------------------------------------------------
  // SMART VOCAL FEEDBACK ENGINE:
  // - Speaks ONLY on specific form errors (warn_*)
  //   * Squats: 'Squat deeper!' or 'Keep your knees out!'
  //   * Push-ups: 'Lower your chest!' or 'Raise your hips!'
  //   * Dumbbell Curls: 'Keep your elbows tucked!'
  // - SILENT during good form, starting position, and successful rep counts
  // - Strict 4-second delay throttle between alerts
  // ----------------------------------------------------
  function speakFormError(errorMessage) {
    if (!voiceEnabled || !window.speechSynthesis || !errorMessage) return;
    const now = Date.now();
    if (now - lastSpokenTime < 4000) return; // Strict 4-second debounce

    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(errorMessage);
    utter.rate = 1.05;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
    lastSpokenTime = now;
  }

  btnVoiceToggle.addEventListener("click", () => {
    voiceEnabled = !voiceEnabled;
    if (voiceEnabled) {
      btnVoiceToggle.className = "btn btn-voice";
      voiceLabel.textContent = "Vocal: ON (Errors Only)";
    } else {
      btnVoiceToggle.className = "btn btn-voice muted";
      voiceLabel.textContent = "Vocal: MUTED";
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
  });

  function updateTelemetry(data) {
    if (!data) return;

    // Reps update VISUALLY only (speech synthesis remains silent on reps)
    if (data.reps !== lastRepCount) {
      lastRepCount = data.reps;
      repsEl.textContent = data.reps;
      repsEl.style.transform = "scale(1.2)";
      setTimeout(() => { repsEl.style.transform = "scale(1)"; }, 180);
    } else {
      repsEl.textContent = data.reps;
    }

    if (data.duration_seconds !== undefined) {
      sessionTimerEl.textContent = formatTime(data.duration_seconds);
    }

    const isDown = data.stage === "down";
    phaseBadgeEl.textContent = isDown ? "DOWN" : "UP";

    if (data.exercise === "squat") {
      primaryAngleLabel.textContent = "Knee Angle";
      primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
      secondaryAngleLabel.textContent = "Target Depth";
      secondaryAngleVal.textContent = "≤ 100°";
      activeSideText.textContent = `Active: ${data.active_side === 'right' ? 'Right Leg' : 'Left Leg'}`;
    } else if (data.exercise === "pushup") {
      primaryAngleLabel.textContent = "Elbow Angle";
      primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
      secondaryAngleLabel.textContent = "Spine Angle";
      secondaryAngleVal.textContent = Math.round(data.secondary_angle) + "°";
      activeSideText.textContent = `Flare: ${Math.round(data.flare_angle)}° (≤ 75°)`;
    } else {
      primaryAngleLabel.textContent = "Arm Angle";
      primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
      secondaryAngleLabel.textContent = "Curl Range";
      secondaryAngleVal.textContent = "< 40° / > 150°";
      activeSideText.textContent = `Active: ${data.active_side === 'right' ? 'Right Arm' : 'Left Arm'}`;
    }

    const score = Math.max(0, Math.min(100, data.form_score));
    scoreValEl.textContent = score + "%";
    scoreBarEl.style.width = score + "%";

    feedbackMsg.textContent = data.feedback;
    if (data.warnings && data.warnings.length > 0) {
      feedbackBanner.style.background = "rgba(239, 68, 68, 0.12)";
      feedbackBanner.style.borderColor = "rgba(239, 68, 68, 0.4)";
      feedbackAvatar.style.background = "var(--accent-red)";
    } else {
      feedbackBanner.style.background = "rgba(0, 255, 198, 0.08)";
      feedbackBanner.style.borderColor = "rgba(0, 255, 198, 0.3)";
      feedbackAvatar.style.background = "var(--accent)";
    }

    // Trigger speech ONLY on specific form errors with 4s debounce
    if (data.audio_event && data.audio_event.startsWith("warn_") && data.vocal_message) {
      speakFormError(data.vocal_message);
    }
  }

  async function pollStatus() {
    if (currentInputSource === "browser_cam") return;
    try {
      const res = await fetch("/status");
      if (res.ok) {
        const data = await res.json();
        updateTelemetry(data);
      }
    } catch (e) {}
  }
  setInterval(pollStatus, 250);

  exerciseSelect.addEventListener("change", async () => {
    const selected = exerciseSelect.value;
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
    videoFeedImg.src = "/video_feed?t=" + Date.now();
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
      videoFeedImg.src = "/video_feed?t=" + Date.now();
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
    sessionTimerEl.textContent = "00:00";
  });

  function runCameraCaptureLoop() {
    if (!isCapturing || currentInputSource !== "browser_cam") return;

    if (clientWebcam.readyState >= 2) {
      const vw = clientWebcam.videoWidth || 640;
      const vh = clientWebcam.videoHeight || 480;
      captureCanvas.width = vw;
      captureCanvas.height = vh;

      captureCtx.save();
      captureCtx.translate(vw, 0);
      captureCtx.scale(-1, 1);
      captureCtx.drawImage(clientWebcam, 0, 0, vw, vh);
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
      }, "image/jpeg", 0.8);

    } else {
      requestAnimationFrame(runCameraCaptureLoop);
    }
  }
</script>
"""

DASHBOARD_BODY = """
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px;">
  <div class="card">
    <div class="stat-sub">TOTAL REPS COMPLETED</div>
    <div class="stat-val">{{ total_reps }}</div>
    <div class="stat-sub">Across all exercises</div>
  </div>
  <div class="card">
    <div class="stat-sub">TOTAL WORKOUTS</div>
    <div class="stat-val" style="color: #fff;">{{ total_workouts }}</div>
    <div class="stat-sub">Completed sessions</div>
  </div>
  <div class="card">
    <div class="stat-sub">ACTIVE TIME</div>
    <div class="stat-val" style="color: var(--accent-blue);">{{ total_duration | format_duration }}</div>
    <div class="stat-sub">Under exercise tracking</div>
  </div>
  <div class="card">
    <div class="stat-sub">AVERAGE FORM</div>
    <div class="stat-val" style="color: var(--accent-amber);">{{ avg_form }}%</div>
    <div class="stat-sub">Movement accuracy</div>
  </div>
</div>

<div class="card" style="margin-bottom: 24px;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
    <div>
      <h3 style="font-size: 1.15rem; font-weight: 800; color: #fff;">Daily Rep Volume Over Time</h3>
      <p style="font-size: 0.78rem; color: var(--subtext);">Historical progress across logged workouts</p>
    </div>
    <span class="badge-tag" style="background: var(--accent-dim); color: var(--accent);">Activity</span>
  </div>
  <div style="height: 300px; width: 100%;">
    <canvas id="progressChart"></canvas>
  </div>
</div>

<div class="card">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
    <h3 style="font-size: 1.15rem; font-weight: 800; color: #fff;">Recent Exercise Sessions</h3>
    <a href="/workout" class="btn" style="padding: 6px 14px; font-size: 0.8rem;">Start New Workout</a>
  </div>

  {% if sessions %}
    <div style="overflow-x: auto;">
      <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem;">
        <thead>
          <tr style="border-bottom: 1px solid var(--card-border); color: var(--subtext); font-size: 0.74rem; text-transform: uppercase;">
            <th style="padding: 10px;">Date & Time</th>
            <th style="padding: 10px;">Exercise</th>
            <th style="padding: 10px;">Reps</th>
            <th style="padding: 10px;">Duration</th>
            <th style="padding: 10px;">Form Score</th>
          </tr>
        </thead>
        <tbody>
          {% for s in sessions %}
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
              <td style="padding: 12px 10px; font-family: var(--mono); color: var(--subtext); font-size: 0.8rem;">
                {{ s.timestamp.strftime('%b %d, %Y - %H:%M') if s.timestamp else 'N/A' }}
              </td>
              <td style="padding: 12px 10px; font-weight: 700;">
                {{ s.exercise_type }}
              </td>
              <td style="padding: 12px 10px; font-family: var(--mono); font-weight: 800; color: var(--accent);">
                {{ s.rep_count }}
              </td>
              <td style="padding: 12px 10px; font-family: var(--mono); color: var(--subtext);">
                {{ s.total_duration_seconds | format_duration }}
              </td>
              <td style="padding: 12px 10px; font-family: var(--mono); font-weight: 700; color: {% if s.avg_form_accuracy >= 80 %}var(--accent){% elif s.avg_form_accuracy >= 60 %}var(--accent-amber){% else %}var(--accent-red){% endif %};">
                {{ s.avg_form_accuracy }}%
              </td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  {% else %}
    <div style="text-align: center; padding: 40px; color: var(--subtext);">
      <p style="font-size: 1rem; color: #fff; margin-bottom: 6px;">No workout sessions recorded yet.</p>
      <p style="font-size: 0.82rem; margin-bottom: 16px;">Perform your first workout to start tracking your daily progress.</p>
      <a href="/workout" class="btn">Start Your First Workout</a>
    </div>
  {% endif %}
</div>

<script>
  const labels = {{ chart_labels | tojson }};
  const data = {{ chart_data | tojson }};
  const ctx = document.getElementById('progressChart').getContext('2d');

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels.length > 0 ? labels : ['No Data'],
      datasets: [{
        label: 'Repetitions',
        data: data.length > 0 ? data : [0],
        borderColor: '#00ffc6',
        backgroundColor: 'rgba(0, 255, 198, 0.1)',
        fill: true,
        tension: 0.35,
        borderWidth: 2.5,
        pointBackgroundColor: '#00ffc6',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', precision: 0 } }
      }
    }
  });
</script>
"""

AUTH_BODY = """
<div class="card" style="max-width: 420px; margin: 40px auto; padding: 32px 28px;">
  <div style="text-align: center; margin-bottom: 24px;">
    <div style="font-size: 2rem; margin-bottom: 8px;">⚡</div>
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #fff;">{{ title }}</h2>
    <p style="font-size: 0.82rem; color: var(--subtext);">{{ subtitle }}</p>
  </div>

  <form method="POST" action="{{ action }}">
    <label style="font-size: 0.8rem; font-weight: 700; color: var(--subtext); display: block; margin-bottom: 6px;">Username</label>
    <input type="text" name="username" placeholder="Username" required autocomplete="username">

    <label style="font-size: 0.8rem; font-weight: 700; color: var(--subtext); display: block; margin-bottom: 6px;">Password</label>
    <input type="password" name="password" placeholder="••••••••" required autocomplete="current-password">

    <button type="submit" class="btn" style="width: 100%; padding: 12px; margin-top: 8px;">
      {{ button_label }}
    </button>
  </form>

  <div style="text-align: center; font-size: 0.84rem; color: var(--subtext); margin-top: 18px;">
    {% if is_register %}
      Already have an account? <a href="/login" style="color: var(--accent); font-weight: 700; text-decoration: none;">Log In</a>
    {% else %}
      Don't have an account? <a href="/register" style="color: var(--accent); font-weight: 700; text-decoration: none;">Register</a>
    {% endif %}
  </div>
</div>
"""


# ---------------------------------------------------------
# FLASK ROUTE HANDLERS
# ---------------------------------------------------------
@app.route('/')
def index():
    """Default landing route."""
    return render_layout(WORKOUT_BODY)


@app.route('/workout')
def workout():
    """Live workout tracking view."""
    return render_layout(WORKOUT_BODY)


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
            return render_layout(
                AUTH_BODY,
                title="Create Account",
                subtitle="Join Sisyphus AI to record workouts",
                button_label="Sign Up",
                action="/register",
                is_register=True
            )

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("Username already exists. Please pick another.", "danger")
            return render_layout(
                AUTH_BODY,
                title="Create Account",
                subtitle="Join Sisyphus AI to record workouts",
                button_label="Sign Up",
                action="/register",
                is_register=True
            )

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        flash(f"Welcome, {new_user.username}!", "success")
        return redirect(url_for('dashboard'))

    return render_layout(
        AUTH_BODY,
        title="Create Account",
        subtitle="Join Sisyphus AI to record workouts",
        button_label="Sign Up",
        action="/register",
        is_register=True
    )


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

    return render_layout(
        AUTH_BODY,
        title="Welcome Back",
        subtitle="Sign in to view your workout analytics",
        button_label="Log In",
        action="/login",
        is_register=False
    )


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
    """Save active workout session into SQLite for the logged-in user."""
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
        avg_acc = tracker.get_avg_accuracy()

        if "rep_count" in data and data["rep_count"] is not None:
            reps = int(data["rep_count"])
        if "total_duration_seconds" in data and data["total_duration_seconds"] is not None:
            duration = float(data["total_duration_seconds"])
        if "avg_form_accuracy" in data and data["avg_form_accuracy"] is not None:
            avg_acc = float(data["avg_form_accuracy"])

        new_session = ExerciseSession(
            user_id=current_user.id,
            exercise_type=exercise_type,
            rep_count=reps,
            total_duration_seconds=round(duration, 1),
            avg_form_accuracy=round(avg_acc, 1),
            timestamp=datetime.now(timezone.utc)
        )
        db.session.add(new_session)
        db.session.commit()

        # Reset active session counter
        tracker.reset_reps()

    if request.is_json:
        return jsonify({
            "success": True,
            "message": "Workout saved successfully!",
            "session": new_session.to_dict()
        })

    flash(f"Workout saved! {reps} {exercise_type} logged.", "success")
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
@login_required
def dashboard():
    """User analytics dashboard."""
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

    daily_volume = defaultdict(int)
    for s in sorted(user_sessions, key=lambda x: x.timestamp):
        if s.timestamp:
            date_key = s.timestamp.strftime('%b %d')
            daily_volume[date_key] += s.rep_count

    chart_labels = list(daily_volume.keys())
    chart_data = list(daily_volume.values())

    return render_layout(
        DASHBOARD_BODY,
        sessions=user_sessions,
        total_reps=total_reps,
        total_workouts=total_workouts,
        total_duration=total_duration,
        avg_form=avg_form,
        chart_labels=chart_labels,
        chart_data=chart_data
    )


@app.route('/video_feed')
def video_feed():
    """Video feed endpoint."""
    return Response(
        stream_mgr.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """Real-time status endpoint."""
    data = tracker.get_status()
    data['source'] = stream_mgr.source_mode
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

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

        processed = tracker.process_frame(frame)
        ret, buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"[*] Starting Sisyphus AI Exercise Tracker on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

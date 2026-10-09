import os
import sys
import time
import math
import json
import urllib.parse
import threading
import cv2
import numpy as np
from flask import Flask, render_template_string, Response, jsonify, request

# MediaPipe Tasks imports
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

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
    Calculate elbow flare angle relative to the torso axis (Shoulder -> Hip).
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


# ---------------------------------------------------------
# EXERCISE FORM & REPETITION TRACKER
# ---------------------------------------------------------
class ExerciseTracker:
    def __init__(self, model_path=MODEL_PATH):
        self.lock = threading.Lock()
        self.exercise = "squat"  # "squat" or "pushup"
        self.counter = 0
        self.stage = "up"        # "up" or "down"

        # Filtered Joint Angles
        self.smoothed_primary_angle = 180.0
        self.smoothed_secondary_angle = 180.0
        self.smoothed_flare_angle = 0.0

        # Form Quality & Feedback
        self.form_score = 100
        self.rep_score_deductions = set()
        self.feedback = "Ready - Stand straight to begin"
        self.warning_flags = []
        self.audio_event = None
        self.vocal_message = ""
        self.min_angle_in_rep = 180.0
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
            if ex in ["squat", "pushup"] and ex != self.exercise:
                self.exercise = ex
                self.counter = 0
                self.stage = "up"
                self.form_score = 100
                self.rep_score_deductions.clear()
                self.smoothed_primary_angle = 180.0
                self.smoothed_secondary_angle = 180.0
                self.smoothed_flare_angle = 0.0
                self.warning_flags.clear()
                self.feedback = f"{ex.capitalize()} mode activated"
                self.audio_event = "mode_change"
                self.vocal_message = f"{ex.capitalize()} mode"

    def reset_reps(self):
        with self.lock:
            self.counter = 0
            self.stage = "up"
            self.form_score = 100
            self.rep_score_deductions.clear()
            self.warning_flags.clear()
            self.feedback = "Counter reset to 0."
            self.audio_event = "reset"
            self.vocal_message = "Counter reset"

    def get_status(self):
        with self.lock:
            return {
                "exercise": self.exercise,
                "reps": self.counter,
                "stage": self.stage,
                "primary_angle": round(self.smoothed_primary_angle, 1),
                "secondary_angle": round(self.smoothed_secondary_angle, 1),
                "flare_angle": round(self.smoothed_flare_angle, 1),
                "form_score": int(self.form_score),
                "feedback": self.feedback,
                "warnings": list(self.warning_flags),
                "audio_event": self.audio_event,
                "vocal_message": self.vocal_message,
                "active_side": self.active_side
            }

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        with self.lock:
            audio_ev = None
            vocal_msg = None

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                self.warning_flags = []

                if self.exercise == "squat":
                    audio_ev, vocal_msg = self._process_squat(landmarks, w, h)
                else:
                    audio_ev, vocal_msg = self._process_pushup(landmarks, w, h)

                self._draw_skeleton(frame, landmarks, w, h)
            else:
                self.feedback = "No person detected. Position body in frame."
                self.warning_flags = ["no_person"]

            if audio_ev and vocal_msg:
                self.audio_event = audio_ev
                self.vocal_message = vocal_msg

            self._draw_hud(frame)

        return frame

    def _process_squat(self, landmarks, w, h):
        """Squat tracking: Knee angle < 100 deg (down) and > 160 deg (up)."""
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

        # Requirement: Count rep when angle drops below 100° and returns above 160°
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

            # Rep Completion
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
                self.form_score = int(0.7 * self.form_score + 0.3 * rep_score)

                if deductions == 0:
                    audio_ev = "rep_success"
                    vocal_msg = f"{self.counter}! Perfect depth."
                    self.feedback = f"Rep #{self.counter} Complete! Perfect Depth (100%)"
                else:
                    audio_ev = "rep_warning"
                    critique = "Squat lower next time." if "shallow_depth" in self.rep_score_deductions else "Keep knees out."
                    vocal_msg = f"{self.counter}. {critique}"
                    self.feedback = f"Rep #{self.counter} Complete ({rep_score}% Form) - {critique}"

        # Real-time descent warnings
        if audio_ev is None:
            if knee_cave:
                audio_ev = "warn_kneecave"
                vocal_msg = "Push knees out"
            elif self.stage == "down" and knee_angle < 120.0 and self.min_angle_in_rep > 105.0:
                audio_ev = "warn_depth"
                vocal_msg = "Squat lower"

        return audio_ev, vocal_msg

    def _process_pushup(self, landmarks, w, h):
        """Push-up tracking: Elbow angle < 100 deg (down) and > 160 deg (up)."""
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

        # Form checks:
        # 1. Back sags or pikes (< 160 deg)
        back_sags = False
        if spine_angle < 160.0:
            back_sags = True
            self.warning_flags.append("Back sagging or piked! Keep straight plank.")
            self.rep_score_deductions.add("spine_sag")

        # 2. Elbow flare (> 75 deg)
        elbow_flare = False
        if elbow_angle < 140.0 and flare_angle > 75.0:
            elbow_flare = True
            self.warning_flags.append("Elbows flaring! Tuck elbows to 45°.")
            self.rep_score_deductions.add("elbow_flare")

        audio_ev = None
        vocal_msg = None

        # Requirement: Count rep when elbow angle drops below 100° and returns above 160°
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
                self.feedback = "Keep core tight! Do not sag back"

            # Rep Completion
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
                self.form_score = int(0.7 * self.form_score + 0.3 * rep_score)

                if deductions == 0:
                    audio_ev = "rep_success"
                    vocal_msg = f"{self.counter}! Great push up."
                    self.feedback = f"Rep #{self.counter} Complete! Solid Form (100%)"
                else:
                    audio_ev = "rep_warning"
                    critique = "Chest lower." if "shallow_chest" in self.rep_score_deductions else "Keep body straight."
                    vocal_msg = f"{self.counter}. {critique}"
                    self.feedback = f"Rep #{self.counter} Complete ({rep_score}% Form) - {critique}"

        if audio_ev is None:
            if back_sags:
                audio_ev = "warn_back"
                vocal_msg = "Keep back straight"
            elif elbow_flare:
                audio_ev = "warn_flare"
                vocal_msg = "Tuck your elbows"
            elif self.stage == "down" and elbow_angle < 120.0 and self.min_angle_in_rep > 105.0:
                audio_ev = "warn_chest"
                vocal_msg = "Lower chest"

        return audio_ev, vocal_msg

    def _draw_skeleton(self, frame, landmarks, w, h):
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        color = (0, 230, 80) if not self.warning_flags else (0, 165, 255)
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
        cv2.rectangle(overlay, (0, 0), (w, 95), (14, 17, 24), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 95), (w, 95), (45, 52, 70), 1, cv2.LINE_AA)

        cv2.putText(frame, f"EXERCISE: {self.exercise.upper()}", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "REPS", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        rep_col = (0, 240, 90) if self.stage == "down" else (255, 255, 255)
        cv2.putText(frame, str(self.counter), (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, rep_col, 3, cv2.LINE_AA)

        cv2.putText(frame, "STAGE", (120, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        stage_col = (0, 230, 80) if self.stage == "down" else (240, 160, 0)
        cv2.putText(frame, self.stage.upper(), (120, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, stage_col, 2, cv2.LINE_AA)

        angle_title = "KNEE ANGLE" if self.exercise == "squat" else "ELBOW ANGLE"
        cv2.putText(frame, angle_title, (240, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(self.smoothed_primary_angle)} deg", (240, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        score_x = max(390, w - 210)
        cv2.putText(frame, "FORM SCORE", (score_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        score_col = (0, 230, 80) if self.form_score >= 80 else ((0, 180, 255) if self.form_score >= 60 else (0, 80, 240))
        cv2.putText(frame, f"{self.form_score}%", (score_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, score_col, 2, cv2.LINE_AA)

        fb_h = 42
        fb_overlay = frame.copy()
        cv2.rectangle(fb_overlay, (0, h - fb_h), (w, h), (14, 17, 24), -1)
        cv2.addWeighted(fb_overlay, 0.8, frame, 0.2, 0, frame)

        msg_col = (0, 100, 245) if self.warning_flags else ((0, 230, 80) if "Complete" in self.feedback or "Great" in self.feedback else (220, 230, 245))
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, msg_col, 2, cv2.LINE_AA)


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
        self._open_source()

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _get_demo_path(self):
        if self.tracker.exercise == "pushup" and os.path.exists(PUSHUP_DEMO_PATH):
            return PUSHUP_DEMO_PATH
        return SQUAT_DEMO_PATH

    def _open_source(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.source_mode == "webcam":
            print("[*] Attempting to open live webcam (cv2.VideoCapture(0))...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[!] Physical webcam not found, switching back to demo video.")
                self.source_mode = "demo"
                demo_path = self._get_demo_path()
                self.cap = cv2.VideoCapture(demo_path)
            else:
                print("[+] Live webcam opened successfully.")
        elif self.source_mode == "demo":
            demo_path = self._get_demo_path()
            print(f"[*] Opening demo video source: {demo_path}")
            self.cap = cv2.VideoCapture(demo_path)

    def set_source(self, mode):
        with self.lock:
            self.source_mode = mode
            self._open_source()
            self.tracker.reset_reps()

    def update_demo_for_exercise(self):
        """When exercise changes in demo mode, switch to the appropriate demo video."""
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
            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    self._open_source()

                ret, frame = (False, None)
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    # Automatic video looping: reset to frame 0 when end of file is reached
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()

                if not ret or frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for video source...", (140, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            h, w = frame.shape[:2]
            if w > 720:
                scale = 720.0 / w
                frame = cv2.resize(frame, (720, int(h * scale)), interpolation=cv2.INTER_AREA)

            processed = self.tracker.process_frame(frame)
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


# ---------------------------------------------------------
# FLASK WEB APP & EMBEDDED DASHBOARD
# ---------------------------------------------------------
app = Flask(__name__)
tracker = ExerciseTracker()
stream_mgr = VideoStreamManager(tracker)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Vocal Fitness Coach - MediaPipe Exercise Tracker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #080a10;
      --card-bg: #111522;
      --card-border: #1e2638;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --font: 'Plus Jakarta Sans', system-ui, sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 18px;
    }
    .container { width: 100%; max-width: 1320px; display: flex; flex-direction: column; gap: 18px; }
    
    /* Header */
    .header {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 10px 30px -5px rgba(0,0,0,0.5);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-icon {
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-size: 1.4rem;
    }
    .brand-title h1 { font-size: 1.25rem; font-weight: 800; }
    .brand-title p { font-size: 0.78rem; color: var(--text-muted); }
    .header-actions { display: flex; align-items: center; gap: 12px; }
    .btn {
      padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; font-weight: 700;
      border: none; cursor: pointer; display: inline-flex; align-items: center; gap: 8px;
      transition: all 0.15s ease;
    }
    .btn-voice {
      background: rgba(16, 185, 129, 0.15); color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .btn-voice.muted {
      background: rgba(239, 68, 68, 0.15); color: var(--accent-red);
      border-color: rgba(239, 68, 68, 0.35);
    }
    .btn-reset {
      background: rgba(255, 255, 255, 0.08); color: var(--text-main);
      border: 1px solid var(--card-border);
    }
    .btn-reset:hover { background: rgba(255, 255, 255, 0.15); }
    
    /* Layout */
    .dashboard-grid {
      display: grid; grid-template-columns: 1fr 410px; gap: 18px;
    }
    @media (max-width: 1024px) { .dashboard-grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg); border: 1px solid var(--card-border);
      border-radius: 16px; padding: 20px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
    }

    /* Video Player */
    .video-card { padding: 14px; display: flex; flex-direction: column; gap: 14px; }
    .video-box {
      position: relative; width: 100%; aspect-ratio: 16 / 9;
      background: #000; border-radius: 12px; overflow: hidden;
      border: 1px solid var(--card-border); display: flex; align-items: center; justify-content: center;
    }
    #video-feed-img { width: 100%; height: 100%; object-fit: contain; display: block; }
    
    /* Toolbar Dropdowns */
    .video-toolbar {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
    }
    .control-group { display: flex; align-items: center; gap: 10px; }
    .control-group label { font-size: 0.82rem; font-weight: 700; color: var(--text-muted); }
    select.custom-select {
      background: #0b0e16; color: #fff; border: 1px solid var(--accent-blue);
      border-radius: 8px; padding: 8px 12px; font-size: 0.88rem; font-weight: 700;
      cursor: pointer; outline: none;
    }

    /* Analytics Cards */
    .metrics-col { display: flex; flex-direction: column; gap: 16px; }
    .hero-stat {
      background: linear-gradient(145deg, #151a28, #10131e);
      border: 1px solid #263046; position: relative; overflow: hidden;
    }
    .hero-stat-header { display: flex; justify-content: space-between; align-items: center; }
    .stat-title {
      font-size: 0.78rem; font-weight: 800; letter-spacing: 0.05em; color: var(--text-muted); text-transform: uppercase;
    }
    .phase-badge {
      padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 800;
      background: rgba(59, 130, 246, 0.2); color: var(--accent-cyan); border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .phase-badge.down {
      background: rgba(16, 185, 129, 0.2); color: var(--accent-green); border-color: rgba(16, 185, 129, 0.4);
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }
    .reps-display {
      font-size: 4.6rem; font-weight: 800; font-family: var(--mono); color: #fff;
      line-height: 1; margin: 12px 0; transition: transform 0.15s ease;
    }

    /* Metrics Row */
    .metrics-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .sub-stat { padding: 16px; }
    .stat-val {
      font-size: 1.9rem; font-weight: 800; font-family: var(--mono); color: var(--accent-cyan); margin-top: 6px;
    }

    /* Score Card */
    .score-card { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .score-info { flex: 1; }
    .score-val {
      font-size: 2.2rem; font-weight: 800; font-family: var(--mono); color: var(--accent-green);
    }
    .score-bar-bg {
      width: 100%; height: 8px; background: rgba(0,0,0,0.4); border-radius: 999px; overflow: hidden; margin-top: 8px;
    }
    .score-bar-fill {
      height: 100%; background: var(--accent-green); width: 100%; border-radius: 999px; transition: width 0.3s ease;
    }

    /* Feedback Banner */
    .feedback-banner {
      background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 16px; border-radius: 14px; display: flex; align-items: center; gap: 14px;
    }
    .feedback-banner.warning {
      background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.4);
    }
    .feedback-avatar {
      width: 38px; height: 38px; border-radius: 50%; background: var(--accent-green);
      display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0; color: #000;
    }
    .feedback-banner.warning .feedback-avatar { background: var(--accent-amber); }
    .feedback-text h4 { font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); }
    .feedback-text p { font-size: 0.95rem; font-weight: 700; color: #fff; margin-top: 2px; }

    /* Guide Checklist */
    .guide-card h3 { font-size: 0.82rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px; }
    .guide-list { list-style: none; display: flex; flex-direction: column; gap: 6px; font-size: 0.8rem; color: var(--text-muted); }
    .guide-list li { display: flex; align-items: center; gap: 8px; }
    .guide-bullet { width: 6px; height: 6px; border-radius: 50%; background: var(--accent-cyan); }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header class="header">
      <div class="brand">
        <div class="brand-icon">⚡</div>
        <div class="brand-title">
          <h1>AI Vocal Fitness Coach</h1>
          <p>Real-Time MediaPipe Pose Tracking (Squats & Push-Ups)</p>
        </div>
      </div>
      <div class="header-actions">
        <button id="btn-voice-toggle" class="btn btn-voice">
          <span id="voice-icon">🎙️</span>
          <span id="voice-label">Vocal Coach: ON</span>
        </button>
        <button id="btn-reset" class="btn btn-reset">
          🔄 Reset Reps
        </button>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="dashboard-grid">
      <!-- Video Stream Column -->
      <div class="card video-card">
        <div class="video-box">
          <img id="video-feed-img" src="/video_feed" alt="Real-Time MediaPipe Pose Feed" />
        </div>
        
        <!-- Toolbar with Exercise and Source Dropdowns -->
        <div class="video-toolbar">
          <div class="control-group">
            <label for="exercise-select">Exercise:</label>
            <select id="exercise-select" class="custom-select">
              <option value="squat" selected>Squats (Hip-Knee-Ankle)</option>
              <option value="pushup">Push-Ups (Chest & Spine)</option>
            </select>
          </div>

          <div class="control-group">
            <label for="source-select">Video Input Source:</label>
            <select id="source-select" class="custom-select">
              <option value="demo" selected>📹 Pre-Recorded Demo Video</option>
              <option value="webcam">🎥 Live Webcam (cv2.VideoCapture(0))</option>
              <option value="browser_cam">💻 Browser Camera (getUserMedia)</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Live Fitness Analytics Column -->
      <div class="metrics-col">
        <!-- Hero Reps Card -->
        <div class="card hero-stat">
          <div class="hero-stat-header">
            <span class="stat-title">Completed Repetitions</span>
            <span class="phase-badge" id="phase-badge">UP</span>
          </div>
          <div class="reps-display" id="reps-num">0</div>
          <div style="font-size: 0.8rem; color: var(--text-muted);" id="active-side-text">
            Active Side: Right Side
          </div>
        </div>

        <!-- Joint Angle Cards -->
        <div class="metrics-row">
          <div class="card sub-stat">
            <span class="stat-title" id="primary-angle-label">Knee Angle</span>
            <div class="stat-val" id="primary-angle-val">180°</div>
          </div>
          <div class="card sub-stat">
            <span class="stat-title" id="secondary-angle-label">Target Angle</span>
            <div class="stat-val" id="secondary-angle-val">≤ 100°</div>
          </div>
        </div>

        <!-- Form Quality Score -->
        <div class="card score-card">
          <div class="score-info">
            <span class="stat-title">Form Quality Score</span>
            <div class="score-val" id="score-val">100%</div>
            <div class="score-bar-bg">
              <div class="score-bar-fill" id="score-bar"></div>
            </div>
          </div>
        </div>

        <!-- AI Coach Feedback Banner -->
        <div class="feedback-banner" id="feedback-banner">
          <div class="feedback-avatar" id="feedback-avatar">🗣️</div>
          <div class="feedback-text">
            <h4>Live AI Vocal Coach</h4>
            <p id="feedback-message">Ready - Stand straight to begin</p>
          </div>
        </div>

        <!-- Form Checklist -->
        <div class="card guide-card">
          <h3 id="guide-title">Squat Form Checklist</h3>
          <ul class="guide-list" id="guide-list">
            <li><span class="guide-bullet"></span> Knee angle drops below 100° for valid depth</li>
            <li><span class="guide-bullet"></span> Returns above 160° to complete repetition</li>
            <li><span class="guide-bullet"></span> Warning triggered if knees cave or depth is shallow</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <!-- Hidden camera capture elements for client browser webcam -->
  <video id="client-webcam" playsinline autoplay muted style="display:none;"></video>
  <canvas id="capture-canvas" style="display:none;"></canvas>

  <script>
    let voiceEnabled = true;
    let lastSpokenText = "";
    let lastSpokenTime = 0;
    let lastRepCount = 0;
    let currentInputSource = "demo";
    let isCapturing = false;
    let webcamStream = null;
    let currentBlobUrl = null;

    const repsEl = document.getElementById("reps-num");
    const phaseBadgeEl = document.getElementById("phase-badge");
    const primaryAngleLabel = document.getElementById("primary-angle-label");
    const primaryAngleVal = document.getElementById("primary-angle-val");
    const secondaryAngleLabel = document.getElementById("secondary-angle-label");
    const secondaryAngleVal = document.getElementById("secondary-angle-val");
    const scoreValEl = document.getElementById("score-val");
    const scoreBarEl = document.getElementById("score-bar");
    const feedbackBanner = document.getElementById("feedback-banner");
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
    const guideTitle = document.getElementById("guide-title");
    const guideList = document.getElementById("guide-list");

    // ----------------------------------------------------
    // REAL-TIME VOCAL COACH (Web Speech API) WITH 3S DEBOUNCE
    // ----------------------------------------------------
    function speakVocalAlert(text, isRep = false) {
      if (!voiceEnabled || !window.speechSynthesis || !text) return;
      const now = Date.now();

      // Immediate callout for rep count
      if (isRep) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        window.speechSynthesis.speak(utter);
        lastSpokenText = text;
        lastSpokenTime = now;
        return;
      }

      // 3-second debounce throttle for repeated form correction alerts
      if (text === lastSpokenText && (now - lastSpokenTime < 3000)) {
        return;
      }

      // Throttle all form warnings to at least 1.5s interval
      if (now - lastSpokenTime > 1500) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        window.speechSynthesis.speak(utter);
        lastSpokenText = text;
        lastSpokenTime = now;
      }
    }

    btnVoiceToggle.addEventListener("click", () => {
      voiceEnabled = !voiceEnabled;
      if (voiceEnabled) {
        btnVoiceToggle.className = "btn btn-voice";
        voiceLabel.textContent = "Vocal Coach: ON";
        speakVocalAlert("Vocal coach activated", true);
      } else {
        btnVoiceToggle.className = "btn btn-voice muted";
        voiceLabel.textContent = "Vocal Coach: MUTED";
        if (window.speechSynthesis) window.speechSynthesis.cancel();
      }
    });

    // ----------------------------------------------------
    // TELEMETRY REFRESH
    // ----------------------------------------------------
    function updateTelemetry(data) {
      if (!data) return;

      if (data.reps !== lastRepCount) {
        const isNewRep = data.reps > lastRepCount;
        lastRepCount = data.reps;
        repsEl.textContent = data.reps;
        repsEl.style.transform = "scale(1.25)";
        setTimeout(() => { repsEl.style.transform = "scale(1)"; }, 200);

        if (isNewRep && data.vocal_message) {
          speakVocalAlert(data.vocal_message, true);
        }
      } else {
        repsEl.textContent = data.reps;
      }

      const isDown = data.stage === "down";
      phaseBadgeEl.textContent = isDown ? "DOWN" : "UP";
      phaseBadgeEl.className = isDown ? "phase-badge down" : "phase-badge";

      if (data.exercise === "squat") {
        primaryAngleLabel.textContent = "Knee Angle";
        primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
        secondaryAngleLabel.textContent = "Depth Target";
        secondaryAngleVal.textContent = "≤ 100°";
        activeSideText.textContent = `Active: ${data.active_side === 'right' ? 'Right Leg' : 'Left Leg'}`;
      } else {
        primaryAngleLabel.textContent = "Elbow Angle";
        primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
        secondaryAngleLabel.textContent = "Spine Alignment";
        secondaryAngleVal.textContent = Math.round(data.secondary_angle) + "°";
        activeSideText.textContent = `Flare: ${Math.round(data.flare_angle)}° (Max 75°)`;
      }

      const score = Math.max(0, Math.min(100, data.form_score));
      scoreValEl.textContent = score + "%";
      scoreBarEl.style.width = score + "%";
      if (score >= 80) {
        scoreValEl.style.color = "var(--accent-green)";
        scoreBarEl.style.background = "var(--accent-green)";
      } else if (score >= 60) {
        scoreValEl.style.color = "var(--accent-amber)";
        scoreBarEl.style.background = "var(--accent-amber)";
      } else {
        scoreValEl.style.color = "var(--accent-red)";
        scoreBarEl.style.background = "var(--accent-red)";
      }

      feedbackMsg.textContent = data.feedback;
      if (data.warnings && data.warnings.length > 0) {
        feedbackBanner.className = "feedback-banner warning";
        if (data.vocal_message && !data.vocal_message.startsWith(String(data.reps))) {
          speakVocalAlert(data.vocal_message, false);
        }
      } else {
        feedbackBanner.className = "feedback-banner";
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
    setInterval(pollStatus, 200);

    // ----------------------------------------------------
    // EXERCISE DROPDOWN
    // ----------------------------------------------------
    exerciseSelect.addEventListener("change", async () => {
      const selected = exerciseSelect.value;
      if (selected === "squat") {
        guideTitle.textContent = "Squat Form Checklist";
        guideList.innerHTML = `
          <li><span class="guide-bullet"></span> Knee angle drops below 100° for valid depth</li>
          <li><span class="guide-bullet"></span> Returns above 160° to complete repetition</li>
          <li><span class="guide-bullet"></span> Warning triggered if knees cave or depth is shallow</li>
        `;
      } else {
        guideTitle.textContent = "Push-Up Form Checklist";
        guideList.innerHTML = `
          <li><span class="guide-bullet"></span> Elbow angle drops below 100° for chest depth</li>
          <li><span class="guide-bullet"></span> Returns above 160° to complete repetition</li>
          <li><span class="guide-bullet"></span> Warning triggered if back sags (< 160°) or elbows flare (> 75°)</li>
        `;
      }

      await fetch("/set_exercise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exercise: selected })
      });
      speakVocalAlert(selected === "squat" ? "Squats mode selected" : "Push-ups mode selected", true);
      videoFeedImg.src = "/video_feed?t=" + Date.now();
    });

    // ----------------------------------------------------
    // SOURCE DROPDOWN (Demo vs Live Webcam vs Browser Cam)
    // ----------------------------------------------------
    sourceSelect.addEventListener("change", async () => {
      const selected = sourceSelect.value;
      currentInputSource = selected;

      // Clean up browser webcam stream if leaving browser_cam mode
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
          speakVocalAlert("Browser camera active", true);
        } catch(err) {
          alert("Could not access browser camera: " + err.message);
          sourceSelect.value = "demo";
          currentInputSource = "demo";
        }
      } else {
        // "demo" or "webcam"
        const res = await fetch("/set_source", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: selected })
        });
        const data = await res.json();
        videoFeedImg.src = "/video_feed?t=" + Date.now();
        if (selected === "webcam" && data.source === "demo") {
          alert("Physical webcam device index 0 is unavailable on this machine. Falling back to pre-recorded demo video.");
          sourceSelect.value = "demo";
        } else {
          speakVocalAlert(selected === "webcam" ? "Live webcam connected" : "Demo video loop loaded", true);
        }
      }
    });

    // Reset Button
    btnReset.addEventListener("click", async () => {
      await fetch("/reset", { method: "POST" });
      lastRepCount = 0;
      repsEl.textContent = "0";
      speakVocalAlert("Counter reset", true);
    });

    // In-browser camera capture loop
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
</body>
</html>
"""

# ---------------------------------------------------------
# FLASK ROUTE HANDLERS
# ---------------------------------------------------------
@app.route('/')
def index():
    """Serve the single-file HTML/CSS/JS frontend."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Stream camera or looped demo video frames using multipart/x-mixed-replace."""
    return Response(
        stream_mgr.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """Return JSON metrics and AI vocal feedback state."""
    data = tracker.get_status()
    data['source'] = stream_mgr.source_mode
    return jsonify(data)


@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    """Switch exercise mode between 'squat' and 'pushup'."""
    data = request.get_json(silent=True) or request.form
    ex = data.get('exercise', 'squat')
    tracker.set_exercise(ex)
    stream_mgr.update_demo_for_exercise()
    return jsonify({"success": True, "exercise": tracker.exercise})


@app.route('/reset', methods=['POST', 'GET'])
def reset():
    """Reset repetition counter and form score."""
    tracker.reset_reps()
    return jsonify({"success": True, "reps": 0})


@app.route('/set_source', methods=['POST'])
def set_source():
    """Switch source between 'demo', 'webcam', and 'browser_cam'."""
    data = request.get_json(silent=True) or request.form
    src = data.get('source', 'demo')
    if src in ['demo', 'webcam', 'browser_cam']:
        stream_mgr.set_source(src)
        return jsonify({"success": True, "source": stream_mgr.source_mode})
    return jsonify({"success": False, "error": "Invalid source"}), 400


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Process frames uploaded from in-browser camera (getUserMedia).
    Returns processed JPEG with skeletal tracking + HUD, and telemetry in X-Metrics header.
    """
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
            return jsonify({"error": "Failed to encode processed image"}), 500

        metrics = tracker.get_status()
        metrics_json = json.dumps(metrics)

        resp = Response(buf.tobytes(), mimetype='image/jpeg')
        resp.headers['X-Metrics'] = urllib.parse.quote(metrics_json)
        resp.headers['Access-Control-Expose-Headers'] = 'X-Metrics'
        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = '0.0.0.0'

    banner = f"""
======================================================================
🚀 AI VOCAL FITNESS COACH IS RUNNING!
----------------------------------------------------------------------
🌐 Local URL:             http://localhost:{port}
☁️ Google Cloud Shell:    Click 'Web Preview' -> 'Preview on port {port}'
🏋️ Exercises Supported:   Squats & Push-Ups (MediaPipe Pose)
📹 Demo Files:            squat_demo.mp4 & pushup_demo.mp4 (Auto-Looping)
🎥 Live Webcam:           cv2.VideoCapture(0) & In-Browser Camera
🎙️ AI Vocal Coach:        Web Speech API with 3s debounce
======================================================================
"""
    print(banner, flush=True)
    app.run(host=host, port=port, debug=False, threaded=True)

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
# CONSTANTS & SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')
SAMPLE_VIDEO_PATH = os.path.join(BASE_DIR, 'sample_squat.mp4')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'

def ensure_model_exists():
    """Ensure the MediaPipe Pose Landmarker model is downloaded."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print("[*] Downloading pose landmarker model...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Model download complete.")

ensure_model_exists()

# ---------------------------------------------------------
# ANGLE & MATH UTILITIES
# ---------------------------------------------------------
def calculate_angle(a, b, c):
    """
    Calculate 2D planar angle at joint b (in degrees) given points a, b, c.
    Returns angle in [0, 180].
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
    Calculate elbow flare angle relative to the torso.
    0 deg: elbow tucked tight to torso/ribs.
    45 deg: optimal push-up form.
    90 deg: flared perpendicular (T-shape).
    """
    v_torso = np.array(hip, dtype=np.float32) - np.array(shoulder, dtype=np.float32)
    v_arm = np.array(elbow, dtype=np.float32) - np.array(shoulder, dtype=np.float32)
    
    norms = np.linalg.norm(v_torso) * np.linalg.norm(v_arm)
    if norms == 0:
        return 0.0
    cos_ang = np.clip(np.dot(v_torso, v_arm) / norms, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))

def ema_filter(current_val, prev_val, alpha=0.35):
    """Exponential Moving Average smoothing to eliminate keypoint jitter."""
    if prev_val is None:
        return current_val
    return float(alpha * current_val + (1.0 - alpha) * prev_val)

# ---------------------------------------------------------
# EXERCISE FORM & AI COACH ENGINE
# ---------------------------------------------------------
class ExerciseTracker:
    def __init__(self, model_path=MODEL_PATH):
        self.lock = threading.Lock()
        self.exercise = "squat"  # "squat" or "pushup"
        self.counter = 0
        self.stage = "up"        # "up" or "down"
        
        # Smoothed angles
        self.smoothed_primary_angle = 180.0
        self.smoothed_secondary_angle = 180.0
        self.smoothed_flare_angle = 0.0
        
        # Form Scoring & Real-Time Feedback
        self.form_score = 100
        self.rep_score_deductions = set()
        self.feedback = "Ready - Stand straight to begin"
        self.warning_flags = []
        self.audio_event = None
        self.vocal_message = ""
        self.last_audio_trigger_time = 0
        
        # Rep depth tracking
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
                if ex == "squat":
                    self.feedback = "Squats Selected: Stand straight to begin"
                else:
                    self.feedback = "Push-Ups Selected: Get into plank position"
                self.audio_event = "exercise_changed"
                self.vocal_message = f"{self.exercise.capitalize()} mode activated"

    def reset_reps(self):
        with self.lock:
            self.counter = 0
            self.stage = "up"
            self.form_score = 100
            self.rep_score_deductions.clear()
            self.warning_flags.clear()
            self.feedback = "Rep counter reset to 0."
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
            # Reset one-shot vocal message per frame unless freshly triggered
            current_audio_event = None
            current_vocal_msg = None

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                self.warning_flags = []

                if self.exercise == "squat":
                    current_audio_event, current_vocal_msg = self._process_squat(landmarks, w, h)
                else:
                    current_audio_event, current_vocal_msg = self._process_pushup(landmarks, w, h)

                self._draw_skeleton(frame, landmarks, w, h)
            else:
                self.feedback = "No person detected. Step into camera view."
                self.warning_flags = ["no_person"]

            # Save vocal feedback
            if current_audio_event and current_vocal_msg:
                self.audio_event = current_audio_event
                self.vocal_message = current_vocal_msg

            # Render OpenCV Head-Up Display (HUD)
            self._draw_hud(frame)

        return frame

    def _process_squat(self, landmarks, w, h):
        """Squat landmark tracking, knee angle, knee cave check, and rep counting."""
        # Left side: 23 Hip, 25 Knee, 27 Ankle
        # Right side: 24 Hip, 26 Knee, 28 Ankle
        l_hip = [landmarks[23].x * w, landmarks[23].y * h]
        l_knee = [landmarks[25].x * w, landmarks[25].y * h]
        l_ankle = [landmarks[27].x * w, landmarks[27].y * h]

        r_hip = [landmarks[24].x * w, landmarks[24].y * h]
        r_knee = [landmarks[26].x * w, landmarks[26].y * h]
        r_ankle = [landmarks[28].x * w, landmarks[28].y * h]

        l_ang = calculate_angle(l_hip, l_knee, l_ankle)
        r_ang = calculate_angle(r_hip, r_knee, r_ankle)

        # Pick visible side
        r_vis = getattr(landmarks[26], 'visibility', 1.0)
        l_vis = getattr(landmarks[25], 'visibility', 1.0)
        if r_vis >= l_vis:
            raw_knee_angle = r_ang
            self.active_side = "right"
        else:
            raw_knee_angle = l_ang
            self.active_side = "left"

        # Apply EMA smoothing
        self.smoothed_primary_angle = ema_filter(raw_knee_angle, self.smoothed_primary_angle, alpha=0.35)
        knee_angle = self.smoothed_primary_angle

        # Check Knee Cave (Valgus)
        knee_dist = abs(landmarks[25].x - landmarks[26].x)
        ankle_dist = abs(landmarks[27].x - landmarks[28].x)
        hip_dist = abs(landmarks[23].x - landmarks[24].x)

        knee_cave_detected = False
        if knee_angle < 140.0 and ankle_dist > 0.1:
            if knee_dist < 0.72 * ankle_dist or knee_dist < 0.80 * hip_dist:
                knee_cave_detected = True
                self.warning_flags.append("Knees caving inward! Push knees out.")
                self.rep_score_deductions.add("knee_cave")

        audio_ev = None
        vocal_msg = None

        # Squat Repetition State Machine
        UP_THRESHOLD = 155.0
        DOWN_THRESHOLD = 100.0  # Must break parallel/deep squat

        if self.stage == "up":
            if knee_angle < 140.0:
                self.stage = "down"
                self.min_angle_in_rep = knee_angle
                self.rep_score_deductions.clear()
                self.feedback = "Squatting down... Go deep!"
            else:
                self.feedback = "Standing: Begin your squat"
        elif self.stage == "down":
            self.min_angle_in_rep = min(self.min_angle_in_rep, knee_angle)

            if knee_angle <= DOWN_THRESHOLD:
                self.feedback = "Good Depth! Now push up"
            elif knee_angle > 115.0 and self.min_angle_in_rep > DOWN_THRESHOLD:
                self.feedback = "Squat deeper! Knee angle must be <= 100°"
                self.warning_flags.append("Insufficient depth! Squat deeper.")
                self.rep_score_deductions.add("shallow_depth")

            if knee_cave_detected:
                self.feedback = "Watch knees! Push knees outward"

            # Transition back to UP (Rep Completed)
            if knee_angle >= UP_THRESHOLD:
                self.stage = "up"
                self.counter += 1
                self.last_rep_time = time.time()

                # Calculate score for this rep
                deductions = 0
                if "shallow_depth" in self.rep_score_deductions:
                    deductions += 30
                if "knee_cave" in self.rep_score_deductions:
                    deductions += 25
                
                rep_score = max(30, 100 - deductions)
                self.form_score = int(0.7 * self.form_score + 0.3 * rep_score)

                # Vocal feedback for rep
                if deductions == 0:
                    audio_ev = "rep_success"
                    vocal_msg = f"{self.counter}! Perfect depth."
                    self.feedback = f"Rep #{self.counter} Complete! Perfect Form (100%)"
                else:
                    audio_ev = "rep_warning"
                    critique = "Squat lower next time." if "shallow_depth" in self.rep_score_deductions else "Keep knees out."
                    vocal_msg = f"{self.counter}. {critique}"
                    self.feedback = f"Rep #{self.counter} Complete ({rep_score}% Form) - {critique}"

        # Real-time vocal warning if severe flaw during descent
        if audio_ev is None:
            if knee_cave_detected:
                audio_ev = "warn_kneecave"
                vocal_msg = "Push your knees out"
            elif self.stage == "down" and knee_angle < 120.0 and self.min_angle_in_rep > 105.0:
                audio_ev = "warn_depth"
                vocal_msg = "Go lower"

        return audio_ev, vocal_msg

    def _process_pushup(self, landmarks, w, h):
        """Push-up landmark tracking, elbow angle, spine alignment, and elbow flare."""
        # Left side: 11 Shoulder, 13 Elbow, 15 Wrist, 23 Hip, 27 Ankle
        # Right side: 12 Shoulder, 14 Elbow, 16 Wrist, 24 Hip, 28 Ankle
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

        # Smooth angles
        self.smoothed_primary_angle = ema_filter(raw_elbow_angle, self.smoothed_primary_angle, alpha=0.35)
        self.smoothed_secondary_angle = ema_filter(raw_spine_angle, self.smoothed_secondary_angle, alpha=0.35)
        self.smoothed_flare_angle = ema_filter(raw_flare_angle, self.smoothed_flare_angle, alpha=0.35)

        elbow_angle = self.smoothed_primary_angle
        spine_angle = self.smoothed_secondary_angle
        flare_angle = self.smoothed_flare_angle

        # Form checks
        # 1. Elbow flare (> 75 degrees)
        elbow_flare_detected = False
        if elbow_angle < 140.0 and flare_angle > 75.0:
            elbow_flare_detected = True
            self.warning_flags.append("Elbows flaring! Tuck elbows to 45 degrees.")
            self.rep_score_deductions.add("elbow_flare")

        # 2. Spine alignment / Hips sagging or piking (< 160 degrees)
        spine_fault_detected = False
        if spine_angle < 160.0:
            spine_fault_detected = True
            self.warning_flags.append("Hips sagging or piked! Keep body in straight line.")
            self.rep_score_deductions.add("spine_alignment")

        audio_ev = None
        vocal_msg = None

        # Push-Up State Machine
        PLANK_THRESHOLD = 150.0  # Arms extended
        CHEST_DEPTH_THRESHOLD = 90.0  # Chest to floor, elbow <= 90 deg

        if self.stage == "up":
            if elbow_angle < 135.0:
                self.stage = "down"
                self.min_angle_in_rep = elbow_angle
                self.rep_score_deductions.clear()
                self.feedback = "Descending... Lower your chest!"
            else:
                self.feedback = "Plank Position: Begin push-up"
        elif self.stage == "down":
            self.min_angle_in_rep = min(self.min_angle_in_rep, elbow_angle)

            if elbow_angle <= CHEST_DEPTH_THRESHOLD:
                self.feedback = "Great Chest Depth! Push back up"
            elif elbow_angle > 105.0 and self.min_angle_in_rep > CHEST_DEPTH_THRESHOLD:
                self.feedback = "Go lower! Elbows must bend to <= 90°"
                self.warning_flags.append("Chest not low enough! Go deeper.")
                self.rep_score_deductions.add("shallow_chest")

            # Transition back to UP (Rep Completed)
            if elbow_angle >= PLANK_THRESHOLD:
                self.stage = "up"
                self.counter += 1
                self.last_rep_time = time.time()

                deductions = 0
                if "shallow_chest" in self.rep_score_deductions:
                    deductions += 30
                if "elbow_flare" in self.rep_score_deductions:
                    deductions += 25
                if "spine_alignment" in self.rep_score_deductions:
                    deductions += 25

                rep_score = max(25, 100 - deductions)
                self.form_score = int(0.7 * self.form_score + 0.3 * rep_score)

                if deductions == 0:
                    audio_ev = "rep_success"
                    vocal_msg = f"{self.counter}! Excellent push up."
                    self.feedback = f"Rep #{self.counter} Complete! Solid Form (100%)"
                else:
                    audio_ev = "rep_warning"
                    critique = "Chest lower." if "shallow_chest" in self.rep_score_deductions else "Keep body straight."
                    vocal_msg = f"{self.counter}. {critique}"
                    self.feedback = f"Rep #{self.counter} Complete ({rep_score}% Form) - {critique}"

        # Real-time warnings during descent
        if audio_ev is None:
            if elbow_flare_detected:
                audio_ev = "warn_flare"
                vocal_msg = "Tuck your elbows"
            elif spine_fault_detected:
                audio_ev = "warn_spine"
                vocal_msg = "Keep body straight"
            elif self.stage == "down" and elbow_angle < 110.0 and self.min_angle_in_rep > 95.0:
                audio_ev = "warn_chest"
                vocal_msg = "Chest lower"

        return audio_ev, vocal_msg

    def _draw_skeleton(self, frame, landmarks, w, h):
        """Draw aesthetic keypoint connections and joints on frame."""
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24), # Torso
            (11, 13), (13, 15), (12, 14), (14, 16), # Arms
            (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
        ]
        
        # Color based on current form quality
        bone_color = (0, 230, 80) if not self.warning_flags else (0, 165, 255)
        
        for p1_idx, p2_idx in connections:
            if p1_idx < len(landmarks) and p2_idx < len(landmarks):
                p1 = (int(landmarks[p1_idx].x * w), int(landmarks[p1_idx].y * h))
                p2 = (int(landmarks[p2_idx].x * w), int(landmarks[p2_idx].y * h))
                cv2.line(frame, p1, p2, bone_color, 2, cv2.LINE_AA)

        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(landmarks):
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, bone_color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame):
        """Draw modern OpenCV Head-Up Display directly on camera frame."""
        h, w, _ = frame.shape

        # Top Banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (14, 17, 24), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 95), (w, 95), (45, 52, 70), 1, cv2.LINE_AA)

        # Exercise Mode Badge
        ex_label = f"EXERCISE: {self.exercise.upper()}"
        cv2.putText(frame, ex_label, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1, cv2.LINE_AA)

        # Rep Counter
        cv2.putText(frame, "REPS", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        rep_col = (0, 240, 90) if self.stage == "down" else (255, 255, 255)
        cv2.putText(frame, str(self.counter), (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, rep_col, 3, cv2.LINE_AA)

        # Stage Badge
        cv2.putText(frame, "STAGE", (120, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        stage_col = (0, 230, 80) if self.stage == "down" else (240, 160, 0)
        cv2.putText(frame, self.stage.upper(), (120, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, stage_col, 2, cv2.LINE_AA)

        # Primary Angle Display
        angle_name = "KNEE ANGLE" if self.exercise == "squat" else "ELBOW ANGLE"
        cv2.putText(frame, angle_name, (240, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(self.smoothed_primary_angle)} deg", (240, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        # Form Quality Score Gauge
        score_x = max(390, w - 210)
        cv2.putText(frame, "FORM SCORE", (score_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        score_col = (0, 230, 80) if self.form_score >= 80 else ((0, 180, 255) if self.form_score >= 60 else (0, 80, 240))
        cv2.putText(frame, f"{self.form_score}%", (score_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, score_col, 2, cv2.LINE_AA)

        # Bottom Feedback Strip
        fb_h = 42
        fb_y = h - fb_h
        fb_overlay = frame.copy()
        cv2.rectangle(fb_overlay, (0, fb_y), (w, h), (14, 17, 24), -1)
        cv2.addWeighted(fb_overlay, 0.8, frame, 0.2, 0, frame)

        # Feedback icon / text
        msg_col = (0, 100, 245) if self.warning_flags else ((0, 230, 80) if "Complete" in self.feedback or "Great" in self.feedback else (220, 230, 245))
        fb_text = f"COACH: {self.feedback}"
        cv2.putText(frame, fb_text, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, msg_col, 2, cv2.LINE_AA)


# ---------------------------------------------------------
# VIDEO STREAM & CAMERA MANAGER
# ---------------------------------------------------------
class VideoStreamManager:
    def __init__(self, tracker):
        self.tracker = tracker
        self.cap = None
        self.source_type = "sample"  # "camera", "sample", or "browser_camera"
        self.is_camera_available = False
        self.latest_frame_bytes = None
        self.lock = threading.Lock()
        self.running = True
        self._detect_camera()
        self._open_stream()

        # Dedicated background processing thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _detect_camera(self):
        """Check for physical camera availability."""
        try:
            test_cap = cv2.VideoCapture(0)
            if test_cap and test_cap.isOpened():
                self.is_camera_available = True
                self.source_type = "camera"
                test_cap.release()
            else:
                self.is_camera_available = False
                self.source_type = "sample"
        except Exception:
            self.is_camera_available = False
            self.source_type = "sample"

    def _open_stream(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.source_type == "camera":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[!] Physical camera (device 0) unavailable, falling back to sample loop.")
                self.source_type = "sample"
                self.cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
        else:
            self.cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)

    def switch_source(self, st):
        with self.lock:
            self.source_type = st
            self._open_stream()
            self.tracker.reset_reps()

    def _capture_loop(self):
        target_fps = 25.0
        frame_interval = 1.0 / target_fps

        while self.running:
            # When client browser webcam is streaming via /process_frame, pause server loop
            if self.source_type == "browser_camera":
                time.sleep(0.1)
                continue

            start_t = time.time()
            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    self._open_stream()

                ret, frame = (False, None)
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
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
# FLASK WEB APP & EMBEDDED TEMPLATE
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
  <title>AI Vocal Fitness Coach - MediaPipe Pose Form Tracker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #090b10;
      --card-bg: #121622;
      --card-border: #202738;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --text-primary: #f8fafc;
      --text-muted: #94a3b8;
      --font: 'Plus Jakarta Sans', system-ui, sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-dark);
      color: var(--text-primary);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }
    .container {
      width: 100%;
      max-width: 1320px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    /* Top Header */
    .header {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-icon {
      width: 44px; height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-size: 1.4rem;
    }
    .brand-title h1 { font-size: 1.25rem; font-weight: 800; }
    .brand-title p { font-size: 0.78rem; color: var(--text-muted); }
    .header-controls {
      display: flex; align-items: center; gap: 12px;
    }
    .btn {
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 700;
      border: none;
      cursor: pointer;
      display: inline-flex; align-items: center; gap: 8px;
      transition: all 0.15s ease;
    }
    .btn-voice {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .btn-voice.muted {
      background: rgba(239, 68, 68, 0.15);
      color: var(--accent-red);
      border-color: rgba(239, 68, 68, 0.35);
    }
    .btn-reset {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-primary);
      border: 1px solid var(--card-border);
    }
    .btn-reset:hover { background: rgba(255, 255, 255, 0.15); }
    
    /* Layout Grid */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr 400px;
      gap: 20px;
    }
    @media (max-width: 1024px) {
      .dashboard-grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
    }
    
    /* Video Section */
    .video-card { padding: 14px; display: flex; flex-direction: column; gap: 14px; }
    .video-box {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--card-border);
      display: flex; align-items: center; justify-content: center;
    }
    #video-feed-img {
      width: 100%; height: 100%; object-fit: contain; display: block;
    }
    .video-toolbar {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    }
    .exercise-picker {
      display: flex; align-items: center; gap: 10px;
    }
    .exercise-picker label { font-size: 0.85rem; font-weight: 700; color: var(--text-muted); }
    select.custom-select {
      background: #0d111a;
      color: #fff;
      border: 1px solid var(--accent-blue);
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 0.9rem;
      font-weight: 700;
      cursor: pointer;
      outline: none;
    }
    .source-buttons {
      display: flex; gap: 6px;
    }
    .btn-source {
      background: transparent; color: var(--text-muted);
      border: 1px solid var(--card-border);
      border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; cursor: pointer;
    }
    .btn-source.active {
      background: var(--accent-blue); color: #fff; border-color: var(--accent-blue);
    }
    
    /* Metrics Column */
    .metrics-col {
      display: flex; flex-direction: column; gap: 18px;
    }
    .hero-stat {
      background: linear-gradient(145deg, #161b29, #111420);
      border: 1px solid #28324a;
      position: relative; overflow: hidden;
    }
    .hero-stat-header {
      display: flex; justify-content: space-between; align-items: center;
    }
    .stat-title {
      font-size: 0.8rem; font-weight: 800; letter-spacing: 0.05em; color: var(--text-muted); text-transform: uppercase;
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
      font-size: 4.8rem; font-weight: 800; font-family: var(--mono); color: #fff;
      line-height: 1; margin: 12px 0;
      transition: transform 0.15s ease;
    }
    
    /* Metrics Row */
    .metrics-row {
      display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    }
    .sub-stat { padding: 16px; }
    .stat-val {
      font-size: 2rem; font-weight: 800; font-family: var(--mono); color: var(--accent-cyan); margin-top: 6px;
    }
    
    /* Form Score Card */
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
    
    /* Coach Feedback Banner */
    .feedback-banner {
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 16px; border-radius: 14px; display: flex; align-items: center; gap: 14px;
    }
    .feedback-banner.warning {
      background: rgba(245, 158, 11, 0.08);
      border-color: rgba(245, 158, 11, 0.4);
    }
    .feedback-avatar {
      width: 38px; height: 38px; border-radius: 50%; background: var(--accent-green);
      display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0; color: #000;
    }
    .feedback-banner.warning .feedback-avatar { background: var(--accent-amber); }
    .feedback-text h4 { font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); }
    .feedback-text p { font-size: 0.95rem; font-weight: 700; color: #fff; margin-top: 2px; }

    /* Instructions & Checklist Card */
    .guide-card h3 { font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px; }
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
      <div class="header-controls">
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
        <div class="video-toolbar">
          <div class="exercise-picker">
            <label for="exercise-select">Exercise:</label>
            <select id="exercise-select" class="custom-select">
              <option value="squat" selected>Squats (Hip-Knee-Ankle)</option>
              <option value="pushup">Push-Ups (Chest & Spine)</option>
            </select>
          </div>
          <div class="source-buttons">
            <button id="btn-cam-source" class="btn-source">💻 Browser Camera</button>
            <button id="btn-sample-source" class="btn-source active">📹 Demo Stream</button>
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
            <h4>Live AI Coach</h4>
            <p id="feedback-message">Ready - Stand straight to begin</p>
          </div>
        </div>

        <!-- Form Checklist -->
        <div class="card guide-card">
          <h3 id="guide-title">Squat Form Checklist</h3>
          <ul class="guide-list" id="guide-list">
            <li><span class="guide-bullet"></span> Knee angle reaches ≤ 100° for deep squat</li>
            <li><span class="guide-bullet"></span> Knees must not cave inward (keep pushed out)</li>
            <li><span class="guide-bullet"></span> Stand straight (≥ 155°) to finish repetition</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <!-- Hidden camera capture elements for client browser webcam -->
  <video id="client-webcam" playsinline autoplay muted style="display:none;"></video>
  <canvas id="capture-canvas" style="display:none;"></canvas>

  <script>
    // State & Elements
    let voiceEnabled = true;
    let lastSpokenText = "";
    let lastSpokenTime = 0;
    let lastRepCount = 0;
    let currentMode = "sample";
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
    const btnVoiceToggle = document.getElementById("btn-voice-toggle");
    const voiceLabel = document.getElementById("voice-label");
    const btnReset = document.getElementById("btn-reset");
    const btnCamSource = document.getElementById("btn-cam-source");
    const btnSampleSource = document.getElementById("btn-sample-source");
    const videoFeedImg = document.getElementById("video-feed-img");
    const clientWebcam = document.getElementById("client-webcam");
    const captureCanvas = document.getElementById("capture-canvas");
    const captureCtx = captureCanvas.getContext("2d");
    const guideTitle = document.getElementById("guide-title");
    const guideList = document.getElementById("guide-list");

    // ----------------------------------------------------
    // AI VOCAL COACH (Web Speech API) WITH 3-SECOND DEBOUNCE
    // ----------------------------------------------------
    function speakVocalAlert(text, isRep = false) {
      if (!voiceEnabled || !window.speechSynthesis || !text) return;
      const now = Date.now();

      // If rep complete (e.g. "1! Perfect depth"), speak immediately
      if (isRep) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        utter.pitch = 1.0;
        window.speechSynthesis.speak(utter);
        lastSpokenText = text;
        lastSpokenTime = now;
        return;
      }

      // Debounce non-rep coaching warnings: 3-second throttle for same advice
      if (text === lastSpokenText && (now - lastSpokenTime < 3000)) {
        return;
      }

      // If different warning or 3 seconds passed
      if (now - lastSpokenTime > 1500) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        utter.pitch = 1.0;
        window.speechSynthesis.speak(utter);
        lastSpokenText = text;
        lastSpokenTime = now;
      }
    }

    // Toggle Voice Button
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
    // TELEMETRY UPDATE
    // ----------------------------------------------------
    function updateTelemetry(data) {
      if (!data) return;

      // Check for Rep Count bump
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

      // Phase/Stage Badge
      const isDown = data.stage === "down";
      phaseBadgeEl.textContent = isDown ? "DOWN" : "UP";
      phaseBadgeEl.className = isDown ? "phase-badge down" : "phase-badge";

      // Angles & Labels
      if (data.exercise === "squat") {
        primaryAngleLabel.textContent = "Knee Angle";
        primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
        secondaryAngleLabel.textContent = "Depth Target";
        secondaryAngleVal.textContent = "≤ 100°";
        activeSideText.textContent = `Active: ${data.active_side === 'right' ? 'Right Leg' : 'Left Leg'}`;
      } else {
        primaryAngleLabel.textContent = "Elbow Angle";
        primaryAngleVal.textContent = Math.round(data.primary_angle) + "°";
        secondaryAngleLabel.textContent = "Body Alignment";
        secondaryAngleVal.textContent = Math.round(data.secondary_angle) + "°";
        activeSideText.textContent = `Flare: ${Math.round(data.flare_angle)}° (Max 75°)`;
      }

      // Form Quality Score
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

      // Feedback Banner
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

    // Polling server status
    async function pollStatus() {
      if (currentMode === "browser_camera") return;
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
    // EXERCISE SELECTION SWITCH
    // ----------------------------------------------------
    exerciseSelect.addEventListener("change", async () => {
      const selected = exerciseSelect.value;
      if (selected === "squat") {
        guideTitle.textContent = "Squat Form Checklist";
        guideList.innerHTML = `
          <li><span class="guide-bullet"></span> Knee angle reaches ≤ 100° for deep squat</li>
          <li><span class="guide-bullet"></span> Knees must not cave inward (keep pushed out)</li>
          <li><span class="guide-bullet"></span> Stand straight (≥ 155°) to finish repetition</li>
        `;
      } else {
        guideTitle.textContent = "Push-Up Form Checklist";
        guideList.innerHTML = `
          <li><span class="guide-bullet"></span> Lower chest until elbow angle reaches ≤ 90°</li>
          <li><span class="guide-bullet"></span> Keep elbows tucked at 45° (flare must stay < 75°)</li>
          <li><span class="guide-bullet"></span> Maintain straight plank (hips must not sag/pike)</li>
        `;
      }

      const res = await fetch("/set_exercise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exercise: selected })
      });
      const data = await res.json();
      speakVocalAlert(selected === "squat" ? "Squats mode selected" : "Push-ups mode selected", true);
    });

    // Reset Button
    btnReset.addEventListener("click", async () => {
      await fetch("/reset", { method: "POST" });
      lastRepCount = 0;
      repsEl.textContent = "0";
      speakVocalAlert("Counter reset", true);
    });

    // ----------------------------------------------------
    // IN-BROWSER WEBCAM STREAMING (getUserMedia)
    // ----------------------------------------------------
    btnCamSource.addEventListener("click", async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Webcam access not supported in this browser.");
        return;
      }
      try {
        if (webcamStream) {
          webcamStream.getTracks().forEach(t => t.stop());
        }
        webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false
        });
        clientWebcam.srcObject = webcamStream;
        await clientWebcam.play();

        currentMode = "browser_camera";
        btnCamSource.classList.add("active");
        btnSampleSource.classList.remove("active");
        isCapturing = true;

        await fetch("/set_source", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: "browser_camera" })
        });

        runCameraCaptureLoop();
        speakVocalAlert("Browser camera connected", true);
      } catch (err) {
        alert("Camera access denied or unavailable: " + err.message);
      }
    });

    btnSampleSource.addEventListener("click", async () => {
      isCapturing = false;
      if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
      }
      clientWebcam.srcObject = null;
      if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
        currentBlobUrl = null;
      }

      currentMode = "sample";
      btnSampleSource.classList.add("active");
      btnCamSource.classList.remove("active");

      await fetch("/set_source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "sample" })
      });
      videoFeedImg.src = "/video_feed?t=" + Date.now();
    });

    function runCameraCaptureLoop() {
      if (!isCapturing || currentMode !== "browser_camera") return;

      if (clientWebcam.readyState >= 2) {
        const vw = clientWebcam.videoWidth || 640;
        const vh = clientWebcam.videoHeight || 480;
        captureCanvas.width = vw;
        captureCanvas.height = vh;

        captureCtx.save();
        captureCtx.translate(vw, 0);
        captureCtx.scale(-1, 1); // Selfie mirror view
        captureCtx.drawImage(clientWebcam, 0, 0, vw, vh);
        captureCtx.restore();

        captureCanvas.toBlob(async (blob) => {
          if (!blob || !isCapturing || currentMode !== "browser_camera") {
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

          if (isCapturing && currentMode === "browser_camera") {
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
# FLASK ROUTE DEFINITIONS
# ---------------------------------------------------------
@app.route('/')
def index():
    """Serve the single-file HTML/CSS/JS frontend."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Stream camera / demo video frames using multipart/x-mixed-replace."""
    return Response(
        stream_mgr.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    """Return JSON metrics and AI vocal feedback state."""
    data = tracker.get_status()
    data['source'] = stream_mgr.source_type
    return jsonify(data)

@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    """Switch exercise mode between 'squat' and 'pushup'."""
    data = request.get_json(silent=True) or request.form
    ex = data.get('exercise', 'squat')
    tracker.set_exercise(ex)
    return jsonify({"success": True, "exercise": tracker.exercise})

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    """Reset repetition counter and form score."""
    tracker.reset_reps()
    return jsonify({"success": True, "reps": 0})

@app.route('/set_source', methods=['POST'])
def set_source():
    """Switch source between browser camera and sample demo."""
    data = request.get_json(silent=True) or request.form
    src = data.get('source', 'sample')
    if src in ['sample', 'browser_camera', 'camera']:
        stream_mgr.switch_source(src)
        return jsonify({"success": True, "source": stream_mgr.source_type})
    return jsonify({"success": False, "error": "Invalid source"}), 400

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Process frames uploaded from client browser webcam via getUserMedia.
    Returns processed JPEG with skeletal tracking + HUD, and telemetry in X-Metrics header.
    """
    try:
        if stream_mgr.source_type != "browser_camera":
            stream_mgr.source_type = "browser_camera"

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
# ENTRY POINT
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
🏋️ Supported Exercises:   Squats & Push-Ups (MediaPipe Pose)
🎙️ AI Vocal Coach:        Web Speech API with 3s debounce
📷 Camera:                Automatic physical camera fallback &
                          in-browser webcam support (getUserMedia)
======================================================================
"""
    print(banner, flush=True)
    app.run(host=host, port=port, debug=False, threaded=True)

import os
import time
import math
import json
import base64
import urllib.parse
import urllib.request
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename

# MediaPipe Tasks imports
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')
SAMPLE_VIDEO_PATH = os.path.join(BASE_DIR, 'sample_squat.mp4')
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'


def ensure_model_exists():
    """Download the MediaPipe pose landmarker model if not present."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print(f"Downloading pose landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model download complete.")


ensure_model_exists()


def calculate_angle(a, b, c):
    """
    Calculate 2D angle (in degrees) at joint b given points a, b, c.
    a: hip [x, y]
    b: knee [x, y] (vertex)
    c: ankle [x, y]
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return float(angle)


class SquatTracker:
    """Tracks squat repetitions, knee joint angles, and pose keypoints."""

    def __init__(self, model_path=MODEL_PATH):
        self.lock = threading.Lock()
        self.counter = 0
        self.stage = "up"  # "up" or "down"
        self.feedback = "Ready - Stand straight to begin"
        self.current_angle = 180.0
        self.left_angle = 180.0
        self.right_angle = 180.0
        self.active_side = "right"
        self.squat_pct = 0.0

        # Configurable thresholds
        self.down_threshold = 100.0  # Angle to count as reached squat depth
        self.up_threshold = 155.0    # Angle to count as standing upright

        # Rep history
        self.rep_history = []
        self.last_rep_time = 0

        # Initialize MediaPipe PoseLandmarker
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def reset(self):
        with self.lock:
            self.counter = 0
            self.stage = "up"
            self.feedback = "Counter reset. Stand straight to begin."
            self.rep_history.clear()

    def set_thresholds(self, down_th, up_th):
        with self.lock:
            self.down_threshold = max(60.0, min(130.0, float(down_th)))
            self.up_threshold = max(135.0, min(175.0, float(up_th)))

    def get_status(self):
        with self.lock:
            return {
                "reps": self.counter,
                "stage": self.stage,
                "angle": round(self.current_angle, 1),
                "left_angle": round(self.left_angle, 1),
                "right_angle": round(self.right_angle, 1),
                "active_side": self.active_side,
                "depth_pct": round(self.squat_pct, 1),
                "feedback": self.feedback,
                "down_threshold": self.down_threshold,
                "up_threshold": self.up_threshold,
            }

    def process_frame(self, frame):
        """Run pose detection, compute knee angle, update rep state, and draw HUD."""
        h, w, _ = frame.shape

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)

        with self.lock:
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                # Keypoint indices:
                # 23: Left Hip, 25: Left Knee, 27: Left Ankle
                # 24: Right Hip, 26: Right Knee, 28: Right Ankle
                l_hip = [landmarks[23].x * w, landmarks[23].y * h]
                l_knee = [landmarks[25].x * w, landmarks[25].y * h]
                l_ankle = [landmarks[27].x * w, landmarks[27].y * h]

                r_hip = [landmarks[24].x * w, landmarks[24].y * h]
                r_knee = [landmarks[26].x * w, landmarks[26].y * h]
                r_ankle = [landmarks[28].x * w, landmarks[28].y * h]

                self.left_angle = calculate_angle(l_hip, l_knee, l_ankle)
                self.right_angle = calculate_angle(r_hip, r_knee, r_ankle)

                # Determine active leg based on visibility / presence confidence
                l_vis = getattr(landmarks[25], 'visibility', 1.0)
                r_vis = getattr(landmarks[26], 'visibility', 1.0)

                if r_vis >= l_vis:
                    self.active_side = "right"
                    active_angle = self.right_angle
                    hip_pt, knee_pt, ankle_pt = r_hip, r_knee, r_ankle
                else:
                    self.active_side = "left"
                    active_angle = self.left_angle
                    hip_pt, knee_pt, ankle_pt = l_hip, l_knee, l_ankle

                self.current_angle = active_angle

                # Calculate squat depth percentage (0% = standing, 100% = target depth)
                angle_range = self.up_threshold - self.down_threshold
                if angle_range > 0:
                    pct = ((self.up_threshold - active_angle) / angle_range) * 100.0
                    self.squat_pct = float(np.clip(pct, 0.0, 100.0))
                else:
                    self.squat_pct = 0.0

                # Repetition State Machine
                if active_angle < self.down_threshold:
                    if self.stage != "down":
                        self.stage = "down"
                        self.feedback = "Good Depth! Now Push Up"
                elif active_angle > self.up_threshold:
                    if self.stage == "down":
                        self.stage = "up"
                        self.counter += 1
                        self.last_rep_time = time.time()
                        self.feedback = f"Rep #{self.counter} Complete! Great Form"
                        self.rep_history.append({
                            "rep": self.counter,
                            "time": time.time(),
                            "side": self.active_side
                        })
                    elif self.stage == "up":
                        if time.time() - self.last_rep_time > 1.8:
                            self.feedback = "Standing - Descend for next rep"
                else:
                    # In transition
                    if self.stage == "down":
                        self.feedback = "Pushing Up..."
                    else:
                        self.feedback = "Squatting Down..."

                # Draw Pose Skeleton and Highlights
                self._draw_pose_skeleton(frame, landmarks, w, h)
                self._draw_angle_callout(frame, hip_pt, knee_pt, ankle_pt, active_angle)

            else:
                self.feedback = "No person detected. Step into frame."

            # Draw Modern On-Screen HUD Overlay
            self._draw_hud(frame)

        return frame

    def _draw_pose_skeleton(self, frame, landmarks, w, h):
        """Draw aesthetic keypoint connections and joints on the video frame."""
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32),
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, p1, p2, (230, 200, 100), 2, cv2.LINE_AA)

        for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
            if idx < len(landmarks):
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, (230, 200, 100), 1, cv2.LINE_AA)

    def _draw_angle_callout(self, frame, hip, knee, ankle, angle):
        """Draw highlighted knee angle lines, circular vertex, and text badge."""
        p_hip = (int(hip[0]), int(hip[1]))
        p_knee = (int(knee[0]), int(knee[1]))
        p_ankle = (int(ankle[0]), int(ankle[1]))

        if angle <= self.down_threshold:
            color = (0, 230, 80)       # Vibrant Green for valid squat depth
        elif angle >= self.up_threshold:
            color = (255, 200, 0)      # Cyan / Light Blue for upright
        else:
            color = (0, 180, 255)      # Amber / Orange in transition

        cv2.line(frame, p_hip, p_knee, color, 4, cv2.LINE_AA)
        cv2.line(frame, p_knee, p_ankle, color, 4, cv2.LINE_AA)

        cv2.circle(frame, p_knee, 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p_knee, 14, (255, 255, 255), 2, cv2.LINE_AA)

        text = f"{int(angle)} deg"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        bx = max(10, min(frame.shape[1] - tw - 20, p_knee[0] + 18))
        by = max(30, min(frame.shape[0] - 20, p_knee[1] - 10))

        cv2.rectangle(frame, (bx - 6, by - th - 6), (bx + tw + 6, by + 6), (20, 20, 24), -1)
        cv2.rectangle(frame, (bx - 6, by - th - 6), (bx + tw + 6, by + 6), color, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (bx, by), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_hud(self, frame):
        """Draw an informational HUD overlay directly onto the video feed."""
        h, w, _ = frame.shape

        banner_h = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 18, 26), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        cv2.line(frame, (0, banner_h), (w, banner_h), (60, 65, 85), 1, cv2.LINE_AA)

        rep_color = (0, 230, 80) if self.stage == "down" else (255, 255, 255)
        cv2.putText(frame, "REPS", (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 185, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter), (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.6, rep_color, 3, cv2.LINE_AA)

        stage_text = self.stage.upper()
        stage_bg = (0, 200, 80) if self.stage == "down" else (240, 160, 0)
        cv2.putText(frame, "STAGE", (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 185, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, stage_text, (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, stage_bg, 3, cv2.LINE_AA)

        cv2.putText(frame, "KNEE ANGLE", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 185, 200), 1, cv2.LINE_AA)
        angle_str = f"{int(self.current_angle)} deg"
        cv2.putText(frame, angle_str, (310, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        bar_x = max(490, w - 240)
        bar_y = 52
        bar_w = 210
        bar_h = 18

        cv2.putText(frame, f"DEPTH: {int(self.squat_pct)}%", (bar_x, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 225), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 60), -1)

        fill_w = int((self.squat_pct / 100.0) * bar_w)
        fill_color = (0, 230, 80) if self.squat_pct >= 95 else (0, 180, 255)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 110, 130), 1, cv2.LINE_AA)

        fb_h = 44
        fb_y = h - fb_h
        fb_overlay = frame.copy()
        cv2.rectangle(fb_overlay, (0, fb_y), (w, h), (16, 18, 26), -1)
        cv2.addWeighted(fb_overlay, 0.75, frame, 0.25, 0, frame)

        fb_color = (0, 230, 80) if ("Complete" in self.feedback or "Depth" in self.feedback) else (220, 225, 240)
        cv2.putText(frame, f"FEEDBACK: {self.feedback}", (24, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fb_color, 2, cv2.LINE_AA)


class VideoStreamManager:
    """Manages demo video capture, auto-pausing when client-side browser camera is active."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.cap = None
        self.source_type = "sample"  # "sample", "custom", or "browser_camera"
        self.custom_path = None
        self.latest_frame_bytes = None
        self.lock = threading.Lock()
        self.running = True
        self._open_stream()

        # Start background processing loop for server video
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _open_stream(self):
        """Open the server-side video file."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.source_type == "custom" and self.custom_path and os.path.exists(self.custom_path):
            self.cap = cv2.VideoCapture(self.custom_path)
        elif self.source_type == "sample":
            self.cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)

    def switch_source(self, source_type, file_path=None):
        with self.lock:
            self.source_type = source_type
            if file_path:
                self.custom_path = file_path
            self._open_stream()
            self.tracker.reset()

    def _capture_loop(self):
        """Background loop generating frames when in server demo or custom video mode."""
        target_fps = 25.0
        frame_interval = 1.0 / target_fps

        while self.running:
            # If user is using their browser camera directly, pause server loop to conserve CPU
            if self.source_type == "browser_camera":
                time.sleep(0.1)
                continue

            start_time = time.time()
            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    self._open_stream()

                ret, frame = (False, None)
                if self.cap is not None and self.cap.isOpened():
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

            processed_frame = self.tracker.process_frame(frame)

            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                with self.lock:
                    self.latest_frame_bytes = buffer.tobytes()

            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def generate_frames(self):
        """Yield multipart JPEG frames to connected HTTP clients."""
        last_yielded = None
        while self.running:
            with self.lock:
                frame_bytes = self.latest_frame_bytes

            if frame_bytes and frame_bytes != last_yielded:
                last_yielded = frame_bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.02)


# Global instances
tracker = SquatTracker()
stream_manager = VideoStreamManager(tracker)


# Flask Routes
@app.route('/')
def index():
    """Main dashboard HTML page."""
    return render_template(
        'index.html',
        source=stream_manager.source_type
    )


@app.route('/video_feed')
def video_feed():
    """Streaming endpoint for the processed server-side video feed."""
    return Response(
        stream_manager.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Process an individual video frame captured directly from the client's browser webcam.
    Returns the processed JPEG with skeletal tracking and HUD overlays, plus telemetry.
    """
    try:
        # Switch mode to browser_camera so server loop pauses
        if stream_manager.source_type != "browser_camera":
            stream_manager.source_type = "browser_camera"

        # Check for uploaded file or raw binary post
        if 'frame' in request.files:
            file_bytes = np.frombuffer(request.files['frame'].read(), np.uint8)
        else:
            raw_data = request.get_data()
            if not raw_data:
                return jsonify({"error": "No frame data received"}), 400
            file_bytes = np.frombuffer(raw_data, np.uint8)

        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Downscale large webcam inputs for smooth real-time response
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Process pose keypoints & squat repetitions
        processed_frame = tracker.process_frame(frame)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return jsonify({"error": "Failed to encode frame"}), 500

        metrics = tracker.get_status()
        metrics_json = json.dumps(metrics)

        # Return JPEG binary image with X-Metrics header
        resp = Response(buffer.tobytes(), mimetype='image/jpeg')
        resp.headers['X-Metrics'] = urllib.parse.quote(metrics_json)
        resp.headers['Access-Control-Expose-Headers'] = 'X-Metrics'
        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status')
def status():
    """API endpoint providing current exercise metrics in JSON."""
    data = tracker.get_status()
    data['source'] = stream_manager.source_type
    return jsonify(data)


@app.route('/reset', methods=['POST', 'GET'])
def reset():
    """Reset the squat rep counter."""
    tracker.reset()
    return jsonify({"success": True, "message": "Rep counter reset to 0"})


@app.route('/set_source', methods=['POST'])
def set_source():
    """Switch video source between 'browser_camera', 'sample', or 'custom'."""
    data = request.get_json(silent=True) or request.form
    source = data.get('source', 'sample')
    if source in ['browser_camera', 'sample']:
        stream_manager.switch_source(source)
        return jsonify({"success": True, "source": stream_manager.source_type})
    return jsonify({"success": False, "error": "Invalid source"}), 400


@app.route('/set_thresholds', methods=['POST'])
def set_thresholds():
    """Update knee angle thresholds for squat depth and upright standing."""
    data = request.get_json(silent=True) or request.form
    down_th = data.get('down_threshold')
    up_th = data.get('up_threshold')
    if down_th is not None and up_th is not None:
        tracker.set_thresholds(float(down_th), float(up_th))
        return jsonify({
            "success": True,
            "down_threshold": tracker.down_threshold,
            "up_threshold": tracker.up_threshold
        })
    return jsonify({"success": False, "error": "Missing parameters"}), 400


@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload a custom workout video for squat analysis."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    stream_manager.switch_source("custom", file_path=save_path)
    return jsonify({"success": True, "filename": filename})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

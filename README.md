# AI Squat Tracker & Form Analyzer (Flask + MediaPipe Pose)

An AI-powered computer vision application built with **Flask**, **MediaPipe Pose**, and **OpenCV** that accesses the computer's video camera within the browser, tracks body keypoints for squats, computes real-time knee joint angles, detects rep completion, and streams the processed video feed to a web dashboard.

---

## Features

- **Direct In-Browser Webcam Access (`getUserMedia`)**:
  - Accesses the computer's webcam directly inside the browser using the HTML5 `MediaDevices.getUserMedia()` API.
  - Works across client laptops, desktops, and mobile devices even when running Flask on a remote or headless server.
  - Device selector for switching between front/back/USB webcams.
  - "Mirror View (Selfie)" toggle for natural mirror preview.
- **MediaPipe Pose Keypoint Tracking**: Tracks 33 3D body keypoints, highlighting hips (23/24), knees (25/26), and ankles (27/28) with dynamic skeleton overlays.
- **Knee Joint Angle Calculation**: Calculates the knee joint angle formed by Hip $\rightarrow$ Knee (vertex) $\rightarrow$ Ankle using trigonometry:
  $$\theta = |\text{atan2}(y_{\text{ankle}} - y_{\text{knee}}, x_{\text{ankle}} - x_{\text{knee}}) - \text{atan2}(y_{\text{hip}} - y_{\text{knee}}, x_{\text{hip}} - x_{\text{knee}})| \times \frac{180}{\pi}$$
- **Automated Repetition Counter**: State machine (`UP` $\leftrightarrow$ `DOWN`):
  - **Down Stage**: Triggered when knee angle drops below the squat target threshold (default $\le 100^\circ$).
  - **Up Stage (Rep Complete)**: Triggered when user returns to upright standing posture (default $\ge 155^\circ$), incrementing the counter and triggering visual celebration feedback.
- **Real-Time Video HUD Overlay**: Renders Rep Count, Exercise Stage, Knee Angle badge, Squat Depth Meter, and Coaching Feedback directly on the video feed.
- **Multi-Source Support**:
  - **Browser Camera**: Uses the computer's camera in the browser.
  - **Demo Squat Video**: Built-in demonstration video (`sample_squat.mp4`) for testing without a camera.
  - **Custom Video Upload**: Upload custom workout clips to analyze form.

---

## Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask web app:**
   ```bash
   python3 app.py
   ```

3. **Open in browser:**
   Navigate to `http://localhost:5000`.
   - Click **"Use Browser Camera"** to grant camera access and start live squat tracking.

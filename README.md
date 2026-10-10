# Sisyphus AI

> **Real-time AI pose tracking, biomechanical form correction, and gamified workout dashboard powered by MediaPipe and Flask.**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-black.svg)](https://palletsprojects.com/p/flask/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00A67E.svg)](https://developers.google.com/mediapipe)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Tests](https://img.shields.io/badge/Tests-28%20Passing-brightgreen.svg)](test_app.py)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Site-blue.svg?logo=github)](https://reach-1191.github.io/sisyphus/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Sisyphus AI** is an intelligent fitness companion designed to elevate your personal training. Combining state-of-the-art computer vision pose estimation with real-time biometric analysis and client-side vocal coaching, Sisyphus tracks your workout form rep-by-rep, detects kinematic faults before they lead to injury, and turns fitness into an addictive RPG leveling experience.

Try it here: [Sisyphus Live App](https://sisyphus-783417011137.northamerica-northeast1.run.app/workout?exercise=pushup)

---

## Features:

- ⚡ **Adaptive 1€ Filter (One Euro Filter) Motion Tracking**:
  - Replaced rigid static filtering with adaptive velocity-aware 1€ filters (Casiez et al., 2012) on joint angles and 2D keypoints.
  - Eradicates high-frequency spatial jitter during static holds (planks, squat pauses) while eliminating latency and phase lag on rapid explosive ascents.
- 🎯 **Landmark Jitter Reduction & Confidence Weighting**:
  - Independent 2D coordinate smoothing on all primary keypoints with visibility-based confidence weighting (`visibility >= 0.35`).
  - Gracefully bridges momentary single-frame occlusions without dropping skeleton tracking or triggering false "no person detected" flickers.
- 📐 **Active Side Kinematic Hysteresis**:
  - Added visibility threshold hysteresis (`0.15` margin) to prevent rapid alternating between left and right limb tracking when facing the camera.
- 🏋️ **Repetition Kinematic & Depth Precision**:
  - Fixed premature shallow depth deductions: reaching valid depth ($\le 100^\circ$) confirms depth compliance and clears premature deductions.
  - Optimized dumbbell curl peak contraction threshold ($\le 45^\circ$) for realistic biomechanical range of motion.
- 🌐 **In-Browser Webcam Integration (WebRTC / `getUserMedia`)**:
  - Fixed client-side browser webcam pipeline: captures local camera video stream via `navigator.mediaDevices.getUserMedia`, mirrors frames on a hidden canvas for intuitive selfie viewing, and posts frames to `/process_frame`.
  - Server processes each webcam frame with MediaPipe pose estimation, draws crisp anti-aliased HUD skeleton overlays with active joint ring highlights, and streams annotated frames back to the client alongside dynamic `X-Metrics` telemetry headers.
  - Safe lifecycle management: releases camera hardware tracks on mode changes, pauses, and page unloads.
- 🐳 **Docker Containerization**: Full production-ready `Dockerfile` (Python 3.10-slim with OpenCV/FFmpeg system libraries) and `.dockerignore` for single-command builds and isolated deployment on any cloud or container platform.
- 📊 **Post-Workout Form Analytics & Coach Summary (`/summary/<session_id>`)**:
  - **Dynamic 'Coach Insights' Feedback Card**: Prominent Bento card styled in dark charcoal (`#18181b`) with vibrant warm yellow accent borders (`#facc15`), providing ultra-concise, affirmative **Positive Encouragement** and actionable **Technical Improvement Tips** tailored per exercise (under 12 words per sentence, strictly prohibiting litotes or double negatives). Displays flawless precision guidance when session accuracy reaches or exceeds 90%.
  - **Form Status Rating**: Classifies movement quality into **Optimal** (≥ 85%), **Moderate** (70–84%), or **Needs Adjustment** (< 70%).
  - **Biomechanical Precision Feedback**: Exercise-specific movement reviews (depth consistency, joint alignment, descent tempo, spinal plank alignment, tucked elbow flare protection, peak bicep contraction, and eccentric control).
  - **Frame-Accurate Form Score**: Dynamically calculated using valid repetition frame ratios against flagged kinematic warnings:
    $$\text{Form Accuracy} = \frac{\text{Valid Rep Frames}}{\text{Valid Rep Frames} + \text{Form Warnings}} \times 100\%$$
  - **Session Rewards**: Displays EXP earned (with form score bonus multiplier), current level progress, and unlocked achievement badges.
- 🔐 **Dedicated Authentication Suite**:
  - Independent auth views (`login.html`, `register.html`, `layout.html`) with Bento-inspired aesthetic, validation, and session management.
  - Password hashing via Werkzeug PBKDF2 and session cookies via Flask-Login.
- 🎯 **Adaptive Athlete Dashboard**:
  - Contextual greetings: First-time athletes receive a tailored welcome and Quick Onboarding Guide; returning athletes see streak counters, total reps, and level badges.
  - Visual exercise selection cards with targeted muscle groups (Glutes / Legs for Squats, Chest / Core for Push-ups, Biceps for Dumbbell Curls).
- 📹 **Optimized Headless Demo Fallbacks**:
  - High-definition fallback demo videos for all three supported exercises (`squat_demo.mp4`, `pushup_demo.mp4`, `dumbbell_demo.mp4`), enabling out-of-the-box testing without a webcam.
- 🧪 **Comprehensive Automated Test Suite**:
  - 28 automated unit and integration tests in `test_app.py` covering OneEuroFilter, LandmarkSmoother, browser webcam streaming, kinematics, auth flows, camera lifecycle, EXP/badges, audio alerts, and summary views.

---

## Key Features

- **Bento Box UI Layout**: Dynamic, responsive aesthetic organizing live camera feed, real-time joint angle gauges, rep velocity, workout history, and EXP progress into intuitive modular cards.
- **Multi-Exercise Pose Estimation**:
  - **Squats**: Precise hip-knee-ankle kinematic tracking, depth validation (≤ 100°), EMA filter smoothing ($\alpha = 0.35$), and knee cave (valgus) detection.
  - **Push-Ups**: Chest depth measurement (≤ 90°), elbow flare detection (> 75°), and real-time **back-sag detection** ensuring strict spinal plank alignment (≥ 160°).
  - **Dumbbell Curls**: Bilateral arm tracking, peak contraction angles (≤ 50°), full extension validation (≥ 155°), and excessive elbow sway/drift warnings (≤ 15°).
- **Client-Side Web Speech API Voice Coaching**: Zero-latency in-browser vocal corrections with a smart **4-second debounced error throttling** algorithm, providing audible coaching tips only when form faults occur without nagging.
- **Persistent EXP, Leveling, and Badge Unlocks**: Earn EXP for every completed rep (with bonus multipliers for high-fidelity form scores). Level up dynamically ($\lfloor\sqrt{\text{EXP}/50}\rfloor + 1$) and unlock achievement badges automatically (e.g., *First Step*, *Form Perfectionist*, *Iron Core*, *Century Club*).
- **Camera Hardware Lifecycle Control**: Dedicated pause and resume endpoints (`/pause_feed`, `/resume_feed`) safely release hardware video capture to avoid device lockouts.
- **Headless Video Fallback Support**: Seamlessly falls back to pre-packaged high-definition demo video streams if no local webcam is detected—enabling zero-configuration testing on headless Linux virtual machines and cloud environments.

---

## Quick Start Guide

You can launch Sisyphus using **Docker**, **Automated Setup Script**, or directly in **Google Cloud Shell** / **GitHub Codespaces**.

### Option A: Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/reach-1191/sisyphus.git
   cd sisyphus
   ```

2. **Build and run the container**:
   ```bash
   docker build -t sisyphus .
   docker run -p 8080:8080 sisyphus
   ```

3. **Open in browser**: Navigate to [http://localhost:8080](http://localhost:8080).

---

### Option B: Local Environment with `setup.sh`

1. **Clone the repository**:
   ```bash
   git clone https://github.com/reach-1191/sisyphus.git
   cd sisyphus
   ```

2. **Run automated environment setup**:
   Execute the automated setup script to provision a virtual environment (`.venv`), install pinned dependencies, and initialize the SQLite database:
   ```bash
   bash setup.sh
   ```

3. **Start the application**:
   ```bash
   source .venv/bin/activate
   python app.py
   ```

4. **Open in browser**: Visit [http://localhost:8080](http://localhost:8080).

---

### Option C: Cloud Environments

#### Google Cloud Shell
1. Open terminal in Cloud Shell and run:
   ```bash
   git clone https://github.com/reach-1191/sisyphus.git
   cd sisyphus
   bash setup.sh
   source .venv/bin/activate
   python app.py
   ```
2. Click the **Web Preview** icon in the top-right toolbar.
3. Select **Preview on port 8080**.

#### GitHub Codespaces
1. Create a Codespace on `main` branch.
2. In the terminal, run `bash setup.sh && python app.py`.
3. VS Code will detect port `8080` and prompt you to **Open in Browser**.

---

## Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port the Flask server listens on (cloud container compatible) |
| `HOST` | `0.0.0.0` | Bind address for all network interfaces |
| `DATABASE` | `workout_data.db` | Local SQLite database file auto-migrated on startup |
| `VISION_MODEL` | `pose_landmarker_lite.task` | Bundled MediaPipe Vision Task model (auto-downloaded if absent) |

---

## Supported Exercises & Biomechanical Form Criteria

```
       [Squat]                          [Push-Up]                     [Dumbbell Curl]
  Hip-Knee-Ankle Angle           Shoulder-Elbow-Wrist (Depth)         Shoulder-Elbow-Wrist
  Target Depth: <= 100°          Target Depth: <= 90°                 Contraction: <= 50°
  Valgus (Cave): Inward knees    Plank Spine Align: >= 160°           Extension: >= 155°
                                 Elbow Flare: <= 75°                  Elbow Drift: <= 15°
```

| Exercise | Target Angles | Error Checks | Coaching Cues |
| :--- | :--- | :--- | :--- |
| **Squats** | Knee flexion $\le 100^\circ$ for valid rep; extension $\ge 160^\circ$ | Knee valgus (inward collapse) | *"Squat lower"*, *"Drive knees outward"*, *"Stand all the way up"* |
| **Push-Ups** | Elbow angle $\le 90^\circ$ at bottom; extension $\ge 160^\circ$ | Elbow flare $> 75^\circ$; Back sag $< 160^\circ$ | *"Lower your chest"*, *"Tuck your elbows"*, *"Keep your back straight"* |
| **Dumbbell Curls** | Peak contraction $\le 50^\circ$; Full extension $\ge 155^\circ$ | Upper arm sway / elbow drift $> 15^\circ$ | *"Curl higher"*, *"Lower all the way down"*, *"Keep elbows pinned to sides"* |

---

## Project Structure

```
sisyphus/
├── app.py                      # Core Flask application, pose estimation engine, auth, and REST APIs
├── Dockerfile                  # Containerized deployment with OpenCV and FFmpeg system libraries
├── .dockerignore               # Optimized Docker build context exclusions
├── _config.yml                 # Jekyll configuration for GitHub Pages deployment
├── requirements.txt            # Pinned production dependencies (Flask, MediaPipe, OpenCV, SQLAlchemy)
├── setup.sh                    # Automated environment, virtualenv, and database provisioning script
├── test_app.py                 # 22 automated unit & integration tests validating math, routes, and DB
├── pose_landmarker_lite.task   # MediaPipe Vision Task pose estimation model asset
├── dumbbell_demo.mp4           # Dumbbell Curl headless demo video fallback
├── pushup_demo.mp4             # Push-Up headless demo video fallback
├── squat_demo.mp4              # Squat headless demo video fallback
├── workout_data.db             # Local SQLite database (users, exercise sessions, badges)
├── templates/                  # Jinja2 template views
│   ├── index.html              # Bento athlete dashboard & analytics
│   ├── workout.html            # Real-time workout tracking feed & HUD gauges
│   ├── summary.html            # Post-workout form precision summary & coach insights
│   ├── layout.html             # Base HTML template wrapper
│   ├── login.html              # Dedicated athlete sign-in portal
│   └── register.html           # Athlete registration portal
├── static/                     # Frontend styling and scripts
│   ├── css/
│   │   └── style.css           # Bento design system, animations & responsive layout
│   └── js/
│       └── main.js             # Client-side audio coach, camera handlers & HUD interactions
└── README.md                   # Comprehensive project documentation
```

---

## REST & Streaming API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` or `/dashboard` | `GET` | Authenticated athlete dashboard with stats, streaks, and exercise cards |
| `/workout` | `GET` | Live workout tracker interface with camera HUD and audio coach |
| `/summary/<session_id>` | `GET` | Post-workout precision review, biomechanical feedback, and EXP summary |
| `/video_feed` | `GET` | Multipart MJPEG stream of annotated camera / fallback video feed |
| `/status` | `GET` | Real-time workout metrics JSON (reps, form score, current angles, stage) |
| `/process_frame` | `POST` | Processes a raw JPEG frame from the browser webcam and returns annotated image |
| `/set_exercise` | `POST` | Switches active exercise tracking mode (`squat`, `pushup`, `curl`) |
| `/save_workout` | `POST` | Persists completed workout session, calculates accuracy, and awards EXP / badges |
| `/reset` | `POST` | Resets current repetition counter and workout session state |
| `/pause_feed` | `POST` | Safely releases camera hardware |
| `/resume_feed` | `POST` | Re-initializes camera video stream |
| `/api/audio_alert` | `GET` | Polls queued vocal audio alerts for client-side Web Speech synthesis |
| `/login` | `GET`, `POST` | Athlete authentication |
| `/register` | `GET`, `POST` | New athlete account creation |
| `/logout` | `GET` | Destroys active user session |

---

## Running Automated Tests

A comprehensive suite of 22 unit and integration tests verifies the pose calculation pipeline, EMA smoothing, route protection, camera hardware control, EXP progression, Coach Insights feedback logic, and summary view generation:

```bash
source .venv/bin/activate
python test_app.py
```

Expected output:
```text
Ran 22 tests in 1.929s

OK
```

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.

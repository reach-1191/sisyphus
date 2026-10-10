# Sisyphus AI Coach

> **Real-time AI pose tracking, form correction, and gamified workout dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-black.svg)](https://palletsprojects.com/p/flask/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00A67E.svg)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Sisyphus AI Coach** is an intelligent fitness companion designed to elevate your personal training. Combining state-of-the-art computer vision pose estimation with real-time biometric analysis and client-side vocal coaching, Sisyphus tracks your workout form rep-by-rep, identifies kinematic errors before they cause injury, and turns fitness into an addictive RPG leveling experience.

---

## Key Features

- **Bento Box UI Layout**: Dynamic, responsive dark/light aesthetic dashboard organizing live camera feed, real-time joint angle gauges, rep velocity, workout history, and EXP progress into intuitive modular cards.
- **Dual-Mode Authentication**: User registration and login protected by Werkzeug password hashing and Flask-Login session management, with SQLite database persistence.
- **Multi-Exercise Pose Estimation**:
  - **Squats**: Precise hip-knee-ankle kinematic tracking, depth validation ($\le 100^\circ$), and knee cave (valgus) detection.
  - **Push-Ups**: Chest depth measurement ($\le 90^\circ$), elbow flare detection ($> 75^\circ$), and real-time **back-sag detection** ensuring strict spinal plank alignment ($\ge 160^\circ$).
  - **Dumbbell Curls**: Bilateral arm tracking, peak contraction angles ($\le 50^\circ$), full extension validation ($\ge 155^\circ$), and excessive elbow sway/drift warnings.
- **Client-Side Web Speech API Voice Coaching**: Zero-latency in-browser vocal corrections with a smart **4-second debounced error throttling** algorithm, providing audible coaching tips only when form faults occur without nagging.
- **Persistent EXP, Leveling, and Badge Unlocks**: Earn EXP for every completed rep (with bonus multipliers for high-fidelity form scores). Level up dynamically ($\lfloor\sqrt{\text{EXP}/50}\rfloor + 1$) and unlock achievement badges automatically (e.g., *First Rep*, *Form Master*, *Iron Core*, *Century Club*).
- **Post-Workout Completion Summary View (`/summary/<session_id>`)**: Comprehensive analytics screen displaying total reps completed, average form score, exercise duration, calorie burn estimates, EXP earned, and new achievements unlocked.
- **Headless Video Fallback Support**: Seamlessly falls back to pre-packaged high-definition demo video streams if no local webcam is detected—enabling zero-configuration testing on headless Linux virtual machines and cloud environments.

---

## Quick Start Guide

Follow these steps to clone, configure, and launch the project on your **Local Machine**, **Google Cloud Shell**, or **GitHub Codespaces**.

### 1. Clone the Repository
```bash
git clone https://github.com/reach-1191/sisyphus.git
cd sisyphus
```

### 2. Run Automated Setup
Execute the automated setup script to provision a virtual environment (`.venv`), install exact stable dependencies, and initialize the SQLite database:
```bash
bash setup.sh
```

### 3. Launch the Application
Activate the virtual environment and start the Flask server:
```bash
source .venv/bin/activate
python app.py
```

### 4. Open in Your Browser
Open [http://localhost:8080](http://localhost:8080) in your web browser.

---

## Environment & Cloud Deployment Details

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| **Port** | `8080` | Configurable via the `PORT` environment variable |
| **Host** | `0.0.0.0` | Listens on all interfaces for cloud container compatibility |
| **Database** | `workout_data.db` | Local SQLite database file auto-migrated at startup |
| **Vision Model** | `pose_landmarker_lite.task` | Bundled MediaPipe Vision Task model (auto-downloaded if absent) |

### Google Cloud Shell
1. Follow steps 1–3 above in Cloud Shell terminal.
2. Click the **Web Preview** button in the top-right toolbar.
3. Select **Preview on port 8080** to view the app in a new browser tab.

### GitHub Codespaces
1. Open the repository in GitHub Codespaces.
2. Open a bash terminal and run `bash setup.sh && python app.py`.
3. VS Code will detect port `8080` and prompt you to **Open in Browser**.

---

## Supported Exercises & Form Criteria

```
       [Squat]                          [Push-Up]                     [Dumbbell Curl]
  Hip-Knee-Ankle Angle           Shoulder-Elbow-Wrist (Depth)         Shoulder-Elbow-Wrist
  Target Depth: <= 100°          Target Depth: <= 90°                 Contraction: <= 50°
  Valgus (Cave): Inward knees    Plank Spine Align: >= 160°           Extension: >= 155°
                                 Elbow Flare: <= 75°                  Elbow Drift: <= 15°
```

- **Squats**: Evaluates knee angle using an Exponential Moving Average (EMA) filter. Detects full depth vs. partial squats, and flags inward knee movement (valgus collapse).
- **Push-Ups**: Assesses arm flexion for depth, monitors elbow flare to protect shoulder joints, and checks hip-shoulder-ankle line for hip sag or pike.
- **Dumbbell Curls**: Evaluates left/right arm curl depth, enforces full extension at the bottom to ensure full range of motion, and detects body momentum/elbow drift.

---

## Project Structure

```
sisyphus/
├── app.py                      # Core Flask application, pose estimation engine, and routing
├── requirements.txt            # Pinned, tested production dependencies
├── setup.sh                    # Automated environment and database setup script
├── test_app.py                 # Comprehensive unit & integration test suite
├── pose_landmarker_lite.task   # MediaPipe pose estimation model asset
├── pushup_demo.mp4             # Push-up headless demo fallback video
├── squat_demo.mp4              # Squat headless demo fallback video
├── templates/                  # Jinja2 template views (summary, auth, dashboard)
├── static/                     # Static styling, sound assets, and client-side scripts
└── README.md                   # Project documentation
```

---

## Running Automated Tests

To verify that the pose calculation pipeline, database models, and badge logic pass all unit checks:

```bash
source .venv/bin/activate
python test_app.py
```

# AI Vocal Fitness Coach (Flask + MediaPipe Pose)

A real-time AI-powered exercise form tracker built with **Flask**, **MediaPipe Pose**, and **OpenCV** with an **in-browser AI Vocal Coach** (Web Speech API), supporting both **Squats** and **Push-Ups**.

---

## Key Features

1. **Dual Exercise Support**:
   - **Squats**: Tracks Hip-Knee-Ankle angles (EMA smoothed). Detects deep squat depth ($\le 100^\circ$) vs. shallow depth, and identifies knee cave (valgus) when knees move inward.
   - **Push-Ups**: Tracks Shoulder-Elbow-Wrist angle (chest depth $\le 90^\circ$), Shoulder-Hip-Ankle spine alignment ($\ge 160^\circ$ straight plank), and Shoulder-Hip-Elbow flare angle ($> 75^\circ$ warning).

2. **Real-Time AI Vocal Coach (Web Speech API)**:
   - Uses `window.speechSynthesis` for vocal coaching.
   - Speaks completed repetition counts ("One! Perfect depth", "Two! Keep knees out").
   - 3-second debounce throttle preventing repetitive vocal nagging.
   - Mute / Unmute audio toggle button.

3. **Form Quality Score (0–100%)**:
   - Evaluates execution fidelity per rep with real-time feedback and deductions for form faults.

4. **Single-File Architecture & Google Cloud Shell Support**:
   - Everything contained in [`app.py`](app.py) with embedded modern dark HTML5/CSS3 interface.
   - Bound to `0.0.0.0:8080` for immediate Google Cloud Shell Web Preview.
   - Graceful camera disconnect handling with automatic demo video fallback and browser webcam (`getUserMedia`) support.

---

## Running the Application

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the server:**
   ```bash
   python3 app.py
   ```

3. **Access the Web Dashboard:**
   - **Local**: `http://localhost:8080`
   - **Google Cloud Shell**: Click **Web Preview** -> **Preview on port 8080**

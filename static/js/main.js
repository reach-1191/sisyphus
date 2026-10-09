document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const repsCountEl = document.getElementById('reps-count');
  const stageBadgeEl = document.getElementById('stage-badge');
  const stageTextEl = document.getElementById('stage-text');
  const kneeAngleEl = document.getElementById('knee-angle');
  const activeSideEl = document.getElementById('active-side');
  const depthBarEl = document.getElementById('depth-progress-bar');
  const depthPctTextEl = document.getElementById('depth-pct-text');
  const feedbackMsgEl = document.getElementById('feedback-msg');
  const streamStatusEl = document.getElementById('stream-status');
  const streamPulseEl = document.getElementById('stream-pulse');
  const sourceBadgeEl = document.getElementById('source-badge');
  const cameraPermissionAlert = document.getElementById('camera-permission-alert');
  const cameraSettingsBar = document.getElementById('camera-settings-bar');
  const cameraSelect = document.getElementById('camera-select');
  const mirrorCheckbox = document.getElementById('mirror-checkbox');

  const btnReset = document.getElementById('btn-reset');
  const btnResetTop = document.getElementById('btn-reset-top');
  const btnBrowserCamera = document.getElementById('btn-browser-camera');
  const btnSample = document.getElementById('btn-sample-source');
  const videoUploadInput = document.getElementById('video-upload-input');
  const videoStreamImg = document.getElementById('video-stream');

  const clientWebcamVideo = document.getElementById('client-webcam');
  const captureCanvas = document.getElementById('capture-canvas');
  const captureCtx = captureCanvas.getContext('2d');

  const downThInput = document.getElementById('down-th-input');
  const downThVal = document.getElementById('down-th-val');
  const downThDisp = document.getElementById('down-th-disp');

  const upThInput = document.getElementById('up-th-input');
  const upThVal = document.getElementById('up-th-val');
  const upThDisp = document.getElementById('up-th-disp');
  const btnApplyThresholds = document.getElementById('btn-apply-thresholds');

  let activeMode = 'sample'; // 'sample', 'browser_camera', or 'custom'
  let webcamStream = null;
  let isCapturing = false;
  let lastRep = 0;
  let statusPollTimer = null;
  let currentBlobUrl = null;

  // Update UI telemetry cards
  function updateUI(data) {
    if (!data) return;

    // Reps counter with animation on increase
    if (data.reps !== lastRep) {
      lastRep = data.reps;
      repsCountEl.textContent = data.reps;
      repsCountEl.style.transform = 'scale(1.25)';
      setTimeout(() => {
        repsCountEl.style.transform = 'scale(1)';
      }, 200);
    } else {
      repsCountEl.textContent = data.reps;
    }

    // Stage
    const isDown = data.stage === 'down';
    stageBadgeEl.textContent = isDown ? 'DOWN' : 'UP';
    stageBadgeEl.className = isDown ? 'stage-tag down' : 'stage-tag';
    stageTextEl.textContent = isDown ? 'Squatting' : 'Standing';
    stageTextEl.style.color = isDown ? '#10b981' : '#f1f5f9';

    // Angle & Active Side
    kneeAngleEl.innerHTML = `${Math.round(data.angle)}&deg;`;
    const sideName = data.active_side === 'right' ? 'Right Leg' : 'Left Leg';
    activeSideEl.textContent = `${sideName} (${Math.round(data.angle)}°)`;

    // Depth Progress
    const pct = Math.max(0, Math.min(100, Math.round(data.depth_pct)));
    depthBarEl.style.width = `${pct}%`;
    depthPctTextEl.textContent = `${pct}%`;

    // Feedback
    feedbackMsgEl.textContent = data.feedback || 'Tracking pose...';

    // Source Tag
    if (activeMode === 'browser_camera') {
      sourceBadgeEl.textContent = 'Source: Browser Camera (Live)';
    } else if (data.source) {
      sourceBadgeEl.textContent = `Source: ${data.source.charAt(0).toUpperCase() + data.source.slice(1)} Video`;
    }
  }

  // Poll server status when in demo/server video mode
  async function fetchServerStatus() {
    if (activeMode === 'browser_camera') return;
    try {
      const res = await fetch('/status');
      if (!res.ok) return;
      const data = await res.json();
      updateUI(data);

      // Sync threshold sliders if user isn't currently dragging them
      if (!document.activeElement || document.activeElement.type !== 'range') {
        if (data.down_threshold && Math.abs(parseFloat(downThInput.value) - data.down_threshold) > 0.5) {
          downThInput.value = data.down_threshold;
          downThVal.innerHTML = `${Math.round(data.down_threshold)}&deg;`;
          downThDisp.textContent = Math.round(data.down_threshold);
        }
        if (data.up_threshold && Math.abs(parseFloat(upThInput.value) - data.up_threshold) > 0.5) {
          upThInput.value = data.up_threshold;
          upThVal.innerHTML = `${Math.round(data.up_threshold)}&deg;`;
          upThDisp.textContent = Math.round(data.up_threshold);
        }
      }

      streamStatusEl.textContent = 'Active Tracking';
    } catch (err) {
      streamStatusEl.textContent = 'Reconnecting...';
    }
  }

  statusPollTimer = setInterval(fetchServerStatus, 200);

  // Enumerate user camera devices for dropdown
  async function populateCameraDevices() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');

      cameraSelect.innerHTML = '';
      if (videoDevices.length === 0) {
        cameraSelect.innerHTML = '<option value="">Default Camera</option>';
        return;
      }

      videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
      });
    } catch (e) {
      console.warn('Could not enumerate cameras:', e);
    }
  }

  // Start Browser Camera Access
  async function startBrowserCamera(deviceId = null) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Your browser does not support webcam access via getUserMedia.');
      return false;
    }

    try {
      cameraPermissionAlert.style.display = 'flex';
      streamStatusEl.textContent = 'Requesting Camera...';

      // Stop existing webcam tracks if running
      if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: deviceId ? undefined : 'user',
          deviceId: deviceId ? { exact: deviceId } : undefined
        },
        audio: false
      };

      webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
      clientWebcamVideo.srcObject = webcamStream;
      await clientWebcamVideo.play();

      cameraPermissionAlert.style.display = 'none';
      cameraSettingsBar.style.display = 'flex';

      // Switch active mode
      activeMode = 'browser_camera';
      btnBrowserCamera.classList.add('active');
      btnSample.classList.remove('active');
      sourceBadgeEl.textContent = 'Source: Browser Camera (Live)';
      streamStatusEl.textContent = 'Camera Active';

      // Populate device dropdown labels
      await populateCameraDevices();

      // Notify server to switch mode and reset tracker
      await fetch('/set_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'browser_camera' })
      });

      // Start the frame upload loop
      isCapturing = true;
      runCameraProcessingLoop();
      return true;

    } catch (err) {
      console.error('Webcam access error:', err);
      cameraPermissionAlert.style.display = 'none';
      streamStatusEl.textContent = 'Camera Denied';
      alert('Could not access your computer camera. Please allow camera permissions in your browser: ' + err.message);
      switchToDemoVideo();
      return false;
    }
  }

  // Browser Camera Loop: capture -> post /process_frame -> render processed response
  async function runCameraProcessingLoop() {
    if (!isCapturing || activeMode !== 'browser_camera') return;

    if (clientWebcamVideo.readyState >= 2) {
      const vw = clientWebcamVideo.videoWidth || 640;
      const vh = clientWebcamVideo.videoHeight || 480;

      captureCanvas.width = vw;
      captureCanvas.height = vh;

      captureCtx.save();
      // Optional selfie mirror flip
      if (mirrorCheckbox.checked) {
        captureCtx.translate(vw, 0);
        captureCtx.scale(-1, 1);
      }
      captureCtx.drawImage(clientWebcamVideo, 0, 0, vw, vh);
      captureCtx.restore();

      // Convert canvas frame to JPEG blob
      captureCanvas.toBlob(async (blob) => {
        if (!blob || !isCapturing || activeMode !== 'browser_camera') {
          if (isCapturing) requestAnimationFrame(runCameraProcessingLoop);
          return;
        }

        try {
          const res = await fetch('/process_frame', {
            method: 'POST',
            body: blob,
            headers: { 'Content-Type': 'image/jpeg' }
          });

          if (res.ok) {
            // Read telemetry from X-Metrics header
            const metricsHeader = res.headers.get('X-Metrics');
            if (metricsHeader) {
              try {
                const metrics = JSON.parse(decodeURIComponent(metricsHeader));
                updateUI(metrics);
              } catch (e) {
                console.error('Failed to parse metrics header:', e);
              }
            }

            // Update displayed video stream image with processed annotated frame
            const processedBlob = await res.blob();
            const newUrl = URL.createObjectURL(processedBlob);
            videoStreamImg.src = newUrl;

            if (currentBlobUrl) {
              URL.revokeObjectURL(currentBlobUrl);
            }
            currentBlobUrl = newUrl;
          }
        } catch (e) {
          console.warn('Frame send error:', e);
        }

        // Loop next frame
        if (isCapturing && activeMode === 'browser_camera') {
          requestAnimationFrame(runCameraProcessingLoop);
        }
      }, 'image/jpeg', 0.8);

    } else {
      requestAnimationFrame(runCameraProcessingLoop);
    }
  }

  // Stop Browser Camera and Switch to Demo Video
  async function switchToDemoVideo() {
    isCapturing = false;
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
    }
    clientWebcamVideo.srcObject = null;
    cameraSettingsBar.style.display = 'none';

    if (currentBlobUrl) {
      URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = null;
    }

    activeMode = 'sample';
    btnSample.classList.add('active');
    btnBrowserCamera.classList.remove('active');
    sourceBadgeEl.textContent = 'Source: Demo Video';
    streamStatusEl.textContent = 'Demo Video Active';

    await fetch('/set_source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'sample' }),
    });

    videoStreamImg.src = '/video_feed?t=' + Date.now();
  }

  // Event Listeners
  btnBrowserCamera.addEventListener('click', () => {
    startBrowserCamera(cameraSelect.value || null);
  });

  btnSample.addEventListener('click', () => {
    switchToDemoVideo();
  });

  cameraSelect.addEventListener('change', () => {
    if (activeMode === 'browser_camera') {
      startBrowserCamera(cameraSelect.value);
    }
  });

  // Reset Reps
  async function resetReps() {
    try {
      await fetch('/reset', { method: 'POST' });
      lastRep = 0;
      repsCountEl.textContent = '0';
      feedbackMsgEl.textContent = 'Counter reset to 0.';
    } catch (e) {
      console.error(e);
    }
  }

  btnReset.addEventListener('click', resetReps);
  btnResetTop.addEventListener('click', resetReps);

  // Video File Upload
  videoUploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Stop browser webcam if active
    if (webcamStream) {
      webcamStream.getTracks().forEach(t => t.stop());
      webcamStream = null;
    }
    isCapturing = false;
    cameraSettingsBar.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    feedbackMsgEl.textContent = `Uploading ${file.name}...`;

    try {
      const res = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        activeMode = 'custom';
        btnSample.classList.remove('active');
        btnBrowserCamera.classList.remove('active');
        sourceBadgeEl.textContent = `Source: Uploaded (${file.name})`;
        videoStreamImg.src = '/video_feed?t=' + Date.now();
        feedbackMsgEl.textContent = `Analyzing uploaded video: ${file.name}`;
      } else {
        alert(data.error || 'Upload failed');
      }
    } catch (err) {
      alert('Failed to upload video');
    }
  });

  // Threshold slider live displays
  downThInput.addEventListener('input', () => {
    downThVal.innerHTML = `${downThInput.value}&deg;`;
    downThDisp.textContent = downThInput.value;
  });

  upThInput.addEventListener('input', () => {
    upThVal.innerHTML = `${upThInput.value}&deg;`;
    upThDisp.textContent = upThInput.value;
  });

  // Save Thresholds
  btnApplyThresholds.addEventListener('click', async () => {
    const down = parseFloat(downThInput.value);
    const up = parseFloat(upThInput.value);

    if (down >= up) {
      alert('Squat depth angle must be less than upright standing angle!');
      return;
    }

    await fetch('/set_thresholds', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ down_threshold: down, up_threshold: up }),
    });

    feedbackMsgEl.textContent = `Thresholds updated: Depth <= ${down}°, Stand >= ${up}°`;
  });
});

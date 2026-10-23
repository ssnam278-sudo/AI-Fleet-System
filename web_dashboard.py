from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import numpy as np
import time
import math
import struct
import wave
import os
import subprocess
import threading
import pygame
from ultralytics import YOLO
import mediapipe as mp

# Resilient & Pylance-clean Face Mesh loader
try:
    mp_face_mesh = mp.solutions.face_mesh  # type: ignore
except AttributeError:
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh  # type: ignore
    except (ImportError, AttributeError):
        import mediapipe.solutions.face_mesh as mp_face_mesh  # type: ignore

# ==============================================================================
# 1. NATIVE PYTHON SPOKEN VOICE & CHIME ENGINE
# ==============================================================================
AUDIO_DIR = "audio_alerts"
os.makedirs(AUDIO_DIR, exist_ok=True)

if not pygame.mixer.get_init():
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

CHIME_CHANNEL = pygame.mixer.Channel(0)
VOICE_CHANNEL = pygame.mixer.Channel(1)

def generate_pcm_wav(filepath, frequencies, durations, volume=0.45):
    """Generates synthetic chime tones if missing."""
    if os.path.exists(filepath):
        return
    sample_rate = 44100
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for freq, duration in zip(frequencies, durations):
            n_samples = int(sample_rate * duration)
            if freq == 0:
                for _ in range(n_samples):
                    wav_file.writeframesraw(struct.pack('<h', 0))
            else:
                for i in range(n_samples):
                    t = float(i) / sample_rate
                    amplitude = int(32767.0 * volume * math.sin(2.0 * math.pi * freq * t))
                    wav_file.writeframesraw(struct.pack('<h', amplitude))

def generate_windows_voice_wav(filepath, speech_text):
    """Synthesizes human speech into local WAV files via Windows Speech API."""
    if os.path.exists(filepath):
        return
    abs_path = os.path.abspath(filepath).replace("'", "''")
    safe_text = speech_text.replace("'", "''")
    ps_cmd = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 1; "
        f"$s.SetOutputToWaveFile('{abs_path}'); "
        f"$s.Speak('{safe_text}'); "
        f"$s.Dispose();"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                       capture_output=True, timeout=12)
    except Exception as e:
        print(f"[Speech Generation Warning] Failed to render {filepath}: {e}")

FCW_CHIME = os.path.join(AUDIO_DIR, "fcw_chime.wav")
generate_pcm_wav(FCW_CHIME, [1350.0, 0, 1350.0], [0.08, 0.04, 0.09], volume=0.55)

VOICE_FILES = {
    "FORWARD COLLISION WARNING: BRAKE!": (
        os.path.join(AUDIO_DIR, "voice_fcw.wav"),
        "Emergency! Collision hazard ahead! Brake immediately!"
    ),
    "SCHOOL ZONE AHEAD: SLOW DOWN": (
        os.path.join(AUDIO_DIR, "voice_school.wav"),
        "Caution! School zone ahead! Please slow down!"
    ),
    "DISTRACTION: HEAD TURNED": (
        os.path.join(AUDIO_DIR, "voice_distract.wav"),
        "Warning! Face facing other direction! Keep your eyes on the road!"
    ),
    "DROWSINESS: EYES CLOSED": (
        os.path.join(AUDIO_DIR, "voice_drowsy.wav"),
        "Alert! Eyes closed detected! Wake up and stay alert!"
    ),
    "PHONE USE IN CAB": (
        os.path.join(AUDIO_DIR, "voice_phone.wav"),
        "Warning! Mobile phone detected! Put your phone down!"
    ),
    "YAWNING DETECTED": (
        os.path.join(AUDIO_DIR, "voice_yawn.wav"),
        "Fatigue advisory! Yawning detected, please take a break!"
    )
}

print("[System] Initializing voice synthesizer and audio assets...")
for alert_key, (fpath, phrase) in VOICE_FILES.items():
    generate_windows_voice_wav(fpath, phrase)

voice_sounds = {}
for alert_key, (fpath, _) in VOICE_FILES.items():
    if os.path.exists(fpath):
        voice_sounds[alert_key] = pygame.mixer.Sound(fpath)

fcw_chime_sound = pygame.mixer.Sound(FCW_CHIME) if os.path.exists(FCW_CHIME) else None

class NativeAudioAlertManager:
    def __init__(self):
        self.cooldowns = {
            "FORWARD COLLISION WARNING: BRAKE!": 2.0,
            "SCHOOL ZONE AHEAD: SLOW DOWN": 6.0,
            "DISTRACTION: HEAD TURNED": 3.8,
            "DROWSINESS: EYES CLOSED": 3.2,
            "PHONE USE IN CAB": 3.8,
            "YAWNING DETECTED": 5.0,
            "DEFAULT": 3.0
        }
        self.last_played = {}

    def trigger(self, alert_name):
        now = time.time()
        cooldown = self.cooldowns.get(alert_name, self.cooldowns["DEFAULT"])
        last_time = self.last_played.get(alert_name, 0.0)

        if now - last_time < cooldown:
            return

        sound = voice_sounds.get(alert_name)
        if not sound:
            return

        is_emergency = (alert_name == "FORWARD COLLISION WARNING: BRAKE!")

        if is_emergency:
            VOICE_CHANNEL.stop()
            if fcw_chime_sound:
                CHIME_CHANNEL.play(fcw_chime_sound)
            VOICE_CHANNEL.play(sound)
            self.last_played[alert_name] = now
        else:
            if not VOICE_CHANNEL.get_busy():
                VOICE_CHANNEL.play(sound)
                self.last_played[alert_name] = now

audio_engine = NativeAudioAlertManager()

# ==============================================================================
# 2. AI ENGINES (Driver & Road Monitoring with School Zone Vision)
# ==============================================================================
class DriverMonitor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EAR_THRESH = 0.20
        self.EAR_FRAMES = 22
        self.MAR_THRESH = 0.58
        self.MAR_FRAMES = 18
        self.YAW_RATIO_THRESH = 0.28
        self.YAW_FRAMES = 20

        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0

    @staticmethod
    def _pt(landmarks, idx, w, h):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    def calculate_ear(self, landmarks, w, h):
        l_h = np.linalg.norm(self._pt(landmarks, 33, w, h) - self._pt(landmarks, 133, w, h))
        l_v = np.linalg.norm(self._pt(landmarks, 160, w, h) - self._pt(landmarks, 144, w, h)) + \
              np.linalg.norm(self._pt(landmarks, 158, w, h) - self._pt(landmarks, 153, w, h))
        left_ear = l_v / (2.0 * l_h + 1e-6)

        r_h = np.linalg.norm(self._pt(landmarks, 362, w, h) - self._pt(landmarks, 263, w, h))
        r_v = np.linalg.norm(self._pt(landmarks, 385, w, h) - self._pt(landmarks, 380, w, h)) + \
              np.linalg.norm(self._pt(landmarks, 387, w, h) - self._pt(landmarks, 373, w, h))
        right_ear = r_v / (2.0 * r_h + 1e-6)
        return (left_ear + right_ear) / 2.0

    def calculate_mar(self, landmarks, w, h):
        top_lip = self._pt(landmarks, 13, w, h)
        bot_lip = self._pt(landmarks, 14, w, h)
        c_left = self._pt(landmarks, 78, w, h)
        c_right = self._pt(landmarks, 308, w, h)
        return np.linalg.norm(top_lip - bot_lip) / (np.linalg.norm(c_left - c_right) + 1e-6)

    def process(self, frame, rgb_frame, yolo_detections):
        h, w, _ = frame.shape
        alerts = []
        metrics = {"EAR": 0.0, "MAR": 0.0}

        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. EAR (Eyes Closed / Drowsiness)
            ear = self.calculate_ear(landmarks, w, h)
            metrics["EAR"] = float(ear)
            if ear < self.EAR_THRESH:
                self.drowsy_counter += 1
                if self.drowsy_counter >= self.EAR_FRAMES:
                    alerts.append("DROWSINESS: EYES CLOSED")
            else:
                self.drowsy_counter = max(0, self.drowsy_counter - 1)

            # 2. MAR (Yawning)
            mar = self.calculate_mar(landmarks, w, h)
            metrics["MAR"] = float(mar)
            if mar > self.MAR_THRESH:
                self.yawn_counter += 1
                if self.yawn_counter >= self.MAR_FRAMES:
                    alerts.append("YAWNING DETECTED")
            else:
                self.yawn_counter = max(0, self.yawn_counter - 1)

            # 3. Head Turned / Facing Other Direction
            nose = self._pt(landmarks, 1, w, h)
            l_edge = self._pt(landmarks, 234, w, h)
            r_edge = self._pt(landmarks, 454, w, h)
            face_width = np.linalg.norm(l_edge - r_edge)
            mid_x = (l_edge[0] + r_edge[0]) / 2.0

            if abs(nose[0] - mid_x) > self.YAW_RATIO_THRESH * face_width:
                self.distraction_counter += 1
                if self.distraction_counter >= self.YAW_FRAMES:
                    alerts.append("DISTRACTION: HEAD TURNED")
            else:
                self.distraction_counter = max(0, self.distraction_counter - 1)

        # 4. Cell Phone Detection
        for label, conf, (x1, y1, x2, y2) in yolo_detections:
            if label == "cell phone" and conf > 0.45:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Cell Phone {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                alerts.append("PHONE USE IN CAB")

        return alerts, metrics


class RoadMonitor:
    def __init__(self):
        self.TARGET_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
        self.COLLISION_EXPANSION_RATIO = 0.38

    def detect_school_zone_sign(self, frame):
        """
        Detects standard yellow/amber or blue school zone signboards via
        HSV color segmentation and polygon contour geometry analysis.
        """
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected = False

        # 1. Fluorescent Yellow/Amber Sign Range (Global standard school diamond/pentagon)
        lower_yellow = np.array([16, 110, 110])
        upper_yellow = np.array([36, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. Blue Pedestrian/School Sign Range (Standard school walking sign)
        lower_blue = np.array([98, 120, 80])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        mask_signs = cv2.bitwise_or(mask_yellow, mask_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_signs = cv2.morphologyEx(mask_signs, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_signs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Size filter: Must be prominent in view (not tiny distant noise or entire background)
            if 1800 < area < (h * w * 0.18):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / float(bh)

                # Aspect ratio filter: Signboards are roughly square, diamond, or pentagon (0.65 to 1.45)
                if 0.65 <= aspect_ratio <= 1.45:
                    # Draw School Zone Sign Visual Indicator
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 215, 255), 3)
                    cv2.putText(frame, "SCHOOL ZONE SIGN", (bx, max(20, by - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
                    detected = True
                    break

        return detected

    def process(self, frame, yolo_detections):
        h, w, _ = frame.shape
        alerts = []
        center_lane_min = w * 0.28
        center_lane_max = w * 0.72

        # 1. Forward Traffic & Vulnerable Road User Detection
        for label, conf, (x1, y1, x2, y2) in yolo_detections:
            if label in self.TARGET_CLASSES and conf > 0.40:
                color = (0, 255, 0) if label != "person" else (255, 128, 0)
                box_h = y2 - y1
                box_cx = (x1 + x2) / 2.0

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Monocular Forward Collision Warning
                if center_lane_min < box_cx < center_lane_max:
                    relative_height = box_h / float(h)
                    if relative_height >= self.COLLISION_EXPANSION_RATIO:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        alerts.append("FORWARD COLLISION WARNING: BRAKE!")

        # 2. School Zone Sign Detection
        if self.detect_school_zone_sign(frame):
            alerts.append("SCHOOL ZONE AHEAD: SLOW DOWN")

        return alerts

# ==============================================================================
# 3. VIDEO CAPTURE & APP STATE
# ==============================================================================
os.makedirs("uploads", exist_ok=True)
app = Flask(__name__)

shared_state = {
    "mode": "driver",
    "source_type": "webcam",
    "file_name": "Webcam (Live)",
    "ear": 0.0,
    "mar": 0.0,
    "active_alerts": [],
    "recent_logs": []
}

yolo_model = YOLO("yolov8n.pt")
driver_engine = DriverMonitor()
road_engine = RoadMonitor()

camera_lock = threading.Lock()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    camera = cv2.VideoCapture(0)

def set_camera_source(src_type, file_path=None):
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
        if src_type == "webcam":
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera = cv2.VideoCapture(0)
            shared_state["source_type"] = "webcam"
            shared_state["file_name"] = "Webcam (Live)"
        elif src_type == "file" and file_path:
            camera = cv2.VideoCapture(file_path)
            shared_state["source_type"] = "file"
            shared_state["file_name"] = os.path.basename(file_path)

def generate_video_stream():
    global camera
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                time.sleep(0.05)
                continue

            success, frame = camera.read()
            if not success:
                if shared_state["source_type"] == "file":
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue
                else:
                    time.sleep(0.04)
                    continue

        if shared_state["source_type"] == "webcam":
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yolo_res = yolo_model(frame, verbose=False)[0]
        detections = []
        for box in yolo_res.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            lbl = yolo_model.names[cid]
            coords = map(int, box.xyxy[0])
            detections.append((lbl, conf, tuple(coords)))

        active_alerts = []
        metrics = {"EAR": 0.0, "MAR": 0.0}

        if shared_state["mode"] == "driver":
            alerts, metrics = driver_engine.process(frame, rgb_frame, detections)
            active_alerts.extend(alerts)
            shared_state["ear"] = round(metrics.get("EAR", 0.0), 2)
            shared_state["mar"] = round(metrics.get("MAR", 0.0), 2)
        else:
            alerts = road_engine.process(frame, detections)
            active_alerts.extend(alerts)

        shared_state["active_alerts"] = list(set(active_alerts))

        # Direct Python Hardware Voice Alert Output
        for alert_msg in set(active_alerts):
            audio_engine.trigger(alert_msg)

        # Incident Logging
        timestamp = time.strftime("%H:%M:%S")
        for alert_msg in set(active_alerts):
            log_item = f"[{timestamp}] {alert_msg}"
            if not shared_state["recent_logs"] or shared_state["recent_logs"][-1] != log_item:
                shared_state["recent_logs"].append(log_item)
                if len(shared_state["recent_logs"]) > 10:
                    shared_state["recent_logs"].pop(0)

        # Top Banner Overlay
        if active_alerts:
            banner_color = (0, 0, 220) if "COLLISION" in active_alerts[0] else (0, 140, 255)
            cv2.rectangle(frame, (0, 0), (w, 45), banner_color, -1)
            cv2.putText(frame, f"ALERT: {' | '.join(set(active_alerts))}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if shared_state["source_type"] == "file":
            time.sleep(0.015)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==============================================================================
# 4. FRONTEND HTML & API ROUTES
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Fleet Safety Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    header { background: #1e293b; padding: 14px 24px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
    header h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.5px; }
    .header-badges { display: flex; gap: 10px; align-items: center; }
    .badge { background: #16a34a; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .badge-src { background: #3b82f6; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    
    .container { display: flex; flex: 1; padding: 20px; gap: 20px; overflow: hidden; }
    
    .viewport { flex: 1; background: #020617; border-radius: 12px; border: 1px solid #334155; display: flex; flex-direction: column; overflow: hidden; }
    .viewport img { width: 100%; height: 100%; object-fit: contain; }
    
    .sidebar { width: 360px; display: flex; flex-direction: column; gap: 14px; }
    .card { background: #1e293b; border-radius: 10px; border: 1px solid #334155; padding: 15px; }
    .card h3 { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 10px; font-weight: 700; }
    
    .btn-mode { width: 100%; padding: 12px; border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 700; cursor: pointer; transition: 0.2s ease; }
    .mode-driver { background: #16a34a; color: #fff; }
    .mode-driver:hover { background: #15803d; }
    .mode-road { background: #0284c7; color: #fff; }
    .mode-road:hover { background: #0369a1; }

    .src-btn-group { display: flex; gap: 8px; margin-bottom: 12px; }
    .btn-src { flex: 1; padding: 10px; border: 1px solid #475569; background: #334155; color: #f8fafc; font-size: 0.8rem; font-weight: 600; border-radius: 6px; cursor: pointer; }
    .btn-src:hover { background: #475569; }
    .btn-src.active { background: #3b82f6; border-color: #60a5fa; }
    #video-file-input { display: none; }
    
    .voice-badge-box { background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }
    .voice-status { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #4ade80; font-weight: 600; }
    .btn-test { margin-top: 8px; width: 100%; padding: 8px; background: #4f46e5; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: 600; }
    .btn-test:hover { background: #4338ca; }
    
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .metric-box { background: #0f172a; border-radius: 8px; padding: 10px; border: 1px solid #334155; }
    .metric-title { font-size: 0.72rem; color: #64748b; font-weight: 600; }
    .metric-value { font-size: 1.4rem; font-weight: 800; color: #38bdf8; margin-top: 3px; font-family: monospace; }
    
    .log-box { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    .log-list { flex: 1; background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 10px; font-family: monospace; font-size: 0.78rem; overflow-y: auto; color: #f87171; }
    .log-entry { margin-bottom: 5px; }
  </style>
</head>
<body>
  <header>
    <h1>🛡️ AI FLEET SAFETY MONITORING DASHBOARD</h1>
    <div class="header-badges">
      <span class="badge-src" id="badge-source">FEED: WEBCAM</span>
      <span class="badge" id="badge-status">RUNNING</span>
    </div>
  </header>

  <div class="container">
    <div class="viewport">
      <img src="/video_feed" id="video-stream" alt="Live Video Feed">
    </div>

    <div class="sidebar">
      <!-- Mode Toggle -->
      <div class="card">
        <h3>Detection Mode</h3>
        <button id="mode-btn" class="btn-mode mode-driver" onclick="toggleMode()">
          🟢 DRIVER MONITORING ACTIVE
        </button>
      </div>

      <!-- Feed Source & Audio Settings -->
      <div class="card">
        <h3>Input Source & Voice Alert</h3>
        <div class="src-btn-group">
          <button class="btn-src active" id="btn-src-webcam" onclick="useWebcam()">📹 Webcam</button>
          <button class="btn-src" id="btn-src-file" onclick="triggerFileInput()">📁 Upload Video</button>
        </div>
        <input type="file" id="video-file-input" accept="video/mp4,video/avi,video/mov,video/mkv" onchange="uploadVideo(this)">
        
        <div class="voice-badge-box">
          <div class="voice-status">
            <span>🔊</span>
            <span>Hardware Voice Engine: Active</span>
          </div>
          <button class="btn-test" onclick="testSpeaker()">🔊 Test School Zone Voice</button>
        </div>
      </div>

      <!-- Telemetry Readings -->
      <div class="card">
        <h3>Live Telemetry</h3>
        <div class="metrics">
          <div class="metric-box">
            <div class="metric-title">EYE RATIO (EAR)</div>
            <div class="metric-value" id="val-ear">0.00</div>
          </div>
          <div class="metric-box">
            <div class="metric-title">MOUTH RATIO (MAR)</div>
            <div class="metric-value" id="val-mar">0.00</div>
          </div>
        </div>
      </div>

      <!-- Incident Event Table -->
      <div class="card log-box">
        <h3>Incident Log (Time Only)</h3>
        <div class="log-list" id="log-list">
          <div class="log-entry" style="color: #64748b;">Waiting for safety events...</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function testSpeaker() {
      await fetch('/api/test_school_voice');
    }

    async function toggleMode() {
      const res = await fetch('/api/toggle_mode');
      const data = await res.json();
      updateUI(data);
    }

    async function useWebcam() {
      document.getElementById('badge-source').innerText = "SWITCHING...";
      const res = await fetch('/api/use_webcam');
      const data = await res.json();
      document.getElementById('btn-src-webcam').classList.add('active');
      document.getElementById('btn-src-file').classList.remove('active');
      updateUI(data);
    }

    function triggerFileInput() {
      document.getElementById('video-file-input').click();
    }

    async function uploadVideo(input) {
      if (!input.files || input.files.length === 0) return;
      const file = input.files[0];
      const formData = new FormData();
      formData.append('video', file);

      document.getElementById('badge-source').innerText = "UPLOADING...";

      const res = await fetch('/api/upload_video', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      document.getElementById('btn-src-file').classList.add('active');
      document.getElementById('btn-src-webcam').classList.remove('active');
      updateUI(data);
    }

    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        updateUI(data);
      } catch (err) {
        console.error("Telemetry fetch error:", err);
      }
    }

    function updateUI(data) {
      document.getElementById('badge-source').innerText = "FEED: " + data.file_name.toUpperCase();

      const btn = document.getElementById('mode-btn');
      if (data.mode === 'driver') {
        btn.className = 'btn-mode mode-driver';
        btn.innerText = '🟢 DRIVER MONITORING ACTIVE';
        document.getElementById('val-ear').innerText = data.ear.toFixed(2);
        document.getElementById('val-mar').innerText = data.mar.toFixed(2);
      } else {
        btn.className = 'btn-mode mode-road';
        btn.innerText = '🔵 ROAD MONITORING ACTIVE';
        document.getElementById('val-ear').innerText = '--';
        document.getElementById('val-mar').innerText = '--';
      }

      const logBox = document.getElementById('log-list');
      if (data.logs.length > 0) {
        logBox.innerHTML = data.logs.map(log => `<div class="log-entry">${log}</div>`).join('');
        logBox.scrollTop = logBox.scrollHeight;
      }
    }

    setInterval(fetchTelemetry, 350);
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_mode')
def toggle_mode():
    shared_state["mode"] = "road" if shared_state["mode"] == "driver" else "driver"
    return jsonify({
        "mode": shared_state["mode"],
        "source_type": shared_state["source_type"],
        "file_name": shared_state["file_name"],
        "ear": shared_state["ear"],
        "mar": shared_state["mar"],
        "logs": shared_state["recent_logs"]
    })

@app.route('/api/test_school_voice')
def test_school_voice():
    sound = voice_sounds.get("SCHOOL ZONE AHEAD: SLOW DOWN")
    if sound:
        VOICE_CHANNEL.stop()
        VOICE_CHANNEL.play(sound)
    return jsonify({"status": "played"})

@app.route('/api/use_webcam')
def use_webcam():
    set_camera_source("webcam")
    return jsonify({
        "mode": shared_state["mode"],
        "source_type": shared_state["source_type"],
        "file_name": shared_state["file_name"],
        "ear": shared_state["ear"],
        "mar": shared_state["mar"],
        "logs": shared_state["recent_logs"]
    })

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)
    set_camera_source("file", save_path)

    return jsonify({
        "mode": shared_state["mode"],
        "source_type": shared_state["source_type"],
        "file_name": shared_state["file_name"],
        "ear": shared_state["ear"],
        "mar": shared_state["mar"],
        "logs": shared_state["recent_logs"]
    })

@app.route('/api/telemetry')
def telemetry():
    return jsonify({
        "mode": shared_state["mode"],
        "source_type": shared_state["source_type"],
        "file_name": shared_state["file_name"],
        "ear": shared_state["ear"],
        "mar": shared_state["mar"],
        "alerts": shared_state["active_alerts"],
        "logs": shared_state["recent_logs"]
    })

if __name__ == '__main__':
    print("\n=======================================================")
    print(" AI FLEET SAFETY WEB DASHBOARD LAUNCHED")
    print(" Road Mode Feature: School Zone Detection & Voice Advisory Active")
    print(" Open in Browser: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
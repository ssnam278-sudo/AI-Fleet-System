import cv2
import numpy as np
import time
import math
import struct
import wave
import os
import argparse
import pygame
from ultralytics import YOLO
import mediapipe as mp

# ==============================================================================
# 1. AUDIO ALERT ENGINE (Zero External Asset Dependency)
# ==============================================================================
ALERT_WAV_FILE = "alert_chime.wav"

def generate_alert_tone(filepath=ALERT_WAV_FILE):
    """Creates a high-frequency sine wave chime if not present on disk."""
    if not os.path.exists(filepath):
        sample_rate = 44100
        duration = 0.22  # seconds
        freq = 1100.0    # 1.1 kHz warning beep
        n_samples = int(sample_rate * duration)
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)     # 16-bit PCM
            wav_file.setframerate(sample_rate)
            for i in range(n_samples):
                t = float(i) / sample_rate
                amplitude = int(32767.0 * 0.45 * math.sin(2.0 * math.pi * freq * t))
                wav_file.writeframesraw(struct.pack('<h', amplitude))

generate_alert_tone()
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(ALERT_WAV_FILE)


class AlertSystem:
    def __init__(self, cooldown=1.2):
        self.cooldown = cooldown
        self.last_alert_time = 0.0

    def trigger(self, alert_name, severity="CRITICAL"):
        curr = time.time()
        timestamp = time.strftime("%H:%M:%S")
        if curr - self.last_alert_time > self.cooldown:
            alert_sound.play()
            self.last_alert_time = curr
            print(f"[{timestamp}] [ALERT] [{severity}] {alert_name}")


# ==============================================================================
# 2. DRIVER MONITORING MODULE (Drowsiness, Yawn, Distraction, Phone)
# ==============================================================================
class DriverMonitor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Thresholds
        self.EAR_THRESH = 0.20
        self.EAR_FRAMES = 24       # ~1 second consecutive eye closure
        self.MAR_THRESH = 0.58
        self.MAR_FRAMES = 18       # ~0.7 seconds mouth open
        self.YAW_RATIO_THRESH = 0.28
        self.YAW_FRAMES = 20

        # State trackers
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0

    @staticmethod
    def _pt(landmarks, idx, w, h):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    def calculate_ear(self, landmarks, w, h):
        # Left Eye (33, 133 corners; 160, 144, 158, 153 verticals)
        l_h = np.linalg.norm(self._pt(landmarks, 33, w, h) - self._pt(landmarks, 133, w, h))
        l_v = np.linalg.norm(self._pt(landmarks, 160, w, h) - self._pt(landmarks, 144, w, h)) + \
              np.linalg.norm(self._pt(landmarks, 158, w, h) - self._pt(landmarks, 153, w, h))
        left_ear = l_v / (2.0 * l_h + 1e-6)

        # Right Eye (362, 263 corners; 385, 380, 387, 373 verticals)
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

            # 1. EAR (Drowsiness)
            ear = self.calculate_ear(landmarks, w, h)
            metrics["EAR"] = ear
            if ear < self.EAR_THRESH:
                self.drowsy_counter += 1
                if self.drowsy_counter >= self.EAR_FRAMES:
                    alerts.append("DROWSINESS: EYES CLOSED")
            else:
                self.drowsy_counter = max(0, self.drowsy_counter - 1)

            # 2. MAR (Yawning)
            mar = self.calculate_mar(landmarks, w, h)
            metrics["MAR"] = mar
            if mar > self.MAR_THRESH:
                self.yawn_counter += 1
                if self.yawn_counter >= self.MAR_FRAMES:
                    alerts.append("YAWNING DETECTED")
            else:
                self.yawn_counter = max(0, self.yawn_counter - 1)

            # 3. Distraction (Head Yaw)
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

        # 4. Phone Use Detection
        for label, conf, (x1, y1, x2, y2) in yolo_detections:
            if label == "cell phone" and conf > 0.45:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Cell Phone {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                alerts.append("PHONE USE IN CAB")

        return alerts, metrics


# ==============================================================================
# 3. ROAD MONITORING MODULE (Vehicles, Pedestrians, Forward Collision Warning)
# ==============================================================================
class RoadMonitor:
    def __init__(self):
        self.TARGET_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}
        self.COLLISION_EXPANSION_RATIO = 0.38  # Box height / frame height

    def process(self, frame, yolo_detections):
        h, w, _ = frame.shape
        alerts = []
        center_lane_min = w * 0.28
        center_lane_max = w * 0.72

        for label, conf, (x1, y1, x2, y2) in yolo_detections:
            if label in self.TARGET_CLASSES and conf > 0.40:
                color = (0, 255, 0) if label != "person" else (255, 128, 0)
                box_h = y2 - y1
                box_cx = (x1 + x2) / 2.0

                # Render detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Forward Collision Warning (FCW):
                # Target is in driver's central lane and proximate
                if center_lane_min < box_cx < center_lane_max:
                    relative_height = box_h / float(h)
                    if relative_height >= self.COLLISION_EXPANSION_RATIO:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        alerts.append("FORWARD COLLISION WARNING: BRAKE!")

        return alerts


# ==============================================================================
# 4. MAIN PIPELINE INTEGRATION
# ==============================================================================
def run_fleet_safety(source, mode="driver"):
    # Initialize Core Engines
    yolo_model = YOLO("yolov8n.pt")
    driver_engine = DriverMonitor()
    road_engine = RoadMonitor()
    alert_engine = AlertSystem()

    # Video Source Setup (0 for Webcam, or string path)
    cap_src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        print(f"Error: Unable to open input stream: {source}")
        return

    print(f"\n=======================================================")
    print(f" AI FLEET SAFETY PLATFORM INITIATED")
    print(f" MODE: {mode.upper()} | SOURCE: {source}")
    print(f" Press 'q' to quit.")
    print(f"=======================================================\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run shared YOLOv8 inference once per frame
        yolo_res = yolo_model(frame, verbose=False)[0]
        detections = []
        for box in yolo_res.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            lbl = yolo_model.names[cid]
            coords = map(int, box.xyxy[0])
            detections.append((lbl, conf, tuple(coords)))

        active_alerts = []
        metrics = {}

        # Branch processing by chosen mode
        if mode == "driver":
            alerts, metrics = driver_engine.process(frame, rgb_frame, detections)
            active_alerts.extend(alerts)
        elif mode == "road":
            alerts = road_engine.process(frame, detections)
            active_alerts.extend(alerts)

        # Trigger Audio & Print Events
        for alert_msg in set(active_alerts):
            alert_engine.trigger(alert_msg)

        # HUD Overlays
        timestamp = time.strftime("%H:%M:%S")
        if active_alerts:
            # Red Critical Top Bar
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), -1)
            banner = " | ".join(set(active_alerts))
            cv2.putText(frame, f"[{timestamp}] CRITICAL: {banner}", (15, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            # Green Nominal Banner
            cv2.putText(frame, f"[{timestamp}] STATUS: NORMAL | MODE: {mode.upper()}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Telemetry Display on Bottom
        if mode == "driver" and metrics:
            ear_val = metrics.get("EAR", 0.0)
            mar_val = metrics.get("MAR", 0.0)
            cv2.putText(frame, f"EAR: {ear_val:.2f} | MAR: {mar_val:.2f}",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        cv2.imshow("Fleet Safety Edge Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified AI Fleet Safety Platform")
    parser.add_argument("--source", type=str, default="0",
                        help="Video file path or webcam index (e.g. 0, 1, or video.mp4)")
    parser.add_argument("--mode", type=str, choices=["driver", "road"], default="driver",
                        help="Execution mode: 'driver' or 'road'")
    args = parser.parse_args()

    run_fleet_safety(source=args.source, mode=args.mode)
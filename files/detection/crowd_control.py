import cv2
import time
import pygame
import threading
import torch
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# MQTT Setup
MQTT_BROKER = "test.mosquitto.org"
MQTT_TOPIC = "crowd/alert"

_socketio = None

def set_socketio(sio):
    global _socketio
    _socketio = sio

def detect_crowd(webcam_index=0, model_path='yolov8n.pt', alert_sound_path='static/audio/beep.wav', crowd_threshold=5):
    """Detects crowd using YOLOv8 and plays an alert if the crowd exceeds the threshold."""

    # Load YOLOv8 Model
    model = YOLO(model_path)

    # Initialize Pygame for Sound Alerts
    pygame.mixer.init()
    try:
        alert_sound = pygame.mixer.Sound(alert_sound_path)
    except Exception as e:
        print(f"[Crowd] Sound load error: {e}")
        alert_sound = None

    # MQTT
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.connect(MQTT_BROKER, 1883, 60)
    except Exception as e:
        print(f"[Crowd] MQTT connect error: {e}")
        mqtt_client = None

    alert_active = False
    stop_threads = False

    def play_alert():
        nonlocal alert_active
        if not alert_active and alert_sound:
            alert_active = True
            alert_sound.play(-1)

    def stop_alert():
        nonlocal alert_active
        if alert_active and alert_sound:
            alert_sound.stop()
            alert_active = False

    camera = cv2.VideoCapture(webcam_index)  # 0 = laptop built-in webcam
    if not camera.isOpened():
        print("[Crowd] Error: Could not open video stream.")
        if _socketio:
            _socketio.emit("detection_alert", {"type": "error", "message": "Could not open video stream."})
        return

    frame = None
    last_alert_time = 0
    alert_cooldown = 10

    def capture_frame():
        nonlocal frame, stop_threads
        while not stop_threads:
            ret, new_frame = camera.read()
            if ret:
                frame = new_frame

    capture_thread = threading.Thread(target=capture_frame, daemon=True)
    capture_thread.start()

    print("[Crowd] Detection started.")

    while True:
        if frame is None:
            time.sleep(0.05)
            continue

        frame_np = np.array(frame)

        with torch.no_grad():
            results = model.track(frame_np, conf=0.5, persist=True, verbose=False)

        person_count = sum(1 for obj in results[0].boxes.cls if int(obj) == 0)

        for i, box in enumerate(results[0].boxes.xyxy):
            if int(results[0].boxes.cls[i]) == 0:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_np, (x1, y1), (x2, y2), (212, 175, 55), 2)

        cv2.putText(frame_np, f"People: {person_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)

        if person_count >= crowd_threshold:
            threading.Thread(target=play_alert, daemon=True).start()
            now = time.time()
            if now - last_alert_time > alert_cooldown:
                last_alert_time = now
                msg = f"Crowd Alert: {person_count} people detected!"
                if mqtt_client:
                    try:
                        mqtt_client.publish(MQTT_TOPIC, msg)
                    except Exception:
                        pass
                if _socketio:
                    _socketio.emit("detection_alert", {"type": "crowd", "message": msg, "count": person_count})
        else:
            stop_alert()

        cv2.imshow("SecureVision – Crowd Control", frame_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

    camera.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    capture_thread.join()
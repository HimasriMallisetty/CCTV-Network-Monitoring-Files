import cv2
import time
import pygame
import threading
import os
import paho.mqtt.client as mqtt

_socketio = None

def set_socketio(sio):
    global _socketio
    _socketio = sio

def run_weapon_detection(webcam_index=0, cascade_path='detection/cascade.xml',
                          alert_sound='static/audio/beep.wav',
                          mqtt_broker="test.mosquitto.org", mqtt_topic="crowd/alert"):
    """Detects weapons using Haar Cascade Classifier."""

    if not os.path.exists(cascade_path):
        print(f"[Weapon] Cascade file not found: {cascade_path}")
        if _socketio:
            _socketio.emit("detection_alert", {"type": "error", "message": "Weapon cascade file missing."})
        return

    gun_cascade = cv2.CascadeClassifier(cascade_path)
    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound(alert_sound)
    except Exception:
        sound = None

    try:
        mqtt_client = mqtt.Client()
        mqtt_client.connect(mqtt_broker, 1883, 60)
    except Exception:
        mqtt_client = None

    cap = cv2.VideoCapture(webcam_index)  # 0 = laptop built-in webcam
    if not cap.isOpened():
        print("[Weapon] Error: Could not open stream.")
        if _socketio:
            _socketio.emit("detection_alert", {"type": "error", "message": "Weapon cam failed to open."})
        return

    last_alert_time = 0
    alert_cooldown = 3

    print("[Weapon] Detection started.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        weapon_detected = False
        for (x, y, w, h) in guns:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 220), 3)
            weapon_detected = True

        if weapon_detected and (time.time() - last_alert_time) > alert_cooldown:
            last_alert_time = time.time()
            if sound:
                threading.Thread(target=sound.play, daemon=True).start()
            msg = "⚠ Weapon Detected!"
            if mqtt_client:
                try:
                    mqtt_client.publish(mqtt_topic, msg)
                except Exception:
                    pass
            if _socketio:
                _socketio.emit("detection_alert", {"type": "weapon", "message": msg})

        cv2.imshow("SecureVision – Weapon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
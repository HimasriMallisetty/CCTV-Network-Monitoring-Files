import cv2
import numpy as np
import time
import threading
import pygame
import paho.mqtt.client as mqtt

_socketio = None

def set_socketio(sio):
    global _socketio
    _socketio = sio

def run_security_monitoring(webcam_index=0, alert_sound='static/audio/beep.wav',
                              mqtt_broker="test.mosquitto.org", mqtt_topic="crowd/alert"):
    """Monitors for suspicious activity via motion analysis."""

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
        print("[Security] Error: Could not open stream.")
        if _socketio:
            _socketio.emit("detection_alert", {"type": "error", "message": "Security cam failed to open."})
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    last_alert_time = 0
    alert_cooldown = 5
    motion_history = []

    print("[Security] Monitoring started.")

    while cap.isOpened():
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        suspicious = False
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                motion_history.append(center)
                if len(motion_history) > 10:
                    motion_history.pop(0)
                if len(motion_history) >= 5:
                    speed = np.linalg.norm(np.array(motion_history[-1]) - np.array(motion_history[0]))
                    if speed > 50:
                        suspicious = True
                        cv2.putText(frame1, "ALERT: Suspicious Activity!", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 200), 2)

        if suspicious and (time.time() - last_alert_time > alert_cooldown):
            last_alert_time = time.time()
            if sound:
                threading.Thread(target=sound.play, daemon=True).start()
            msg = "Suspicious Activity Detected!"
            if mqtt_client:
                try:
                    mqtt_client.publish(mqtt_topic, msg)
                except Exception:
                    pass
            if _socketio:
                _socketio.emit("detection_alert", {"type": "security", "message": msg})

        cv2.imshow("SecureVision – Security Monitor", frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame1 = cap.read()
            ret, frame2 = cap.read()

    cap.release()
    cv2.destroyAllWindows()
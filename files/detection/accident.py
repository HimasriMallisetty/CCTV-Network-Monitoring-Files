import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt

_socketio = None

def set_socketio(sio):
    global _socketio
    _socketio = sio

def run_accident_detection(webcam_index=0, mqtt_broker="test.mosquitto.org", mqtt_topic="crowd/alert"):
    """Detects accidents via motion analysis."""
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.connect(mqtt_broker, 1883, 60)
    except Exception as e:
        print(f"[Accident] MQTT error: {e}")
        mqtt_client = None

    cap = cv2.VideoCapture(webcam_index)  # 0 = laptop built-in webcam
    if not cap.isOpened():
        print("[Accident] Error: Could not open stream.")
        if _socketio:
            _socketio.emit("detection_alert", {"type": "error", "message": "Accident cam failed to open."})
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    last_alert_time = 0
    alert_cooldown = 10
    motion_history = []

    print("[Accident] Detection started.")

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        accident_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 8000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 100, 255), 2)
                motion_history.append((x, y))
                if len(motion_history) > 10:
                    motion_history.pop(0)
                if len(motion_history) >= 5:
                    speed = np.linalg.norm(np.array(motion_history[-1]) - np.array(motion_history[0]))
                    if speed > 30:
                        accident_detected = True

        if accident_detected and time.time() - last_alert_time > alert_cooldown:
            last_alert_time = time.time()
            msg = "Accident Detected!"
            if mqtt_client:
                try:
                    mqtt_client.publish(mqtt_topic, msg)
                except Exception:
                    pass
            if _socketio:
                _socketio.emit("detection_alert", {"type": "accident", "message": msg})

        cv2.imshow("SecureVision – Accident Detection", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame1 = frame2

    cap.release()
    cv2.destroyAllWindows()
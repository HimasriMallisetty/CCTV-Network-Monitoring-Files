from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, send_from_directory
import json, os, threading, glob, datetime, time
import cv2
import numpy as np
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "securevision_secret_2024"
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")
LOG_PATH    = os.path.join(BASE_DIR, "logs", "detection_logs.txt")
VIDEO_DIR   = os.path.join(BASE_DIR, "static", "videos")

os.makedirs(os.path.join(BASE_DIR, "logs"),      exist_ok=True)
os.makedirs(VIDEO_DIR,                            exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "detection"), exist_ok=True)
if not os.path.exists(LOG_PATH):
    open(LOG_PATH, "w").close()

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ── Config ────────────────────────────────────────────────────────────────────
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def log_event(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

# ── Camera state ──────────────────────────────────────────────────────────────
_camera         = None
_camera_lock    = threading.Lock()
_stop_camera    = threading.Event()
_capture_thread = None
_camera_in_use  = threading.Event()

_current_frame   = None
_annotated_frame = None
_frame_lock      = threading.Lock()

_detection_thread = None
_active_mode      = None

# ── Camera open / release ─────────────────────────────────────────────────────
def open_camera(src):
    global _camera
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None
            time.sleep(0.8)

        # Use DirectShow on Windows (avoids MSMF -1072873821 error)
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)

        time.sleep(0.4)

        if not cap.isOpened():
            print(f"[Camera] CAP_DSHOW failed, retrying default for src={src}")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(src)
            time.sleep(0.4)
            if not cap.isOpened():
                print(f"[Camera] Failed to open: {src}")
                return None

        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        _camera = cap
        _camera_in_use.set()
        print(f"[Camera] Opened: src={src}")
        return cap

def release_camera():
    global _camera
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None
            _camera_in_use.clear()
            time.sleep(0.5)
            print("[Camera] Released.")

# ── Frame helpers ─────────────────────────────────────────────────────────────
def _encode_jpeg(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()

def set_annotated(frame):
    global _annotated_frame
    with _frame_lock:
        _annotated_frame = frame.copy()

# ── MJPEG stream ──────────────────────────────────────────────────────────────
def generate_stream(annotated=False):
    while True:
        time.sleep(0.033)
        with _frame_lock:
            frame = (_annotated_frame if annotated and _annotated_frame is not None
                     else _current_frame)
        if frame is None:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Activate a detection module to start",
                        (55, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 1)
            frame = ph
        try:
            data = _encode_jpeg(frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        except Exception:
            break

@app.route("/video_feed")
def video_feed():
    if "user" not in session:
        return "", 403
    return Response(generate_stream(annotated=False),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed/annotated")
def video_feed_annotated():
    if "user" not in session:
        return "", 403
    return Response(generate_stream(annotated=True),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Frame capture loop ────────────────────────────────────────────────────────
def _frame_capture_loop(src):
    global _current_frame, _annotated_frame

    cap = open_camera(src)
    if cap is None:
        socketio.emit("detection_alert", {
            "type": "error",
            "message": "Could not open camera. Close other apps using it."
        })
        return

    is_video = isinstance(src, str)
    print(f"[Capture] Loop started. src={src}")

    with _frame_lock:
        _current_frame   = None
        _annotated_frame = None

    consecutive_failures = 0

    while not _stop_camera.is_set():
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if is_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                consecutive_failures = 0
                continue
            else:
                if consecutive_failures > 10:
                    print("[Capture] Too many failures, stopping.")
                    break
                time.sleep(0.1)
                continue

        consecutive_failures = 0
        with _frame_lock:
            _current_frame   = frame.copy()
            _annotated_frame = frame.copy()
        time.sleep(0.02)

    release_camera()
    print("[Capture] Loop ended.")

def start_capture(src):
    global _capture_thread
    _stop_camera.set()
    if _capture_thread and _capture_thread.is_alive():
        print("[Capture] Waiting for old thread to finish...")
        _capture_thread.join(timeout=3.0)
    if _camera_in_use.is_set():
        _camera_in_use.wait(timeout=2.0)
        time.sleep(0.5)
    _stop_camera.clear()
    _capture_thread = threading.Thread(
        target=_frame_capture_loop, args=(src,), daemon=True
    )
    _capture_thread.start()
    print(f"[Capture] New thread started for src={src}")

def stop_capture():
    _stop_camera.set()

# ── Detection: Crowd ──────────────────────────────────────────────────────────
def _run_crowd(model_path, threshold):
    from ultralytics import YOLO
    import torch
    print(f"[Crowd] Loading YOLO, threshold={threshold}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        socketio.emit("detection_alert", {"type": "error", "message": f"YOLO load failed: {e}"})
        return

    log_event("Crowd detection started")
    last_alert = 0

    while not _stop_camera.is_set():
        with _frame_lock:
            frame = _current_frame.copy() if _current_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue

        try:
            with torch.no_grad():
                results = model.track(frame, conf=0.4, persist=True, verbose=False)
        except Exception as e:
            print(f"[Crowd] Inference error: {e}")
            time.sleep(0.1)
            continue

        person_count = 0
        annotated = frame.copy()
        if results[0].boxes is not None:
            for i, cls in enumerate(results[0].boxes.cls):
                if int(cls) == 0:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (212, 175, 55), 2)
                    cv2.putText(annotated, "Person", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (212, 175, 55), 1)

        cv2.putText(annotated, f"People: {person_count}  Threshold: {threshold}",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(annotated, f"People: {person_count}  Threshold: {threshold}",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 1)

        if person_count >= threshold:
            cv2.putText(annotated, "! CROWD ALERT !",
                        (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 60, 255), 3)
            now = time.time()
            if now - last_alert > 8:
                last_alert = now
                msg = f"Crowd Alert: {person_count} person(s) detected!"
                socketio.emit("detection_alert", {
                    "type": "crowd", "message": msg,
                    "count": person_count, "audio": "crowd"
                })
                log_event(msg)

        set_annotated(annotated)
        time.sleep(0.03)


# ── Detection: Weapon (YOLOv8-based, accurate) ────────────────────────────────
# The original cascade.xml is a generic Haar cascade that detects face/body
# shapes — NOT guns. It fires on any rectangular object (faces, torsos, boxes).
# Fix: Use YOLOv8 object detection. The COCO dataset does NOT include "gun" as
# a class, but we can detect suspicious scenarios using:
#   - Class 43 = knife
#   - Class 76 = scissors  (proxy for bladed objects)
# For a real gun detector, you would need a custom-trained YOLO model.
# This implementation uses YOLOv8 with strict confidence thresholds
# so it only alerts on objects it is genuinely confident about.
# It also avoids false positives from the broken cascade entirely.

# YOLO COCO classes that are potential threats
WEAPON_CLASSES = {
    43: ("knife",     (0, 0, 220),   0.55),   # knife — min confidence 55%
    76: ("scissors",  (0, 120, 220), 0.60),   # scissors — min confidence 60%
}

def _run_weapon_yolo(model_path):
    from ultralytics import YOLO
    import torch
    print("[Weapon] Loading YOLO for weapon detection...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        socketio.emit("detection_alert", {"type": "error", "message": f"YOLO load failed: {e}"})
        return

    log_event("Weapon detection started (YOLO mode)")
    last_alert = 0

    while not _stop_camera.is_set():
        with _frame_lock:
            frame = _current_frame.copy() if _current_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue

        try:
            with torch.no_grad():
                results = model(frame, conf=0.4, verbose=False)
        except Exception as e:
            print(f"[Weapon] Inference error: {e}")
            time.sleep(0.1)
            continue

        annotated    = frame.copy()
        found_weapon = False
        weapon_names = []

        if results[0].boxes is not None:
            for i, cls in enumerate(results[0].boxes.cls):
                cls_id = int(cls)
                if cls_id in WEAPON_CLASSES:
                    label, color, min_conf = WEAPON_CLASSES[cls_id]
                    conf = float(results[0].boxes.conf[i])
                    if conf >= min_conf:   # only fire if above minimum confidence
                        found_weapon = True
                        weapon_names.append(f"{label} ({conf:.0%})")
                        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{label.upper()} {conf:.0%}",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        status_text = "Weapon Detection Active"
        cv2.putText(annotated, status_text, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(annotated, status_text, (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 1)

        if found_weapon:
            cv2.putText(annotated, "! WEAPON ALERT !",
                        (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 220), 3)
            now = time.time()
            if now - last_alert > 6:
                last_alert = now
                names_str = ", ".join(weapon_names)
                msg = f"Weapon Detected: {names_str}"
                socketio.emit("detection_alert", {
                    "type": "weapon", "message": msg, "audio": "weapon"
                })
                log_event(msg)

        set_annotated(annotated)
        time.sleep(0.03)


# ── Detection launcher ────────────────────────────────────────────────────────
def start_detection(mode):
    global _detection_thread, _active_mode

    config = load_config()

    src = 0
    if config.get("detection_mode") == "recorded":
        active_file = config.get("active_video_file", "")
        specific    = os.path.join(VIDEO_DIR, active_file) if active_file else ""
        videos      = glob.glob(os.path.join(VIDEO_DIR, "*.*"))
        src = specific if (active_file and os.path.exists(specific)) else (videos[0] if videos else 0)

    start_capture(src)
    time.sleep(1.2)

    _active_mode = mode

    if mode == "crowd":
        threshold = config.get("crowd_threshold", 5)
        t = threading.Thread(target=_run_crowd, args=("yolov8n.pt", threshold), daemon=True)
    elif mode == "weapon":
        t = threading.Thread(target=_run_weapon_yolo, args=("yolov8n.pt",), daemon=True)
    else:
        return False, "Unknown detection mode."

    _detection_thread = t
    t.start()
    log_event(f"Detection started: {mode}")
    return True, f"{mode.title()} detection started."


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == "admin" and password == "admin123":
            session["user"] = username
            log_event(f"Login: {username}")
            return redirect(url_for("dashboard"))
        error = "Invalid credentials. Please try again."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    stop_capture()
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    config = load_config()
    videos = [os.path.basename(v) for v in glob.glob(os.path.join(VIDEO_DIR, "*.*"))
              if os.path.splitext(v)[1].lower() in ALLOWED_VIDEO_EXT]
    return render_template("dashboard.html",
                           mode=config.get("detection_mode", "live"),
                           config=config,
                           videos=videos,
                           active_mode=_active_mode)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if "user" not in session:
        return redirect(url_for("login"))
    config = load_config()
    if request.method == "POST":
        config["detection_mode"]  = request.form.get("detection_mode", "live")
        config["camera_quality"]  = request.form.get("camera_quality", "1080p")
        config["storage_limit"]   = request.form.get("storage_limit", "10")
        config["crowd_threshold"] = int(request.form.get("crowd_threshold", 5))
        save_config(config)
        log_event("Settings updated")
        return redirect(url_for("dashboard"))
    return render_template("settings.html", config=config)

@app.route("/logs")
def logs():
    if "user" not in session:
        return redirect(url_for("login"))
    try:
        with open(LOG_PATH, "r") as f:
            entries = [l.strip() for l in f.readlines() if l.strip()]
    except Exception:
        entries = []
    return render_template("logs.html", logs=entries)

@app.route("/about")
def about():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")

@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    open(LOG_PATH, "w").close()
    return jsonify({"message": "Logs cleared."})

# ── Video upload ───────────────────────────────────────────────────────────────
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f   = request.files["video"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": f"Unsupported: {', '.join(ALLOWED_VIDEO_EXT)}"}), 400
    name = secure_filename(f.filename)
    f.save(os.path.join(VIDEO_DIR, name))
    log_event(f"Video uploaded: {name}")
    return jsonify({"success": True, "filename": name})

@app.route("/delete_video", methods=["POST"])
def delete_video():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    name = request.json.get("filename", "")
    path = os.path.join(VIDEO_DIR, secure_filename(name))
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404

# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/api/start_camera", methods=["POST"])
def api_start_camera():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    start_capture(0)
    return jsonify({"success": True, "message": "Camera started."})

@app.route("/api/set_video_source", methods=["POST"])
def api_set_video_source():
    """
    FIX: request.json can be None if Content-Type header is missing.
    Added force=True so Flask parses JSON regardless of content-type.
    Also added fallback to request.get_json(silent=True).
    """
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    # FIX: force=True parses JSON even if Content-Type is not set correctly
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename") or ""

    if not filename:
        print(f"[API] set_video_source: no filename in payload: {data}")
        return jsonify({"error": "No filename provided"}), 400

    safe = os.path.basename(filename)
    path = os.path.join(VIDEO_DIR, safe)
    if not os.path.exists(path):
        print(f"[API] set_video_source: file not found: {path}")
        return jsonify({"error": f"File not found: {safe}"}), 404

    config = load_config()
    config["detection_mode"]    = "recorded"
    config["active_video_file"] = safe
    save_config(config)
    print(f"[API] set_video_source: set to {safe}")
    return jsonify({"success": True, "file": safe})

@app.route("/api/start_detection", methods=["POST"])
def api_start_detection():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "crowd")
    ok, msg = start_detection(mode)
    return jsonify({"success": ok, "message": msg})

@app.route("/api/stop_detection", methods=["POST"])
def api_stop_detection():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    global _active_mode
    _active_mode = None
    stop_capture()
    config = load_config()
    config["detection_mode"]    = "live"
    config["active_video_file"] = ""
    save_config(config)
    return jsonify({"success": True, "message": "Detection stopped."})

@app.route("/api/status")
def api_status():
    return jsonify({
        "active_mode":     _active_mode,
        "camera_open":     _camera is not None,
        "capture_running": _capture_thread is not None and _capture_thread.is_alive()
    })

@app.route("/get_settings")
def get_settings():
    try:
        return jsonify(load_config())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/audio/<filename>")
def serve_audio(filename):
    """Serve audio from templates/static/audio or static/audio, whichever exists."""
    candidates = [
        os.path.join(BASE_DIR, "templates", "static", "audio"),
        os.path.join(BASE_DIR, "static", "audio"),
    ]
    for folder in candidates:
        if os.path.exists(os.path.join(folder, filename)):
            return send_from_directory(folder, filename)
    return f"Audio file not found: {filename}", 404

if __name__ == "__main__":
    socketio.run(app, debug=False, host="0.0.0.0", port=5000)
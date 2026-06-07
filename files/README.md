# SecureVision — AI Security Surveillance Platform

A luxury-themed Flask web application for AI-powered security monitoring using YOLOv8.

---

## Project Structure

```
securevision/
├── app.py                      # Main Flask application
├── settings.py                 # Config loader
├── requirements.txt            # Python dependencies
├── config/
│   └── config.json             # Runtime configuration
├── logs/
│   └── detection_logs.txt      # Event log file
├── detection/
│   ├── __init__.py
│   ├── crowd_control.py        # YOLOv8 crowd detection
│   ├── accident.py             # Motion-based accident detection
│   ├── weapon_detection.py     # Haar Cascade weapon detection
│   └── security_monitoring.py # Suspicious activity monitoring
├── templates/
│   ├── base.html               # Luxury sidebar layout
│   ├── login.html              # Split-screen login
│   ├── dashboard.html          # Main command centre
│   ├── settings.html           # Configuration UI
│   └── logs.html               # Event log viewer
└── static/
    ├── audio/
    │   └── beep.wav            # ← PUT YOUR .wav FILE HERE
    └── videos/                 # ← PUT VIDEO FILES HERE (for recorded mode)
```

---

## Setup Instructions

### 1. Place your files

```
static/audio/beep.wav       ← your beep alert sound
yolov8n.pt                  ← YOLOv8 nano model in project root
detection/cascade.xml       ← Haar cascade for weapon detection (optional)
static/videos/sample.mp4   ← sample video for recorded mode (optional)
```

> **Note:** If you have `yolov8n.pt` or `yolov8.pt`, place it in the project root. The crowd detection module loads `yolov8n.pt` by default. Edit `app.py` to change the model path.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support (optional, faster):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Run the app

```bash
python app.py
```

Open: **http://localhost:5000**  
Login: `admin` / `admin123`

---

## Configuration

Edit via the **Settings** page in the UI, or directly in `config/config.json`:

| Key | Description |
|-----|-------------|
| `detection_mode` | `"live"` or `"recorded"` |
| `ip_address` | Camera URL (e.g. `http://192.168.1.5:8080/video`) |
| `crowd_threshold` | Number of persons to trigger crowd alert |
| `MQTT_BROKER` | MQTT broker address |
| `MQTT_TOPIC` | MQTT topic for alerts |

---

## Detection Modules

| Module | Trigger Method | Alert Output |
|--------|---------------|--------------|
| **Crowd Control** | Click in dashboard / sidebar | OpenCV window + MQTT + WebSocket |
| **Accident Detection** | Click in dashboard / sidebar | MQTT + WebSocket |
| **Weapon Detection** | Click in dashboard / sidebar | Audio + MQTT + WebSocket |
| **Security Monitor** | Click in dashboard / sidebar | Audio + MQTT + WebSocket |

All detections open an **OpenCV window** (press `Q` to stop) and send real-time alerts to the dashboard via WebSocket.

---

## IP Camera Setup (Android)

1. Install **IP Webcam** from Play Store
2. Start server in the app
3. Use the URL shown (e.g. `http://192.168.x.x:8080/video`)
4. Paste into Settings → Camera IP

---

## Changing the Login Password

Edit `app.py`, find this section and update:

```python
if username == "admin" and password == "admin123":
```

---

## Notes

- The weapon detection requires a `cascade.xml` file. Download a gun cascade from GitHub or use your own trained model.
- MQTT uses the free public broker `test.mosquitto.org` — for production, use your own broker.
- All detection modules run in separate daemon threads and communicate via Flask-SocketIO.

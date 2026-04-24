import os
import io
import base64
import numpy as np
import cv2
import subprocess
import sqlite3
import uuid
from PIL import Image
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify, Response
from sensor_utils import get_sensor_data
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    # Define LED pins for classes: Bluefinger (17), Clubbing (27), Healthy (22)
    LED_PINS = [17, 27, 22] 
    for pin in LED_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

app = Flask(__name__)

# Initialize SQLite Database
DB_FILE = os.path.join(os.path.dirname(__file__), "diagnostics.db")
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS diagnostics
                 (id TEXT PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  temp REAL, hr INTEGER, spo2 INTEGER, class_name TEXT, 
                  confidence REAL, img_b64 TEXT, cam_b64 TEXT)''')
    conn.commit()
    conn.close()

init_db()


MODEL_PATH = os.path.join(os.path.dirname(__file__), "Finger19_lung_cam.onnx")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "cam_weights.npy")
try:
    session = ort.InferenceSession(MODEL_PATH)
    CAM_WEIGHTS = np.load(WEIGHTS_PATH)
except:
    session = None
    CAM_WEIGHTS = None

CLASS_NAMES = [
    "Bluefinger- Oxygen deficiency",
    "Clubbing- Chronic lung disease",
    "Healthy- Normal lung function",
]

# ImageNet normalization
NORM_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
NORM_STD  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - NORM_MEAN) / NORM_STD
    return np.expand_dims(arr, axis=0).astype(np.float32)

@app.route("/")
def index():
    # Initial sensor read for the home page
    sensors = get_sensor_data()
    return render_template("index.html", result=None, sensors=sensors)

@app.route("/get_sensors")
def get_sensors():
    """Endpoint for live sensor updates via AJAX."""
    return jsonify(get_sensor_data())

def _process_image_bytes(image_bytes):
    global session, CAM_WEIGHTS
    if session is None or CAM_WEIGHTS is None:
        try: 
            session = ort.InferenceSession(MODEL_PATH)
            CAM_WEIGHTS = np.load(WEIGHTS_PATH)
        except: return None

    input_tensor = preprocess(image_bytes)

    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    logits = outputs[0][0]
    features = outputs[1][0] if len(outputs) > 1 else None
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / exp_l.sum()

    pred_idx = int(np.argmax(probs))
    
    # Update LEDs based on the predicted class index
    if GPIO_AVAILABLE:
        for idx, pin in enumerate(LED_PINS):
            if idx == pred_idx:
                GPIO.output(pin, GPIO.HIGH)
            else:
                GPIO.output(pin, GPIO.LOW)
    
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    cam_b64 = None
    if CAM_WEIGHTS is not None and features is not None:
        try:
            w = CAM_WEIGHTS[pred_idx]
            cam = np.tensordot(w, features, axes=([0], [0]))
            cam = np.maximum(cam, 0) # ReLU

            cam_resized = cv2.resize(cam, (256, 256), interpolation=cv2.INTER_CUBIC)
            c_min, c_max = cam_resized.min(), cam_resized.max()
            if c_max - c_min > 0:
                cam_resized = (cam_resized - c_min) / (c_max - c_min)

            cam_resized = cv2.GaussianBlur(cam_resized, (15, 15), 0)

            cam_uint8 = np.uint8(255 * cam_resized)
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_float = heatmap.astype(np.float32)

            orig_img_bgr = cv2.cvtColor(np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))), cv2.COLOR_RGB2BGR).astype(np.float32)

            alpha = cam_resized[..., np.newaxis] * 0.7 
            overlay = (orig_img_bgr * (1 - alpha) + heatmap_float * alpha)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            _, buffer = cv2.imencode('.jpg', overlay)
            cam_b64 = base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print("CAM generation failed:", e)

    result = {
        "class_name": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "image_data": img_b64,
        "cam_data": cam_b64,
        "all_probs": [
            {"class_name": name, "confidence": round(float(probs[i]) * 100, 2)}
            for i, name in enumerate(CLASS_NAMES)
        ]
    }
    return result

def capture_image_rpicam():
    try:
        cmd = [
            "rpicam-jpeg",
            "--width", "640",
            "--height", "480",
            "--nopreview",
            "--timeout", "1000",
            "--output", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode == 0 and len(result.stdout) > 0:
            return result.stdout
        else:
            print("rpicam-jpeg error:", result.stderr.decode(errors="replace"))
            return None
    except Exception as e:
        print("Exception capturing with rpicam-jpeg:", e)
        return None

@app.route("/capture_predict", methods=["POST"])
def capture_predict():
    image_bytes = capture_image_rpicam()
    sensors = get_sensor_data()
    if not image_bytes:
        return render_template("index.html", result=None, sensors=sensors, error="Failed to capture image from camera.")
    
    result = _process_image_bytes(image_bytes)
    if not result:
        return render_template("index.html", result=None, sensors=sensors, error="Model not ready.")
        
    # Save to database
    report_id = str(uuid.uuid4())
    temp = sensors.get('temperature', 0)
    hr = sensors.get('heart_rate', 0)
    spo2 = sensors.get('spo2', 0)
    
    # Handle 'Calib...' or 'Finger?' non-numeric edge cases
    try: temp = float(temp)
    except: temp = 0.0
    try: hr = int(hr)
    except: hr = 0
    try: spo2 = int(spo2)
    except: spo2 = 0

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO diagnostics (id, temp, hr, spo2, class_name, confidence, img_b64, cam_b64) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (report_id, temp, hr, spo2, result['class_name'], result['confidence'], result['image_data'], result['cam_data']))
    conn.commit()
    conn.close()
        
    return render_template("result.html", result=result, sensors=sensors, report_id=report_id)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    sensors = get_sensor_data()
    if not file: 
        return render_template("index.html", result=None, sensors=sensors)

    image_bytes = file.read()
    result = _process_image_bytes(image_bytes)
    if not result:
        return render_template("index.html", result=None, sensors=sensors, error="Model not ready.")
        
    # Save to database
    report_id = str(uuid.uuid4())
    temp = sensors.get('temperature', 0)
    hr = sensors.get('heart_rate', 0)
    spo2 = sensors.get('spo2', 0)
    
    try: temp = float(temp)
    except: temp = 0.0
    try: hr = int(hr)
    except: hr = 0
    try: spo2 = int(spo2)
    except: spo2 = 0

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO diagnostics (id, temp, hr, spo2, class_name, confidence, img_b64, cam_b64) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (report_id, temp, hr, spo2, result['class_name'], result['confidence'], result['image_data'], result['cam_data']))
    conn.commit()
    conn.close()
        
    return render_template("result.html", result=result, sensors=sensors, report_id=report_id)

@app.route("/report/<report_id>")
def view_report(report_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, temp, hr, spo2, class_name, confidence, img_b64, cam_b64 FROM diagnostics WHERE id = ?", (report_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        return "Report not found", 404
        
    timestamp, temp, hr, spo2, class_name, confidence, img_b64, cam_b64 = row
    
    result = {
        "class_name": class_name,
        "confidence": confidence,
        "image_data": img_b64,
        "cam_data": cam_b64,
    }
    sensors = {
        "temperature": temp,
        "heart_rate": hr,
        "spo2": spo2
    }
    
    return render_template("report.html", result=result, sensors=sensors, timestamp=timestamp)

camera_process = None

def generate_mjpeg():
    global camera_process
    if camera_process is not None:
        try:
            camera_process.terminate()
            camera_process.wait(timeout=2)
        except:
            pass
            
    cmd = [
        "rpicam-vid",
        "--width", "640",
        "--height", "480",
        "--framerate", "15",
        "--codec", "mjpeg",
        "--timeout", "0",
        "--nopreview",
        "--output", "-"
    ]
    camera_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    
    bytes_buffer = b''
    try:
        while True:
            chunk = camera_process.stdout.read(4096)
            if not chunk:
                break
            bytes_buffer += chunk
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
    except GeneratorExit:
        if camera_process:
            camera_process.terminate()

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # RPi recommended to run on 0.0.0.0 to be accessible via IP
    app.run(host="0.0.0.0", port=5000, debug=True)

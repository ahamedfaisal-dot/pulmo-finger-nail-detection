import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rpi_lung_system_with_gc", "Finger19_lung_cam.onnx")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "rpi_lung_system_with_gc", "cam_weights.npy")

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


NORM_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
NORM_STD  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    
    
    arr = np.array(img, dtype=np.float32) / 255.0
    
   
    arr = arr.transpose(2, 0, 1)
    
    
    arr = (arr - NORM_MEAN) / NORM_STD
    
    
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


@app.route("/")
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    global session
    if session is None:
        try:
            session = ort.InferenceSession(MODEL_PATH)
        except Exception as e:
            return f"Model not ready yet: {str(e)}"

    file = request.files.get("image")
    if not file:
        return render_template("index.html", result=None)

    image_bytes = file.read()

    
    input_tensor = preprocess(image_bytes)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    
    logits = outputs[0][0]
    features = outputs[1][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    pred_idx = int(np.argmax(probs))
    confidence = round(float(probs[pred_idx]) * 100, 2)

   
    cam_b64 = None
    if CAM_WEIGHTS is not None:
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

    
    all_probs = []
    for i, name in enumerate(CLASS_NAMES):
        all_probs.append({
            "class_name": name,
            "confidence": round(float(probs[i]) * 100, 2)
        })

    
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    result = {
        "class_name": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "all_probs": all_probs,
        "image_data": img_b64,
        "cam_data": cam_b64,
    }
    return render_template("index.html", result=result)


if __name__ == "__main__":
    
    if not os.path.exists("templates"):
        os.makedirs("templates")
    app.run(debug=True, port=5000)

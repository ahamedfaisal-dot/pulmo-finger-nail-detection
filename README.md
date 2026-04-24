# Finger-Based Lung Health Classification System

> **Edge AI · Raspberry Pi · MobileNetV2 · CAM Explainability · Multi-Sensor Fusion**

A non-invasive, real-time lung health screening system that classifies three clinical conditions directly from a fingertip photograph, fused with live biometric readings (temperature, heart rate, SpO₂) — all running on a Raspberry Pi 4 as a self-contained medical edge device.

---

## Table of Contents

1. [Clinical Background](#-clinical-background)
2. [System Architecture](#-system-architecture)
3. [Project Structure](#-project-structure)
4. [Machine Learning Pipeline (PC / Training Side)](#-machine-learning-pipeline-pc--training-side)
5. [Raspberry Pi Deployment (rpi\_input/)](#-raspberry-pi-deployment-rpi_input)
   - [Hardware Requirements](#hardware-requirements)
   - [Wiring Diagram](#wiring-diagram)
   - [Software Setup on RPi](#software-setup-on-rpi)
   - [Running the System](#running-the-system)
   - [Web Interface &amp; Endpoints](#web-interface--endpoints)
   - [LED Feedback System](#led-feedback-system)
   - [Database &amp; Reports](#database--reports)
6. [Grad-CAM Explainability](#-grad-cam-explainability)
7. [3D Printed Enclosure](#-3d-printed-enclosure)
8. [Troubleshooting](#-troubleshooting)
9. [Class Definitions](#-class-definitions)

---

## Clinical Background

Certain lung and oxygenation disorders manifest visually in the fingertips:

| Condition                       | Visual Indicator            | Root Cause                                    |
| ------------------------------- | --------------------------- | --------------------------------------------- |
| **Bluefinger (Cyanosis)** | Bluish/purple discoloration | Severe oxygen deficiency (SpO₂ < 85%)        |
| **Clubbing**              | Rounded, bulging fingertips | Chronic lung disease (COPD, fibrosis, cancer) |
| **Healthy**               | Normal pink coloration      | Normal lung function                          |

This system automates the visual assessment by training a deep CNN on finger images and fusing the result with real-time SpO₂ and temperature data from hardware sensors.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING SIDE (PC/GPU)                 │
│                                                          │
│   train/ dataset  ──►  train_export.py                  │
│   (3 classes,         MobileNetV2 fine-tune             │
│    256×256 images)    15 epochs, best_model.pt          │
│                        │                                │
│                        ▼                                │
│                   export_cam.py                         │
│                   (hook last conv layer)                │
│                   Finger19_lung_cam.onnx +              │
│                   cam_weights.npy                       │
└──────────────┬──────────────────────────────────────────┘
               │   Copy .onnx + .npy  via SCP/USB
               ▼
┌─────────────────────────────────────────────────────────┐
│               RASPBERRY PI 4 (Edge Device)               │
│                                                          │
│  ┌─────────────┐   ┌───────────────┐  ┌──────────────┐ │
│  │  RPi Camera │   │  MLX90614     │  │  MAX30102    │ │
│  │  (rpicam)   │   │  IR Temp      │  │  HR + SpO₂  │ │
│  └──────┬──────┘   └───────┬───────┘  └──────┬───────┘ │
│         │                  │  I²C Bus         │         │
│         ▼                  └─────────┬────────┘         │
│    image_bytes              sensor_utils.py             │
│         │                            │                  │
│         └────────────┬───────────────┘                  │
│                      ▼                                   │
│               rpi_input/app.py                          │
│               Flask Web Server (:5000)                  │
│                      │                                   │
│          ┌───────────┼───────────────┐                  │
│          ▼           ▼               ▼                  │
│     ONNX Inference  CAM         SQLite DB               │
│     (3-class pred)  Overlay     diagnostics.db          │
│          │                           │                  │
│          ▼                           ▼                  │
│     LED GPIO Pins               PDF Reports             │
│     (17 / 27 / 22)              via /report/<id>        │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Finger_256_lung/
│
├── train/                        # Training dataset (not committed – see .gitignore)
│   ├── Bluefinger/               # Class 0: Oxygen deficiency
│   ├── Clubbing/                 # Class 1: Chronic lung disease
│   └── Healthy/                  # Class 2: Normal
│
├── test/                         # Held-out test images
│
├── train_export.py               # Main training script (MobileNetV2 → ONNX)
├── export_cam.py                 # Exports CAM-capable ONNX + cam_weights.npy
├── inspect_mat.py                # Utility to inspect .mat dataset files
│
├── Finger19_lung.onnx            # Standard ONNX model (no CAM outputs)
├── best_model.pt                 # PyTorch checkpoint (best validation accuracy)
│
├── app.py                        # Standalone Flask demo (PC use, no sensors)
├── requirements.txt              # PC/dev dependencies
│
├── rpi_input/                    # ← DEPLOY THIS FOLDER TO THE RASPBERRY PI
│   ├── app.py                    # Main RPi Flask application
│   ├── sensor_utils.py           # MLX90614 + MAX30102 driver & signal processing
│   ├── Finger19_lung_cam.onnx    # CAM-enabled ONNX model (2 outputs)
│   ├── cam_weights.npy           # Pre-extracted CAM classifier weights
│   ├── requirements_rpi.txt      # RPi Python dependencies
│   │
│   ├── templates/
│   │   ├── index.html            # Home page (live camera feed + upload)
│   │   ├── result.html           # Diagnosis result with CAM overlay
│   │   └── report.html           # Printable/shareable report page
│   │
│   └── hardware_3d_model/
│       └── enclosure.scad        # OpenSCAD enclosure design for the device
│
└── .gitignore
```

---

## Machine Learning Pipeline (PC / Training Side)

### Step 1 — Prepare the Dataset

Organize training images into sub-folders by class:

```
train/
  Bluefinger/   ← finger images showing cyanosis
  Clubbing/     ← finger images showing clubbing
  Healthy/      ← normal healthy finger images
```

Each image should be a clear, well-lit photograph of a fingertip (minimum 100 images per class recommended for good generalization).

### Step 2 — Install PC Dependencies

```bash
pip install torch torchvision onnx onnxruntime flask numpy Pillow opencv-python
```

Or use GPU-accelerated training (requires CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3 — Train the Model

```bash
python train_export.py
```

**What this does:**

- Loads the dataset from `train/` using `ImageFolder`
- Applies data augmentation: random flip, rotation ±15°, color jitter
- Fine-tunes **MobileNetV2** (pretrained on ImageNet) with a 3-class head
- Trains for **15 epochs** with Adam optimizer (lr=1e-3, StepLR scheduler)
- Saves the best checkpoint as `best_model.pt`
- Exports a standard ONNX model: `Finger19_lung.onnx`

**Expected output:**

```
Using device: cuda
Classes: ['Bluefinger', 'Clubbing', 'Healthy']
Epoch 01/15 | Loss=0.9123 | Train=68.3% | Val=71.2%
...
Epoch 15/15 | Loss=0.1204 | Train=97.1% | Val=94.8%
Best val acc: 94.8%
Exported ONNX → Finger19_lung.onnx
```

### Step 4 — Export CAM-Capable Model

```bash
python export_cam.py
```

**What this does:**

- Hooks into MobileNetV2's final convolutional layer to extract feature maps
- Exports a dual-output ONNX: `Finger19_lung_cam.onnx`
  - **Output 0**: class logits `[1, 3]`
  - **Output 1**: feature maps from last conv layer `[1, C, H, W]`
- Saves the classifier weights as `cam_weights.npy` for fast CAM computation at inference

---

## Raspberry Pi Deployment (`rpi_input/`)

### Hardware Requirements

| Component             | Model                                                 | Interface               |
| --------------------- | ----------------------------------------------------- | ----------------------- |
| Microcontroller       | Raspberry Pi 4 (2 GB+ RAM)                            | —                      |
| Camera                | Official RPi Camera Module v2 or v3                   | CSI ribbon cable        |
| IR Temperature Sensor | MLX90614                                              | I²C (SDA/SCL)          |
| Heart Rate + SpO₂    | MAX30102                                              | I²C (SDA/SCL)          |
| Status LEDs           | Standard 5 mm LEDs × 3 (Red, Yellow, Green)          | GPIO (BCM)              |
| Resistors             | 220 Ω × 3                                           | In series with each LED |
| Enclosure             | 3D printed (see `hardware_3d_model/enclosure.scad`) | —                      |

---

### Wiring Diagram

#### I²C Sensors (Both share the same I²C bus)

```
Raspberry Pi 4 GPIO Header
─────────────────────────────────────────────────────
Pin 1  (3.3 V)  ──────────────┬─── MLX90614 VIN
                              └─── MAX30102 VIN

Pin 6  (GND)    ──────────────┬─── MLX90614 GND
                              └─── MAX30102 GND

Pin 3  (SDA1)   ──────────────┬─── MLX90614 SDA
                              └─── MAX30102 SDA

Pin 5  (SCL1)   ──────────────┬─── MLX90614 SCL
                              └─── MAX30102 SCL
─────────────────────────────────────────────────────
```

> **Note:** Add 4.7 kΩ pull-up resistors from SDA and SCL to 3.3 V if your breakout boards don't already include them.

#### LED Indicators (GPIO BCM Numbering)

```
Raspberry Pi GPIO                LED
──────────────────────────────────────────────
GPIO 17 (Pin 11) ── 220 Ω ── [Blue/Red LED]   Bluefinger (Oxygen deficiency)
GPIO 27 (Pin 13) ── 220 Ω ── [Yellow LED]     Clubbing (Chronic lung disease)
GPIO 22 (Pin 15) ── 220 Ω ── [Green LED]      Healthy (Normal)
GND      (Pin 9) ──────────── All LED cathodes
──────────────────────────────────────────────
```

Only the LED corresponding to the predicted class illuminates after each scan. All others go LOW.

---

### Software Setup on RPi

#### 1. Flash OS & Enable Interfaces

Flash **Raspberry Pi OS (64-bit, Bookworm)** and enable required interfaces:

```bash
sudo raspi-config
```

- **Interface Options → Camera** → Enable
- **Interface Options → I2C** → Enable
- Reboot

#### 2. Verify Camera

```bash
rpicam-hello --timeout 3000
```

You should see a brief camera preview. If using a legacy camera with an older OS, ensure `dtoverlay=imx219` is in `/boot/config.txt`.

#### 3. Verify I²C Sensors

```bash
sudo apt install -y i2c-tools
i2cdetect -y 1
```

Expected output (addresses may differ slightly):

```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
50: -- -- -- -- -- -- -- 57                          ← MAX30102
5a: 5a -- -- -- -- -- -- --                          ← MLX90614
```

#### 4. Transfer Project Files

Copy the `rpi_input/` folder to the Pi (replace `<PI_IP>` with your Pi's IP address):

```bash
scp -r rpi_input/ pi@<PI_IP>:/home/pi/lung_system/
```

Or use a USB drive and copy manually.

**The critical files that must be present on the RPi:**

```
lung_system/
├── app.py
├── sensor_utils.py
├── Finger19_lung_cam.onnx      ← generated by export_cam.py on PC
├── cam_weights.npy             ← generated by export_cam.py on PC
├── requirements_rpi.txt
└── templates/
    ├── index.html
    ├── result.html
    └── report.html
```

#### 5. Install Python Dependencies

```bash
cd /home/pi/lung_system/
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_rpi.txt
```

> **Note on `onnxruntime`:** On Raspberry Pi (ARM64), install the ARM-optimized wheel if the standard pip install fails:
>
> ```bash
> pip install onnxruntime   # Try standard first
> # If it fails, use:
> pip install onnxruntime-aarch64
> ```

---

### Running the System

#### Manual Start

```bash
cd /home/pi/lung_system/
source venv/bin/activate
python app.py
```

The Flask server starts on **`http://0.0.0.0:5000`**. Access it from any device on the same network:

```
http://<PI_IP>:5000
```

#### Auto-Start on Boot (systemd Service)

Create a service file to start automatically:

```bash
sudo nano /etc/systemd/system/lunghealth.service
```

Paste the following (adjust paths as needed):

```ini
[Unit]
Description=Lung Health Classification System
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/lung_system
ExecStart=/home/pi/lung_system/venv/bin/python /home/pi/lung_system/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable lunghealth.service
sudo systemctl start lunghealth.service
sudo systemctl status lunghealth.service
```

---

### Web Interface & Endpoints

| Route                   | Method | Description                                                           |
| ----------------------- | ------ | --------------------------------------------------------------------- |
| `/`                   | GET    | Home page — shows live sensor readings, camera feed, and upload form |
| `/video_feed`         | GET    | MJPEG live camera stream (embedded in the home page)                  |
| `/get_sensors`        | GET    | JSON endpoint for live sensor polling (called via AJAX every 2 s)     |
| `/capture_predict`    | POST   | Triggers `rpicam-jpeg` to capture a frame and run inference         |
| `/predict`            | POST   | Accepts an uploaded image file and runs inference                     |
| `/report/<report_id>` | GET    | Shows the full historical diagnostic report by UUID                   |

#### Example Sensor JSON Response (`/get_sensors`)

```json
{
  "temperature": 36.8,
  "heart_rate": 74,
  "spo2": 98
}
```

#### Example Prediction Flow

1. User places finger on MAX30102 sensor pad
2. Taps **"Capture & Analyse"** on the web UI
3. `rpicam-jpeg` captures a 640×480 JPEG in ~1 s
4. Image is preprocessed (resize → 256×256, ImageNet normalization)
5. ONNX inference runs → logits + feature maps
6. Softmax gives class probabilities
7. CAM overlay is generated from `cam_weights.npy` × feature maps
8. LED on predicted class GPIO pin goes HIGH
9. Result page shows:
   - Predicted class + confidence
   - All class probabilities
   - Original image + CAM heatmap overlay
   - Live sensor readings at time of capture
10. Result is saved to `diagnostics.db` with a unique report UUID

---

### LED Feedback System

The Raspberry Pi uses three GPIO-connected LEDs to provide instant physical feedback without needing to look at the screen:

```python
LED_PINS = [17, 27, 22]  # Index 0 = Bluefinger, 1 = Clubbing, 2 = Healthy
```

After each inference, only the LED at `LED_PINS[pred_idx]` is set HIGH; the rest are set LOW. This makes the result immediately visible even from a distance.

---

### Database & Reports

All diagnostic results are stored in **SQLite** at `diagnostics.db`:

```sql
CREATE TABLE diagnostics (
    id         TEXT PRIMARY KEY,     -- UUID
    timestamp  DATETIME,             -- Auto-set to current time
    temp       REAL,                 -- Body temperature (°C)
    hr         INTEGER,              -- Heart rate (BPM)
    spo2       INTEGER,              -- Blood oxygen (%)
    class_name TEXT,                 -- Predicted class label
    confidence REAL,                 -- Model confidence (%)
    img_b64    TEXT,                 -- Base64 original image
    cam_b64    TEXT                  -- Base64 CAM overlay image
);
```

Each report is accessible at:

```
http://<PI_IP>:5000/report/<uuid>
```

This URL can be shared, bookmarked, or opened via a QR code for printable clinic-style reports.

---

## 🔥 Grad-CAM Explainability

Class Activation Mapping (CAM) highlights **which pixels the model focused on** when making its decision. This is critical for clinical trust — it shows doctors whether the model is looking at the fingertip (correct) or the background (incorrect).

**How it works in this system:**

1. `export_cam.py` hooks into MobileNetV2's last convolutional layer during export
2. The ONNX model now outputs **two tensors**: logits and the feature map
3. `cam_weights.npy` stores the classifier's weight matrix `[3 classes × C channels]`
4. At inference: `CAM = ReLU( Σ w_c · feature_map_c )` for the predicted class
5. The resulting activation map is:
   - Upsampled to 256×256 with bicubic interpolation
   - Gaussian-blurred for smoothness
   - Converted to a JET colormap heatmap
   - Alpha-blended (70% weight) over the original image

The result reveals exactly which part of the finger — tip, nail, or skin — drove the classification decision.

---

## Troubleshooting

| Problem                                    | Cause                               | Fix                                                                       |
| ------------------------------------------ | ----------------------------------- | ------------------------------------------------------------------------- |
| `rpicam-jpeg: command not found`         | Camera tools not installed          | `sudo apt install -y rpicam-apps`                                       |
| Camera black screen / no output            | Camera not enabled in raspi-config  | Run `sudo raspi-config` → Interface Options → Camera                  |
| `No module named 'smbus2'`               | smbus2 not installed                | `pip install smbus2`                                                    |
| `No module named 'RPi'`                  | RPi.GPIO not installed              | `pip install RPi.GPIO`                                                  |
| I²C sensors not detected at `i2cdetect` | Wrong wiring or I²C not enabled    | Check SDA/SCL connections; enable via raspi-config                        |
| ONNX model fails to load                   | Wrong path or missing file          | Confirm `Finger19_lung_cam.onnx` is in the same folder as `app.py`    |
| `onnxruntime` install fails on ARM       | Incompatible wheel                  | Try `pip install onnxruntime-aarch64` or install from source            |
| SpO₂ reads "Finger?"                      | Finger not placed on MAX30102       | Press fingertip firmly on sensor; signal threshold is `red > 5000`      |
| SpO₂ reads "Calib..."                     | Buffer not full yet (< 150 samples) | Wait ~3 seconds after placing finger                                      |
| LED not lighting                           | GPIO wiring issue                   | Verify LED polarity (long leg → GPIO, short leg → GND via 220 Ω)       |
| Flask not accessible from phone            | Wrong host binding                  | Ensure `app.run(host="0.0.0.0", port=5000)` — do NOT use `127.0.0.1` |

---

## Class Definitions

| Index | Class Name     | Display Label                    | Clinical Meaning                                                    |
| ----- | -------------- | -------------------------------- | ------------------------------------------------------------------- |
| 0     | `Bluefinger` | Bluefinger – Oxygen Deficiency  | Peripheral cyanosis caused by low arterial O₂ saturation           |
| 1     | `Clubbing`   | Clubbing – Chronic Lung Disease | Nail clubbing associated with COPD, pulmonary fibrosis, lung cancer |
| 2     | `Healthy`    | Healthy – Normal Lung Function  | Normal fingertip coloration, no clinical indicators                 |

---

> **⚠️ Disclaimer:** This system is an educational and research prototype. It is **not a certified medical device** and should not be used as the sole basis for any clinical diagnosis. Always consult a qualified healthcare professional.

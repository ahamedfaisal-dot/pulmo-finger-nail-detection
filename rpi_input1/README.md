# Raspberry Pi 4 - Finger Lung Health System

This sub-folder contains the code and configuration for running the system on a Raspberry Pi 4 Model B (4GB) with I2C health sensors.

## 1. Hardware Connections (I2C)

Both sensors are connected to the Raspberry Pi's I2C pins:
- **SDA**: Pin 3 (GPIO 2)
- **SCL**: Pin 5 (GPIO 3)
- **VCC**: 3.3V (Pin 1 or 17)
- **GND**: Ground (Pin 6, 9, etc.)

| Sensor | SDA | SCL | VCC | GND |
| :--- | :--- | :--- | :--- | :--- |
| **MLX90614** (Temp) | Connect | Connect | 3.3V | Ground |
| **MAX30102** (HR) | Connect | Connect | 3.3V | Ground |

## 2. RPi Software Setup

1. **Enable I2C**:
   - Run `sudo raspi-config`
   - Go to `Interfacing Options` -> `I2C` -> `Yes`
   - Reboot your Pi.

2. **Verify Sensors**:
   - Run `sudo i2cdetect -y 1`
   - You should see `0x5A` (MLX90614) and `0x57` (MAX30102) in the grid.

## 3. Installation

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_rpi.txt
```

## 4. Run the System

```bash
python app.py
```
After running, access the dashboard via your browser using the Pi's IP address:
`http://<YOUR_PI_IP>:5000`

## Features
- **Real-time Monitoring**: The web UI polls the I2C sensors every 3 seconds for live Temperature, Heart Rate, and SpO2.
- **On-Device Classification**: Runs the retrained 96.8% accuracy ONNX model directly on the RPi CPU.

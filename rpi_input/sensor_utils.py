import smbus2
import time
import numpy as np
import threading

# ── MLX90614 Temperature Sensor ──────────────────────────────────────
# Default I2C Address: 0x5A
class MLX90614:
    def __init__(self, bus_num=1, address=0x5A):
        try:
            self.bus = smbus2.SMBus(bus_num)
            self.address = address
        except:
            self.bus = None

    def read_temp(self):
        """Read object temperature in Celsius."""
        if not self.bus: return 0.0
        try:
            # Read 2 bytes from RAM address 0x07 (Object 1 Temp)
            data = self.bus.read_word_data(self.address, 0x07)
            temp = (data * 0.02) - 273.15
            return round(temp, 2)
        except:
            return 0.0

# ── MAX30102 Heart Rate & SpO2 Sensor ────────────────────────────────
# Default I2C Address: 0x57
class MAX30102:
    def __init__(self, bus_num=1, address=0x57):
        try:
            self.bus = smbus2.SMBus(bus_num)
            self.address = address
            self.setup()
        except:
            self.bus = None
        
        # Buffers for signal processing
        self.buffer_size = 150
        self.red_data = []
        self.ir_data = []
        
        self.current_hr = "N/A"
        self.current_spo2 = "N/A"
        self._lock = threading.Lock()
        
        if self.bus:
            self.thread = threading.Thread(target=self._poll_sensor, daemon=True)
            self.thread.start()

    def setup(self):
        """Standard initialization for MAX30102."""
        try:
            self.bus.write_byte_data(self.address, 0x09, 0x40) # Reset
            time.sleep(0.1)
            self.bus.write_byte_data(self.address, 0x06, 0x03) # SpO2 Mode (Red + IR)
            self.bus.write_byte_data(self.address, 0x0A, 0x23) # Config: 411us, 50Hz
            self.bus.write_byte_data(self.address, 0x08, 0x1F) # FIFO roll-over, 15 samples
            self.bus.write_byte_data(self.address, 0x0C, 0x24) # LED1 Current (7mA)
            self.bus.write_byte_data(self.address, 0x0D, 0x24) # LED2 Current (7mA)
        except:
            pass

    def read_fifo(self):
        """Read data from the sensor FIFO."""
        if not self.bus: return None
        try:
            # Read 6 bytes for one sample (3 Red, 3 IR)
            d = self.bus.read_i2c_block_data(self.address, 0x07, 6)
            red = (d[0] << 16 | d[1] << 8 | d[2]) & 0x03FFFF
            ir  = (d[3] << 16 | d[4] << 8 | d[5]) & 0x03FFFF
            return red, ir
        except:
            return None

    def _poll_sensor(self):
        """Continuously pulls data from the FIFO queue in a background thread."""
        while True:
            sample = self.read_fifo()
            if sample:
                red, ir = sample
                with self._lock:
                    if red < 5000:
                        self.red_data = []
                        self.ir_data = []
                        self.current_hr = "Finger?"
                        self.current_spo2 = "Finger?"
                    else:
                        self.red_data.append(red)
                        self.ir_data.append(ir)
                        
                        if len(self.red_data) > self.buffer_size:
                            self.red_data.pop(0)
                        if len(self.ir_data) > self.buffer_size:
                            self.ir_data.pop(0)
                        
                        if len(self.red_data) == self.buffer_size:
                            self._calculate_health()
            time.sleep(0.02) # ~50 Hz polling

    def _calculate_health(self):
        """Estimates HR and SpO2 using moving average and peak detection."""
        r_np = np.array(self.red_data)
        r_norm = r_np - np.mean(r_np)
        
        # Detrending (remove baseline wander)
        ma_long = np.convolve(r_norm, np.ones(50)/50, mode='same')
        r_detrend = r_norm - ma_long
        
        # Smoothing
        window = 5
        r_smooth = np.convolve(r_detrend, np.ones(window)/window, mode='valid')
        
        # Dynamic threshold for peak detection
        threshold = np.max(r_smooth) * 0.3
        
        peaks = np.where((r_smooth[1:-1] > r_smooth[:-2]) & (r_smooth[1:-1] > r_smooth[2:]) & (r_smooth[1:-1] > threshold))[0]
        
        if len(peaks) > 1:
            intervals = np.diff(peaks)
            avg_interval = np.mean(intervals)
            bpm = (60 * 50) / avg_interval if avg_interval > 0 else 72
        else:
            bpm = 72
            
        ac_red = np.max(r_np) - np.min(r_np)
        dc_red = np.mean(r_np) if np.mean(r_np) > 0 else 1
        
        i_np = np.array(self.ir_data)
        ac_ir = np.max(i_np) - np.min(i_np)
        dc_ir = np.mean(i_np) if np.mean(i_np) > 0 else 1
        
        ratio = (ac_red / dc_red) / (ac_ir / dc_ir) if (ac_ir / dc_ir) > 0 else 1
        spo2 = 104 - 17 * ratio
        
        self.current_hr = int(np.clip(bpm, 50, 160))
        self.current_spo2 = int(np.clip(spo2, 85, 100))

    def calculate_health(self):
        """Unified read accessor compatible with prior interface."""
        with self._lock:
            if len(self.red_data) == 0:
                return {"heart_rate": "Finger?", "spo2": "Finger?"}
            if len(self.red_data) < self.buffer_size:
                return {"heart_rate": "Calib...", "spo2": "Calib..."}
            return {"heart_rate": self.current_hr, "spo2": self.current_spo2}

# ── Global Instances ────────────────────────────────────────────────
temp_drv = MLX90614()
hr_drv = MAX30102()

def get_sensor_data():
    """Unified entry point for Flask."""
    temp = temp_drv.read_temp()
    health = hr_drv.calculate_health()
    
    return {
        "temperature": temp if temp > 0 else "Err",
        "heart_rate": health["heart_rate"],
        "spo2": health["spo2"]
    }

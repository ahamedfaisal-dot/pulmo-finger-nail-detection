# Finger Lung Classification System - Documentation

This project provides a web-based interface to classify lung conditions (Bluefinger, Clubbing, or Healthy) based on finger images. Since the original ONNX model provided was not performing correctly, a new model was trained on the provided dataset.

## 1. Model Architecture
- **Base Model**: **MobileNetV2** (Pre-trained on ImageNet).
- **Modification**: The final classification layer was replaced with a 3-unit linear layer to match your dataset.
- **Input Size**: 256x256 pixels (RGB).
- **Inference Engine**: ONNX Runtime (CPU/GPU compatible).

## 2. Training Details
- **Hardware**: Trained using the local **NVIDIA RTX 3050 GPU**.
- **Dataset**: ~1,700 images across 3 classes.
- **Preprocessing**: 
  - Resized to 256x256.
  - ImageNet normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).
  - Augmentations: Random horizontal flip, rotation, and color jitter.
- **Performance**: Achieved **96.8% Validation Accuracy** after 15 epochs.
- **Script**: The training logic is available in `train_export.py`.

## 3. Setup & Installation
The system is already set up in this directory with a virtual environment.

### Activation
```powershell
.\venv\Scripts\activate
```

### Dependencies
Installed via `requirements.txt`:
- `flask`: Web framework.
- `onnxruntime`: Efficient inference engine.
- `numpy & Pillow`: Image processing.

## 4. How to Run the Application
1. **Start the Server**:
   ```powershell
   python app.py
   ```
2. **Access the Web Interface**:
   Open your browser and go to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
3. **Classify**:
   - Upload an image from the `train/`, `test/`, or `Result/` folders.
   - Click **Classify**.
   - Review the predicted class and the confidence breakdown for all categories.

## 5. File Structure
- `app.py`: The Flask backend.
- `Finger19_lung.onnx`: The retrained, high-accuracy model.
- `templates/index.html`: The user interface.
- `train_export.py`: The script used to train the model.
- `venv/`: The main virtual environment for running the app.
- `venv_gpu/`: (Optional) Temporary environment used for GPU training.

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import os

DEVICE = "cpu"
class_names = ["Bluefinger- Oxygen deficiency", "Clubbing- Chronic lung disease", "Healthy- Normal lung function"]

# Load base model
basemodel = models.mobilenet_v2(weights=None)
basemodel.classifier[1] = nn.Linear(basemodel.last_channel, len(class_names))
basemodel.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))

class CAMModel(nn.Module):
    def __init__(self, basemodel):
        super().__init__()
        self.features = basemodel.features
        self.classifier = basemodel.classifier
    def forward(self, x):
        f = self.features(x)
        pooled = nn.functional.adaptive_avg_pool2d(f, (1, 1))
        pooled = torch.flatten(pooled, 1)
        out = self.classifier(pooled)
        return out, f

cam_model = CAMModel(basemodel)
cam_model.eval()

# Export
dummy = torch.randn(1, 3, 256, 256)
onnx_out = os.path.join("rpi_lung_system", "Finger19_lung_cam.onnx")

torch.onnx.export(
    cam_model, dummy, onnx_out,
    input_names=["input"],
    output_names=["logits", "features"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "features": {0: "batch"}},
    opset_version=11,
)
print(f"Exported {onnx_out}")

# Extract weights
weights = basemodel.classifier[1].weight.data.numpy() # shape (3, 1280)
np.save(os.path.join("rpi_lung_system", "cam_weights.npy"), weights)
print("Exported cam_weights.npy")

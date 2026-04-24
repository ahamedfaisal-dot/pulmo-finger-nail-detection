"""
train_export.py
---------------
Fine-tunes MobileNetV2 (pretrained on ImageNet) on the 3-class finger dataset,
then exports a correct ONNX file: Finger19_lung.onnx
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ── Config ──────────────────────────────────────────
TRAIN_DIR  = "train"
IMG_SIZE   = 256
BATCH      = 16
EPOCHS     = 15
LR         = 1e-3
ONNX_OUT   = "Finger19_lung.onnx"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Transforms ──────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset ─────────────────────────────────────────
full_ds = datasets.ImageFolder(TRAIN_DIR)
classes = full_ds.classes
print(f"Classes: {classes}")
print(f"Total images: {len(full_ds)}")

val_size  = max(1, int(0.15 * len(full_ds)))
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

# Apply transforms via wrapper
class WithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y

train_loader = DataLoader(WithTransform(train_ds, train_tf), batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(WithTransform(val_ds,   val_tf),   batch_size=BATCH, shuffle=False, num_workers=0)

# ── Model: MobileNetV2 fine-tuned ───────────────────
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
criterion = nn.CrossEntropyLoss()

# ── Training loop ────────────────────────────────────
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
    scheduler.step()

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += imgs.size(0)

    train_acc = 100 * correct / total
    val_acc   = 100 * val_correct / val_total
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Loss={running_loss/total:.4f} | "
          f"Train={train_acc:.1f}% | Val={val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

# ── Load best weights & export ONNX ─────────────────
print(f"\nBest val acc: {best_val_acc:.1f}%")
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
torch.onnx.export(
    model, dummy, ONNX_OUT,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11,
)
print(f"Exported ONNX → {ONNX_OUT}")

# ── Save class names in order ────────────────────────
with open("class_names.txt", "w") as f:
    for c in classes:
        f.write(c + "\n")
print("Classes saved to class_names.txt")
print("Done!")

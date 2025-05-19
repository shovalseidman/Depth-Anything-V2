import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

# Set arguments manually (you can also use argparse if you want to pass from CLI)
input_size = 480
outdir = './output'
encoder = 'vitb'
grayscale = False
pred_only = False

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load DepthAnythingV2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# Load YOLO pose model
yolo_model = YOLO("yolo11x-pose.pt").to(DEVICE)
yolo_model_1 = YOLO("yolo11x").to(DEVICE)
yolo_model_2 = YOLO("yolo11x-seg.pt").to(DEVICE)


# Colormap for depth
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

# Open camera 0
cap1 = cv2.VideoCapture(2)

os.makedirs(outdir, exist_ok=True)

while True:
    ret, frame = cap1.read()
    if not ret:
        break

    ### ---- YOLO Inference ----
    yolo_results = yolo_model(frame)
    yolo_frame = yolo_results[0].plot()

    ### ---- Depth Inference ----
    with torch.no_grad():
        depth = depth_anything.infer_image(frame, input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    if grayscale:
        depth_color = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth_color = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    ### ---- Compose Display ----
    if pred_only:
        combined = cv2.hconcat([depth_color, yolo_frame])
    else:
        split = np.ones((frame.shape[0], 20, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([frame, split, yolo_frame, split, depth_color])

    cv2.imshow("YOLO + DepthAnythingV2", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()

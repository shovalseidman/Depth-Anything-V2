import cv2
import matplotlib
import numpy as np
import os
import torch

from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

# ---- Settings ----
input_size = 480
outdir = './output'
encoder = 'vitb'
grayscale = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# ---- Load Models ----
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# ---- YOLO Models ----
yolo_model_pose = YOLO("yolo11x-pose.pt").to(DEVICE)
yolo_model_det = YOLO("yolo11x.pt").to(DEVICE)
yolo_model_seg = YOLO("yolo11x-seg.pt").to(DEVICE)

cmap = matplotlib.colormaps.get_cmap('Spectral_r')

# ---- Open Camera ----
cap1 = cv2.VideoCapture(2)
os.makedirs(outdir, exist_ok=True)

while True:
    ret, frame = cap1.read()
    if not ret:
        break

    # YOLO Inference
    pose_frame = yolo_model_pose(frame)[0].plot()
    det_frame = yolo_model_det(frame)[0].plot()
    seg_frame = yolo_model_seg(frame)[0].plot()

    # Depth Inference
    with torch.no_grad():
        depth = depth_anything.infer_image(frame, input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_color = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8) if not grayscale else np.repeat(depth[..., np.newaxis], 3, axis=-1)

    # Resize everything to same height (say, 360)
    target_height = 360
    def resize(img): return cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))

    original_resized = resize(frame)
    pose_resized     = resize(pose_frame)
    det_resized      = resize(det_frame)
    seg_resized      = resize(seg_frame)
    depth_resized    = resize(depth_color)

    # Make placeholder (blank white image)
    blank = np.ones_like(original_resized) * 255

    # ---- Build Grid ----
    row1 = cv2.hconcat([original_resized, pose_resized])
    row2 = cv2.hconcat([det_resized, seg_resized])
    row3 = cv2.hconcat([depth_resized, blank])

    final_display = cv2.vconcat([row1, row2, row3])

    # ---- Show Output ----
    cv2.imshow("Multi-Level YOLO + Depth", final_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()

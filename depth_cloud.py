import cv2
import matplotlib
import numpy as np
import os
import torch
import time

from depth_anything_v2.dpt import DepthAnythingV2

# ==== USER PARAMETERS ====
INPUT_SIZE = 518  # Model input size (e.g., 480, 518, 640, 768)
ENCODER_TYPE = 'vits'  # Options: 'vits', 'vitb', 'vitl', 'vitg'
PRED_ONLY = False       # True = only show depth, False = side-by-side
GRAYSCALE = False       # True = gray depth, False = colorful
CAMERA_INDEX = "rtsp://109.207.78.37:8554/mystream"  # RTSP source
CHECKPOINT_DIR = 'checkpoints'
SAVE_OUTPUTS = True  # Set to True to save frames
OUTPUT_DIR = '/home/shovalseidman/Documents/GitHub/Depth-Anything-V2/vis_depth'
SCALE_FACTOR = 0.7  # Resize input frame before processing
# ==========================

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load model
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**model_configs[ENCODER_TYPE])
depth_anything.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/depth_anything_v2_{ENCODER_TYPE}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# Prepare output folders
if SAVE_OUTPUTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "vis"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "depth"), exist_ok=True)

# Colormap for visualization
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

# Start video stream
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Unable to open video stream.")
    exit()

prev_time = time.time()
frame_id = 0

while cap.isOpened():
    ret, frame0 = cap.read()
    if not ret or frame0 is None or not isinstance(frame0, np.ndarray):
        continue

    # Scale down input image
    frame = cv2.resize(frame0, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    original_shape = frame.shape[:2]

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Resize to model input size
    frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    # Run inference
    depth = depth_anything.infer_image(frame_resized, INPUT_SIZE)
    depth = depth if isinstance(depth, np.ndarray) else depth.cpu().numpy()

    # Normalize depth to 0–255 and convert to 8-bit
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    # Resize depth map back to match original shape
    depth_resized = cv2.resize(depth, (original_shape[1], original_shape[0]))

    # Create depth visualization
    if GRAYSCALE:
        depth_vis = np.repeat(depth_resized[..., np.newaxis], 3, axis=-1)
    else:
        depth_vis = (cmap(depth_resized)[:, :, :3] * 255).astype(np.uint8)
        depth_vis = depth_vis[:, :, ::-1]  # RGB → BGR for OpenCV

    # Combine original + depth
    if PRED_ONLY:
        display_frame = depth_vis
    else:
        split_region = np.ones((original_shape[0], 50, 3), dtype=np.uint8) * 255
        display_frame = cv2.hconcat([frame, split_region, depth_vis])

    # Overlay FPS
    cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Depth Estimation', display_frame)

    # Save frames if needed
    if SAVE_OUTPUTS:
        vis_path = os.path.join(OUTPUT_DIR, "vis", f"frame_{frame_id:05d}.jpg")
        depth_path = os.path.join(OUTPUT_DIR, "depth", f"depth_{frame_id:05d}.png")
        cv2.imwrite(vis_path, display_frame)
        cv2.imwrite(depth_path, depth_resized)
        frame_id += 1

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

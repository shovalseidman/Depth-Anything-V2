# vision_pipeline/main.py
import os
import cv2
import yaml
import torch
import numpy as np
import matplotlib
from ultralytics import YOLO
from ultralytics.engine.results import Keypoints
from models.load_depth_model import load_depth_anything
from core.display import draw_panel_grid
from core.roi_filter import filter_yolo_boxes
from core.depth_mapper import depth_to_2d_map
from core.camera_manager import CameraManager

matplotlib.use('Agg')
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

def load_config(config_path='config/cameras.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, path='config/cameras.yaml'):
    with open(path, 'w') as f:
        yaml.dump(config, f)

def select_roi_live(frame, window_name="Draw ROI"):
    roi = []
    drawing = False
    ix = iy = 0

    def draw(event, x, y, flags, param):
        nonlocal roi, drawing, ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi = [ix, iy, x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw)

    while True:
        preview = frame.copy()
        if len(roi) == 4:
            cv2.rectangle(preview, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(1)

        if key == 13:  # Enter
            break
        elif key == 27:  # Esc
            roi = []
            break

    cv2.destroyWindow(window_name)
    return roi if roi else None

def main():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = {
        'pose': YOLO("yolo11x-obb.pt").to(device),
        'det': YOLO("yolo11x.pt").to(device),
        # 'det2': YOLO("yolo11x-obb.pt").to(device),
        'seg': YOLO("yolo11x-seg.pt").to(device),
        'depth': load_depth_anything(config.get('depth_encoder', 'vitb'), device)
    }

    cams = CameraManager(config)
    cam_keys = [k for k in config.keys() if k != 'depth_encoder']
    cam_index_map = {ord(str(i)): cam_keys[i] for i in range(len(cam_keys))}
    prev_boxes = {k: None for k in cam_keys}  # movement tracking

    while True:
        frames = cams.read_all()
        displays = []

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key in cam_index_map:
            selected_cam_id = cam_index_map[key]
            frame = frames.get(selected_cam_id)
            if frame is not None:
                print(f"[INFO] Selecting ROI for {selected_cam_id}...")
                roi = select_roi_live(frame, window_name=f"ROI: {selected_cam_id}")
                if roi:
                    config[selected_cam_id]['roi'] = roi
                    save_config(config)
                    print(f"[INFO] ROI for {selected_cam_id} updated to: {roi}")

        for cam_id, cam_conf in config.items():
            if cam_id == 'depth_encoder':
                continue

            frame = frames.get(cam_id)
            if frame is None:
                print(f"[WARNING] Skipping {cam_id} — no frame received.")
                continue

            roi = cam_conf.get('roi', None)
            panels = []

            if cam_conf['models'].get('pose'):
                result = models['pose'](frame)[0]
                if result.boxes is not None:
                    result.boxes = filter_yolo_boxes(result.boxes, roi)
                if roi and result.keypoints is not None and result.keypoints.xy is not None:
                    x1, y1, x2, y2 = roi
                    keep_kpts = []
                    for i, kp in enumerate(result.keypoints.xy):
                        cx = kp[:, 0].mean().item()
                        cy = kp[:, 1].mean().item()
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            keep_kpts.append(i)
                    if keep_kpts:
                        xy = result.keypoints.xy[keep_kpts]
                        conf = result.keypoints.conf[keep_kpts]
                    else:
                        device = result.keypoints.xy.device
                        xy = torch.empty((0, 17, 2), device=device)
                        conf = torch.empty((0, 17), device=device)
                    kpts_tensor = torch.cat([xy, conf.unsqueeze(-1)], dim=-1)
                    result.keypoints = Keypoints(kpts_tensor, orig_shape=result.orig_shape)
                panels.append(result.plot())

            if cam_conf['models'].get('det'):
                result = models['det'](frame)[0]
                result.boxes = filter_yolo_boxes(result.boxes, roi)

                movement_threshold = 2  # pixels
                current_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
                prev = prev_boxes[cam_id]

                if prev is not None and len(prev) == len(current_boxes):
                    for i, box in enumerate(current_boxes):
                        prev_box = prev[i]
                        dx = abs((box[0] + box[2]) / 2 - (prev_box[0] + prev_box[2]) / 2)
                        dy = abs((box[1] + box[3]) / 2 - (prev_box[1] + prev_box[3]) / 2)
                        if dx > movement_threshold or dy > movement_threshold:
                            # Draw BIG red rectangle with label
                            padding = 20
                            thickness = 6
                            x1 = max(0, int(box[0]) - padding)
                            y1 = max(0, int(box[1]) - padding)
                            x2 = min(frame.shape[1], int(box[2]) + padding)
                            y2 = min(frame.shape[0], int(box[3]) + padding)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
                            cv2.putText(frame, "MOVING", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                prev_boxes[cam_id] = current_boxes
                panels.append(result.plot())

            if cam_conf['models'].get('seg'):
                seg_result = models['seg'](frame)[0]
                if roi:
                    x1, y1, x2, y2 = roi
                    keep_indices = []
                    for i, box in enumerate(seg_result.boxes.xyxy):
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            keep_indices.append(i)
                    if keep_indices:
                        seg_result.masks.data = seg_result.masks.data[keep_indices]
                        seg_result.boxes = seg_result.boxes[keep_indices]
                        seg_result.probs = None
                    else:
                        seg_result.masks.data = torch.empty((0, *seg_result.masks.data.shape[1:]), device=seg_result.masks.data.device)
                        seg_result.boxes = seg_result.boxes[:0]
                seg_img = seg_result.plot()
                panels.append(seg_img)

            if cam_conf['models'].get('depth'):
                with torch.no_grad():
                    depth = models['depth'].infer_image(frame, cam_conf.get('input_size', 480))
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                depth_color = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                panels.append(depth_color)
                panels.append(depth_to_2d_map(depth))

            displays.extend(panels)

        grid = draw_panel_grid(displays, cols=4)
        cv2.imshow("Vision Pipeline", grid)

    cams.release_all()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
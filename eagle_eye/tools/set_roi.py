# tools/set_roi.py
import cv2
import yaml
import argparse
import os

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(path, config):
    with open(path, 'w') as f:
        yaml.dump(config, f)

def select_roi_from_camera(cam_index, window_name="Draw ROI"):
    cap = cv2.VideoCapture(cam_index)
    roi = []

    def draw_rectangle(event, x, y, flags, param):
        nonlocal roi, drawing, ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            roi = [ix, iy, x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]

    drawing = False
    ix = iy = 0
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preview = frame.copy()
        if len(roi) == 4:
            cv2.rectangle(preview, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(1)

        if key == 13:  # Enter
            break
        elif key == 27:  # ESC
            roi = []
            break

    cap.release()
    cv2.destroyAllWindows()
    return roi if roi else None

def main():
    parser = argparse.ArgumentParser(description="Set ROI for a specific camera.")
    parser.add_argument('--camera', type=str, required=True, help="Camera name from config, e.g. 'camera_0'")
    parser.add_argument('--config', type=str, default='config/cameras.yaml', help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.camera not in config:
        print(f"[ERROR] Camera {args.camera} not found in config.")
        return

    cam_index = config[args.camera]['source']
    print(f"[INFO] Opening camera {args.camera} (index {cam_index})...")
    roi = select_roi_from_camera(cam_index)

    if roi:
        config[args.camera]['roi'] = roi
        save_config(args.config, config)
        print(f"[SUCCESS] ROI saved for {args.camera}: {roi}")
    else:
        print("[CANCELLED] No ROI selected.")

if __name__ == '__main__':
    main()

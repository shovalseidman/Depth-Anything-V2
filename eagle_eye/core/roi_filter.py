# core/roi_filter.py
import torch
from ultralytics.engine.results import Boxes

def filter_yolo_boxes(boxes, roi):
    if roi is None or boxes is None or boxes.xyxy.numel() == 0:
        return boxes

    x1, y1, x2, y2 = roi
    keep_indices = []

    for i, box in enumerate(boxes.xyxy):
        bx1, by1, bx2, by2 = box.tolist()
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            keep_indices.append(i)

    if not keep_indices:
        return Boxes(torch.empty((0, 6), device=boxes.xyxy.device), orig_shape=boxes.orig_shape)

    xyxy = boxes.xyxy[keep_indices]
    conf = boxes.conf[keep_indices].unsqueeze(1)
    cls = boxes.cls[keep_indices].unsqueeze(1)

    data = torch.cat([xyxy, conf, cls], dim=1)  # shape: (N, 6)
    return Boxes(data, orig_shape=boxes.orig_shape)

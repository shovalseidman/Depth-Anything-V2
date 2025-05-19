# core/depth_mapper.py
import numpy as np
import cv2

def depth_to_2d_map(depth):
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    obstacle_mask = (depth_norm < 0.4).astype(np.uint8)
    topview = np.max(obstacle_mask, axis=0)
    map_img = np.zeros((100, topview.shape[0]), dtype=np.uint8)
    for x, v in enumerate(topview):
        if v > 0:
            map_img[-10:, x] = 255
    map_display = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    map_display = cv2.resize(map_display, (depth.shape[1], depth.shape[0]))
    return map_display


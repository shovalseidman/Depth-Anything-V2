import cv2
import numpy as np

def draw_panel_grid(images, cols=4, target_height=360):
    if not images:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    def resize(img):
        return cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))

    images = [resize(img) for img in images if img is not None]
    rows = [images[i:i + cols] for i in range(0, len(images), cols)]

    grid_rows = []
    for row in rows:
        if len(row) < cols:
            blank = np.ones_like(row[0]) * 255
            row += [blank] * (cols - len(row))
        grid_rows.append(cv2.hconcat(row))

    return cv2.vconcat(grid_rows)

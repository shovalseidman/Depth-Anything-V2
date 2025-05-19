import cv2

class CameraManager:
    def __init__(self, config):
        self.cams = {}
        for cam_id, conf in config.items():
            if not isinstance(conf, dict) or 'source' not in conf:
                continue  # skip global configs like 'depth_encoder'
            cam_index = conf['source']
            cap = cv2.VideoCapture(cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cams[cam_id] = cap  # keep string key

    def read_all(self):
        frames = {}
        for cam_id, cap in self.cams.items():
            ret, frame = cap.read()
            frames[cam_id] = frame if ret else None
        return frames

    def release_all(self):
        for cap in self.cams.values():
            cap.release()

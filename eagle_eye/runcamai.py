import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import torch
import numpy as np
import cv2

# Initialize GStreamer
Gst.init(None)

# Load YOLO model (change path to your yolov11X.pt)
model = torch.hub.load('yolo11x.pt')  # Make sure yolov11X.pt is YOLOv5 format
model.conf = 0.3  # Confidence threshold
model.iou = 0.45  # IOU threshold for NMS

# Define the pipeline
pipeline = Gst.parse_launch(
    'rtspsrc location=rtsp://root:12345@192.168.1.100/stream=0 latency=0 ! '
    'rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true sync=false'
)

# Get the appsink element
appsink = pipeline.get_by_name('sink')

# This function is called whenever a new frame is available
def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR

    buf = sample.get_buffer()
    caps = sample.get_caps()
    width = caps.get_structure(0).get_value('width')
    height = caps.get_structure(0).get_value('height')

    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    # Convert to NumPy array
    frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
    buf.unmap(map_info)

    # Inference
    results = model(frame)
    results.print()

    # Optional: Display results (you can comment this for headless mode)
    rendered = results.render()[0]
    cv2.imshow("YOLOv11X Detection", rendered)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        loop.quit()

    return Gst.FlowReturn.OK

# Connect the callback
appsink.connect("new-sample", on_new_sample)

# Handle GStreamer messages
def on_message(bus, message):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print("Error:", err, debug)
        loop.quit()

# Set up the message bus
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Run the main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print("Interrupted by user")

# Cleanup
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()

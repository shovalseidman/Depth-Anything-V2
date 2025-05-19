import argparse
import cv2
import numpy as np
import os
import depthai as dai

def create_pipeline(model_path):
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.createColorCamera()
    nn = pipeline.createNeuralNetwork()
    xout_rgb = pipeline.createXLinkOut()
    xout_nn = pipeline.createXLinkOut()

    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("nn")

    # Properties
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    nn.setBlobPath(model_path)
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)

    # Linking
    cam_rgb.preview.link(nn.input)
    cam_rgb.preview.link(xout_rgb.input)
    nn.out.link(xout_nn.input)

    return pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--model-path', type=str, required=True, help='Path to the OpenVINO IR model')
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    pipeline = create_pipeline(args.model_path)
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        
        while True:
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()
            
            frame = in_rgb.getCvFrame()
            depth = np.array(in_nn.getFirstLayerFp16()).reshape((480, 640))  # Adjust shape as needed
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            if args.pred_only:
                display_frame = depth
            else:
                split_region = np.ones((frame.shape[0], 50, 3), dtype=np.uint8) * 255
                display_frame = cv2.hconcat([frame, split_region, depth])
            
            cv2.imshow('Depth Estimation', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
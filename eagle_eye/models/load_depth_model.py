# vision_pipeline/models/load_depth_model.py
import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

def load_depth_anything(encoder='vitb', device='cuda'):
    model = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model.to(device).eval()

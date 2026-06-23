"""
Texture Analysis Module
Upgraded to use a Deep Learning model (texture_model.pth) if available,
otherwise falls back to Local Binary Patterns (LBP).
"""

import os
import cv2
import numpy as np
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'app/processing/texture_model.pth'
NUM_CLASSES = 3  # 0=Smooth, 1=Fine Lines, 2=Deep Wrinkles

_texture_model = None

def get_texture_model():
    global _texture_model
    if _texture_model is None and os.path.exists(MODEL_PATH):
        try:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            _texture_model = model
            logger.info("Deep Learning Texture Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Texture Model: {e}")
    return _texture_model

# --- Pure-NumPy LBP Fallback ---
def _lbp_numpy(gray, radius=1, n_points=8):
    h, w    = gray.shape
    lbp_img = np.zeros((h, w), dtype=np.uint8)
    padded  = np.pad(gray, radius, mode='reflect').astype(np.float32)

    angles  = [2 * np.pi * p / n_points for p in range(n_points)]

    for i, angle in enumerate(angles):
        dy =  radius * np.sin(angle)
        dx = -radius * np.cos(angle)

        y_base = np.arange(radius, radius + h).astype(np.float32)
        x_base = np.arange(radius, radius + w).astype(np.float32)
        yy, xx = np.meshgrid(y_base + dy, x_base + dx, indexing='ij')

        y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
        y1, x1 = y0 + 1,                   x0 + 1

        y0 = np.clip(y0, 0, padded.shape[0]-1)
        y1 = np.clip(y1, 0, padded.shape[0]-1)
        x0 = np.clip(x0, 0, padded.shape[1]-1)
        x1 = np.clip(x1, 0, padded.shape[1]-1)

        fy, fx  = yy - np.floor(yy), xx - np.floor(xx)
        interp  = (padded[y0, x0] * (1-fy) * (1-fx) +
                   padded[y1, x0] *    fy  * (1-fx) +
                   padded[y0, x1] * (1-fy) *    fx  +
                   padded[y1, x1] *    fy  *    fx)

        center  = gray.astype(np.float32)
        lbp_img += ((interp >= center).astype(np.uint8) << i)

    return lbp_img


def analyse_texture(face_array):
    if face_array is None or face_array.size == 0:
        return {'texture_score': 50.0, 'uniformity': 0.5, 'lbp_image': None}

    model = get_texture_model()
    
    gray    = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)
    
    try:
        from skimage.feature import local_binary_pattern
        lbp_img = local_binary_pattern(gray, P=8, R=1, method='uniform').astype(np.uint8)
    except ImportError:
        lbp_img = _lbp_numpy(gray)
        
    hist, _  = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256), density=True)
    uniformity = float(np.sum(hist ** 2))
    
    lbp_norm  = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX)
    lbp_color = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_VIRIDIS)
    lbp_color = cv2.cvtColor(lbp_color, cv2.COLOR_BGR2RGB)

    if model is not None:
        # DL Path
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        pil_img = Image.fromarray(face_array)
        input_t = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_t)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            expected_severity = probs[0]*0 + probs[1]*1 + probs[2]*2
            texture_score = 100.0 - (float(expected_severity) / 2.0 * 100.0)
            
        return {
            'texture_score': round(max(0.0, min(100.0, texture_score)), 1),
            'uniformity':    round(uniformity, 4),
            'lbp_image':     lbp_color,
        }
    else:
        # Traditional LBP Path
        lbp_var  = float(np.var(lbp_img))
        MAX_VAR  = 4000.0
        texture_score = max(0.0, 100.0 - (lbp_var / MAX_VAR) * 100.0)
        return {
            'texture_score': round(texture_score, 1),
            'uniformity':    round(uniformity, 4),
            'lbp_image':     lbp_color,
        }

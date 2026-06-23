"""
Redness Detection Module
Upgraded to use a Deep Learning model (redness_model.pth) if available,
otherwise falls back to HSV color thresholding.
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
MODEL_PATH = 'app/processing/redness_model.pth'
NUM_CLASSES = 2  # 0=Clear Skin, 1=Redness/Rosacea

_redness_model = None

def get_redness_model():
    global _redness_model
    if _redness_model is None and os.path.exists(MODEL_PATH):
        try:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            _redness_model = model
            logger.info("Deep Learning Redness Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Redness Model: {e}")
    return _redness_model


def analyse_redness(face_array):
    if face_array is None or face_array.size == 0:
        return {'redness_score': 50.0, 'redness_pct': 0.0, 'redness_map': None}

    model = get_redness_model()

    bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red hue masks (wraps at 180)
    lower_red1 = np.array([0,   40,  40])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([155, 40,  40])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Exclude very dark pixels (shadow areas)
    v_channel = hsv[:, :, 2]
    bright_mask = (v_channel > 40).astype(np.uint8) * 255
    red_mask    = cv2.bitwise_and(red_mask, bright_mask)

    total_bright = np.sum(bright_mask > 0)
    red_pixels   = np.sum(red_mask > 0)
    redness_pct  = (red_pixels / (total_bright + 1e-6)) * 100.0

    # Build a smooth redness heatmap
    red_float   = red_mask.astype(np.float32) / 255.0
    red_blurred = cv2.GaussianBlur(red_float, (21, 21), 0)
    redness_map = (red_blurred * 255).astype(np.uint8)
    redness_map = cv2.applyColorMap(redness_map, cv2.COLORMAP_HOT)
    redness_map = cv2.cvtColor(redness_map, cv2.COLOR_BGR2RGB)

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
            # 0=Clear Skin, 1=Redness
            redness_prob = float(probs[1])
            redness_score = 100.0 - (redness_prob * 100.0)
            
        return {
            'redness_score': round(max(0.0, min(100.0, redness_score)), 1),
            'redness_pct':   round(redness_pct, 2),
            'redness_map':   redness_map,
        }
    else:
        # Traditional HSV Path
        redness_score = max(0.0, 100.0 - (redness_pct / 30.0) * 100.0)
        return {
            'redness_score': round(redness_score, 1),
            'redness_pct':   round(redness_pct, 2),
            'redness_map':   redness_map,
        }

def redness_by_region(face_array, regions):
    if not regions:
        return {}
    scores = {}
    for region_name, region_arr in regions.items():
        if region_arr.size > 0:
            res = analyse_redness(region_arr)
            scores[region_name] = res['redness_score']
    return scores

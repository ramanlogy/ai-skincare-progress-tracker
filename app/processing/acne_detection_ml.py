"""
Acne Detection — Approach 2 (ML / Transfer Learning)
Uses fine-tuned MobileNetV2 trained on ACNE04 dataset.
Drop-in replacement for acne_detection.py for AT3.
Returns same dict structure so scorer.py needs zero changes.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import logging

logger = logging.getLogger(__name__)

MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'acne_model.pth')
NUM_CLASSES = 4
IMG_SIZE    = 224
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Lazy-load model (same pattern as your MTCNN)
_model = None

def _get_model():
    global _model
    if _model is None:
        try:
            m = models.mobilenet_v2(pretrained=False)
            m.classifier[1] = torch.nn.Linear(m.last_channel, NUM_CLASSES)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            m.eval()
            _model = m
            logger.info("Acne ML model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load acne ML model: {e}")
            _model = None
    return _model


# Same normalisation used during training
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Severity → score mapping (0=clear=100, 3=severe=10)
SEVERITY_SCORES = {0: 100.0, 1: 72.0, 2: 44.0, 3: 16.0}
SEVERITY_LABELS = {0: 'clear', 1: 'mild', 2: 'moderate', 3: 'severe'}


def detect_acne(face_array, regions=None):
    """
    ML-based acne severity detection.
    Same signature as original detect_acne() — scorer.py unchanged.

    Args:
        face_array: numpy array (H, W, 3) RGB, preprocessed
        regions:    optional (unused — ML model sees full face)

    Returns:
        dict with acne_count, acne_score, annotated_image, method
    """
    if face_array is None or face_array.size == 0:
        return _fallback_result()

    model = _get_model()

    # If model not available, fall back to traditional CV
    if model is None:
        logger.warning("ML model unavailable — falling back to traditional CV")
        from .acne_detection import detect_acne as detect_acne_cv
        result = detect_acne_cv(face_array, regions)
        result['method'] = 'traditional_cv_fallback'
        return result

    # Convert to PIL for transforms
    pil_img = Image.fromarray(face_array.astype(np.uint8))
    tensor  = _transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs      = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        severity     = torch.argmax(probabilities).item()
        confidence   = float(probabilities[severity])

    acne_score = SEVERITY_SCORES[severity]

    # Build annotated image with severity label
    annotated = face_array.copy()
    label     = f"{SEVERITY_LABELS[severity]} ({confidence:.0%})"
    color     = [(80,200,80), (255,200,0), (255,130,0), (220,60,60)][severity]
    cv2.putText(annotated, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return {
        'acne_count':      float(severity),     # 0-3 severity level
        'acne_score':      acne_score,
        'severity_label':  SEVERITY_LABELS[severity],
        'confidence':      confidence,
        'spot_mask':       None,
        'annotated_image': annotated,
        'method':          'mobilenet_v2'
    }


def _fallback_result():
    return {
        'acne_count':      0.0,
        'acne_score':      50.0,
        'severity_label':  'unknown',
        'confidence':      0.0,
        'spot_mask':       None,
        'annotated_image': None,
        'method':          'failed'
    }

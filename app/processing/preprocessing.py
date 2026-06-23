"""
Image Preprocessing Module
Handles lighting normalisation (CLAHE), colour correction, and image
standardisation so that scans taken under different conditions are
comparable over time.
"""

import cv2
import numpy as np
from PIL import Image


def normalize_lighting(image_array):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    L-channel of the LAB colour space.  This is the recommended approach
    from the supervisor's technical deep-dive feedback.

    Args:
        image_array: numpy array (H, W, 3) in RGB
    Returns:
        numpy array (H, W, 3) in RGB with normalised lighting
    """
    bgr  = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe  = clahe.apply(l)

    lab_out  = cv2.merge((l_clahe, a, b))
    bgr_out  = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)


def apply_gray_world(image_array):
    """
    Gray-world colour constancy correction.
    Reduces the effect of coloured lighting (e.g. warm indoor bulbs).
    """
    img   = image_array.astype(np.float32)
    mean  = img.mean()
    r_mean, g_mean, b_mean = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()

    img[:,:,0] *= mean / (r_mean + 1e-6)
    img[:,:,1] *= mean / (g_mean + 1e-6)
    img[:,:,2] *= mean / (b_mean + 1e-6)

    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_face(face_image, target_size=(224, 224)):
    """
    Full preprocessing pipeline for a face image before analysis.

    Steps:
        1. Resize to target_size
        2. Gray-world colour correction
        3. CLAHE lighting normalisation
        4. Gentle Gaussian smoothing (reduce sensor noise)

    Args:
        face_image: PIL Image or numpy array
    Returns:
        numpy array (H, W, 3) uint8 RGB – ready for analysis
    """
    if isinstance(face_image, Image.Image):
        arr = np.array(face_image.convert('RGB').resize(target_size, Image.LANCZOS))
    else:
        arr = cv2.resize(face_image, target_size)

    arr = apply_gray_world(arr)
    arr = normalize_lighting(arr)

    # Very gentle denoise – preserve texture detail for LBP analysis
    arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=0.5)

    return arr


def pil_to_array(pil_image):
    return np.array(pil_image.convert('RGB'))


def array_to_pil(array):
    return Image.fromarray(array.astype(np.uint8))


def save_processed_image(array, path):
    """Save a numpy RGB array as JPEG."""
    Image.fromarray(array.astype(np.uint8)).save(path, quality=92)

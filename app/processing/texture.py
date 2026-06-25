"""
Texture Analysis Module
Implements Local Binary Patterns (LBP) as cited in the proposal:
  Ojala, T., Pietikäinen, M. and Maenpaa, T. (2002) – the LBP reference
  in The AI Skincare Progress Tracker's reference list.

LBP measures micro-texture; lower variance in the LBP histogram
correlates with smoother skin.
"""

import cv2
import numpy as np


# ── Pure-NumPy LBP (no scikit-image dependency required) ────────────────────

def _lbp_numpy(gray, radius=1, n_points=8):
    """
    Compute LBP image using a circular neighbourhood.
    Pure NumPy implementation – no extra library required.
    """
    h, w    = gray.shape
    lbp_img = np.zeros((h, w), dtype=np.uint8)
    padded  = np.pad(gray, radius, mode='reflect').astype(np.float32)

    angles  = [2 * np.pi * p / n_points for p in range(n_points)]

    # Sample neighbour offsets at sub-pixel positions via bilinear interp
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
    """
    Compute LBP-based texture score for a face image.

    Args:
        face_array: numpy array (H, W, 3) RGB, preprocessed

    Returns:
        dict:
            texture_score  (float) 0-100, where 100 = smoothest
            uniformity     (float) histogram uniformity measure
            lbp_image      (numpy array) visualisation of LBP
    """
    if face_array is None or face_array.size == 0:
        return {'texture_score': 50.0, 'uniformity': 0.5, 'lbp_image': None}

    gray    = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)

    # Try scikit-image for accuracy, fall back to numpy implementation
    try:
        from skimage.feature import local_binary_pattern
        lbp_img = local_binary_pattern(gray, P=8, R=1, method='uniform').astype(np.uint8)
    except ImportError:
        lbp_img = _lbp_numpy(gray)

    # LBP histogram (256 bins)
    hist, _  = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256), density=True)

    # Uniformity: sum of squared histogram bins (Shannon-like measure)
    uniformity = float(np.sum(hist ** 2))

    # Variance of LBP image – high variance → rough / uneven skin
    lbp_var  = float(np.var(lbp_img))

    # Normalise variance to 0-100 score (empirically calibrated)
    # Typical lbp_var range: 500-4000
    MAX_VAR  = 4000.0
    texture_score = max(0.0, 100.0 - (lbp_var / MAX_VAR) * 100.0)
    texture_score = round(texture_score, 1)

    # Colourised LBP for display
    lbp_norm  = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX)
    lbp_color = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_VIRIDIS)
    lbp_color = cv2.cvtColor(lbp_color, cv2.COLOR_BGR2RGB)

    return {
        'texture_score': texture_score,
        'uniformity':    round(uniformity, 4),
        'lbp_image':     lbp_color,
    }

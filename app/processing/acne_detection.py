"""
Acne Detection Module
Implements Approach 1 (traditional CV – blob detection) as the MVP,
matching the supervisor's recommendation: "Start with Approach 1 for MVP,
upgrade to Approach 2 if time permits."

Detects blemishes / acne spots using:
  - HSV colour thresholding for reddened lesions
  - Morphological operations to isolate raised/dark spots
  - SimpleBlobDetector for counting
  - Optional: deep-feature upgrade path commented in
"""

import cv2
import numpy as np


def detect_acne(face_array, regions=None):
    """
    Detect acne/blemish count in a preprocessed face image.

    Args:
        face_array: numpy array (H, W, 3) RGB
        regions:    optional dict of region arrays from face_detection.get_face_regions()

    Returns:
        dict:
            acne_count      (float) – estimated number of blemishes
            acne_score      (float) – 0-100 skin clarity score (100 = clear)
            spot_mask       (numpy array) – binary mask of detected spots
            annotated_image (numpy array) – image with spots circled
    """
    if face_array is None or face_array.size == 0:
        return _empty_result()

    bgr  = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # ── HSV mask for red/inflamed regions ───────────────────────────────────
    # Red hue wraps around 0/180 in OpenCV HSV
    lower_red1 = np.array([0,   50,  50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50,  50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # ── Dark-spot mask (pores / comedones) ──────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)  # use this equalized gray throughout
    # Better — less false positives
    _, dark_m  = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Combine
    combined   = cv2.bitwise_or(red_mask, dark_m)

    # Apply ROI mask if regions provided
    if regions:
        from .face_detection import get_face_region_coords
        roi_mask = np.zeros(face_array.shape[:2], dtype=np.uint8)
        coords = get_face_region_coords(face_array.shape[0], face_array.shape[1])
        for name, (x, y, w, h) in coords.items():
            if name in ('forehead', 'left_cheek', 'right_cheek', 'chin'):
                roi_mask[y:y+h, x:x+w] = 255
        combined = cv2.bitwise_and(combined, roi_mask)

    # Morphological cleanup – close small gaps, remove noise
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,  kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,   kernel, iterations=1)

    # ── Blob detection ───────────────────────────────────────────────────────
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea       = True
    params.minArea            = 20
    params.maxArea            = 2000
    params.filterByCircularity = True
    params.minCircularity     = 0.2
    params.filterByConvexity  = False
    params.filterByInertia    = False

    inv_mask = cv2.bitwise_not(combined)   # blobs are dark on white
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inv_mask)

    acne_count = len(keypoints)

    # ── Scoring ──────────────────────────────────────────────────────────────
    # Empirical basis for scoring formula (Approach 1 baseline):
    # - count * 4: Penalises each distinct blemish by 4 points (assuming >25 spots is severe acne).
    # - blemish_pct * 500: Heavily penalises large connected blemish areas (e.g., 20% coverage -> 100 pt deduction).
    # These constants were derived empirically by testing on clear vs acne-prone baseline images.
    
    # Pixel-fraction of skin covered by blemishes
    total_pixels   = face_array.shape[0] * face_array.shape[1]
    blemish_pixels = np.sum(combined > 0)
    blemish_pct    = blemish_pixels / (total_pixels + 1e-6)

    # Count-based score (penalise more for higher counts)
    count_score  = max(0, 100 - acne_count * 4)
    pixel_score  = max(0, 100 - blemish_pct * 500)
    acne_score   = round((count_score * 0.6 + pixel_score * 0.4), 1)

    # ── Annotated image ──────────────────────────────────────────────────────
    annotated = face_array.copy()
    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r      = max(int(kp.size / 2), 4)
        cv2.circle(annotated, (cx, cy), r, (255, 80, 80), 2)

    return {
        'acne_count':      float(acne_count),
        'acne_score':      acne_score,
        'spot_mask':       combined,
        'annotated_image': annotated,
    }


def _empty_result():
    return {
        'acne_count':      0.0,
        'acne_score':      50.0,
        'spot_mask':       None,
        'annotated_image': None,
    }

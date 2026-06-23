"""
Composite Scoring Module
Implements the progress score formula from the supervisor's feedback:

    Progress Score = (Acne_Improvement × 0.4) +
                     (Redness_Reduction × 0.3) +
                     (Texture_Smoothness × 0.3)

All components are normalised to 0-100 where 100 = best possible skin.
Changes are computed relative to the user's first (baseline) scan.
"""

import os
import numpy as np
from PIL import Image
import cv2
import logging

from .face_detection  import detect_and_align_face, get_face_regions
from .preprocessing   import preprocess_face
from .acne_detection_ml import detect_acne
from .redness         import analyse_redness, redness_by_region
from .texture         import analyse_texture

logger = logging.getLogger(__name__)

WEIGHTS = {'acne': 0.4, 'redness': 0.3, 'texture': 0.3}


def composite_score(acne_score, redness_score, texture_score):
    """Weighted composite skin health score (0-100)."""
    score = (acne_score    * WEIGHTS['acne']    +
             redness_score * WEIGHTS['redness'] +
             texture_score * WEIGHTS['texture'])
    return round(score, 1)


def compute_change(current, baseline):
    """Signed improvement vs baseline. Positive = improvement."""
    if baseline is None or baseline == 0:
        return 0.0
    return round(current - baseline, 1)


def analyse_image(image_path, baseline_scan=None):
    """
    Full analysis pipeline for a single image.

    Args:
        image_path:    absolute path to the uploaded image
        baseline_scan: SkinScan ORM object (first scan) for change computation,
                       or None for the very first scan

    Returns:
        dict with all scores, changes, face_detected, and status
    """
    result = {
        'face_detected':   False,
        'acne_count':      0.0,
        'acne_score':      50.0,
        'redness_score':   50.0,
        'texture_score':   50.0,
        'overall_score':   50.0,
        'acne_change':     0.0,
        'redness_change':  0.0,
        'texture_change':  0.0,
        'overall_change':  0.0,
        'analysis_status': 'failed',
        'message':         '',
    }

    # ── 1. Face detection ────────────────────────────────────────────────────
    detection = detect_and_align_face(image_path)
    if not detection['success']:
        result['message'] = detection['message']
        return result

    result['face_detected'] = True
    face_pil  = detection['face_image']

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    face_arr  = preprocess_face(face_pil, target_size=(224, 224))

    # ── 3. Get facial regions ─────────────────────────────────────────────────
    regions   = get_face_regions(face_arr)

    # ── 4. Feature analysis ──────────────────────────────────────────────────
    acne_res    = detect_acne(face_arr, regions)
    redness_res = analyse_redness(face_arr)
    
    # Incorporate per-region redness per supervisor feedback
    region_redness = redness_by_region(face_arr, regions)
    if 'left_cheek' in region_redness and 'right_cheek' in region_redness:
        # Use average cheek redness as it's more meaningful than whole-face
        avg_cheek_redness = (region_redness['left_cheek'] + region_redness['right_cheek']) / 2.0
        redness_res['redness_score'] = round(avg_cheek_redness, 1)

    texture_res = analyse_texture(face_arr)

    acne_score    = acne_res['acne_score']
    redness_score = redness_res['redness_score']
    texture_score = texture_res['texture_score']
    acne_count    = acne_res['acne_count']

    overall = composite_score(acne_score, redness_score, texture_score)

    # ── 5. Change vs baseline ────────────────────────────────────────────────
    acne_change    = 0.0
    redness_change = 0.0
    texture_change = 0.0
    overall_change = 0.0

    if baseline_scan is not None:
        acne_change    = compute_change(acne_score,    baseline_scan.acne_score)
        redness_change = compute_change(redness_score, baseline_scan.redness_score)
        texture_change = compute_change(texture_score, baseline_scan.texture_score)
        overall_change = compute_change(overall,       baseline_scan.overall_score)

    result.update({
        'acne_count':      acne_count,
        'acne_score':      acne_score,
        'redness_score':   redness_score,
        'texture_score':   texture_score,
        'overall_score':   overall,
        'acne_change':     acne_change,
        'redness_change':  redness_change,
        'texture_change':  texture_change,
        'overall_change':  overall_change,
        'analysis_status': 'complete',
        'message':         'Analysis complete.',
    })

    return result

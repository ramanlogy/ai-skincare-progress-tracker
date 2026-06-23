
import os
import cv2
import numpy as np
from PIL import Image
from math import pi
import logging

logger = logging.getLogger(__name__)

_mtcnn = None
def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            import torch
            from facenet_pytorch import MTCNN
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _mtcnn = MTCNN(keep_all=False, device=device,
                           min_face_size=40, thresholds=[0.6, 0.7, 0.7],
                           post_process=False)
            logger.info(f"MTCNN loaded on {device}")
        except Exception as e:
            logger.warning(f"MTCNN unavailable ({e}), will use OpenCV fallback")
    return _mtcnn


def detect_and_align_face(image_path, target_size=(224, 224)):
    """Detect face using MTCNN (Zhang et al.) with OpenCV Haar fallback."""
    result = dict(success=False, face_image=None, box=None,
                  landmarks=None, confidence=0.0, message='')
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        result['message'] = f'Cannot open image: {e}'
        return result

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_val < 30.0:
        result['message'] = f'Image is too blurry (score {blur_val:.0f}<30). Please hold the camera steady.'
        return result

    mtcnn = _get_mtcnn()
    if mtcnn is not None:
        try:
            boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
            if boxes is not None and len(boxes):
                conf = float(probs[0])
                x1, y1, x2, y2 = boxes[0]
                w, h = img.size
                px, py = (x2 - x1) * .10, (y2 - y1) * .10
                x1 = max(0, x1 - px); y1 = max(0, y1 - py)
                x2 = min(w, x2 + px); y2 = min(h, y2 + py)
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                scale_x = target_size[0] / crop_w
                scale_y = target_size[1] / crop_h
                
                shifted_landmarks = []
                if landmarks is not None:
                    for lx, ly in landmarks[0]:
                        sx = (lx - x1) * scale_x
                        sy = (ly - y1) * scale_y
                        shifted_landmarks.append([sx, sy])
                
                face = img.crop((x1, y1, x2, y2)).resize(target_size, Image.LANCZOS)
                result.update(success=True, face_image=face,
                              box=[float(x1), float(y1), float(x2), float(y2)],
                              landmarks=shifted_landmarks if shifted_landmarks else None,
                              confidence=conf,
                              message=f'Face detected (confidence {conf:.2f})')
                return result
            result['message'] = 'No face found - ensure good lighting and a clear frontal view.'
            return result
        except Exception as e:
            logger.warning(f"MTCNN failed: {e}, trying OpenCV Haar cascade")

    # ── OpenCV Haar Cascade fallback ────────────────────────────────
    try:
        casc  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = casc.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces):
            x, y, fw, fh = faces[0]
            pad = int(min(fw, fh) * .1)
            ih, iw = cv_img.shape[:2]
            x1 = max(0, x - pad);       y1 = max(0, y - pad)
            x2 = min(iw, x + fw + pad); y2 = min(ih, y + fh + pad)
            face = img.crop((x1, y1, x2, y2)).resize(target_size, Image.LANCZOS)
            result.update(success=True, face_image=face,
                          box=[float(x1), float(y1), float(x2), float(y2)],
                          confidence=0.75,
                          message='Face detected via OpenCV cascade')
        else:
            result['message'] = ('No face detected. Ensure good natural lighting '
                                 'and a clear frontal view.')
    except Exception as e:
        result['message'] = f'Detection error: {e}'
    return result


def get_skin_mask(target_size, landmarks):
    """Creates a strict binary mask for skin regions using facial landmarks."""
    mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    
    if landmarks is None or len(landmarks) < 5:
        cv2.ellipse(mask, (target_size[0]//2, target_size[1]//2), 
                    (int(target_size[0]*0.4), int(target_size[1]*0.45)), 
                    0, 0, 360, 255, -1)
        return mask

    le, re, n, ml, mr = landmarks
    eye_dist = np.linalg.norm(np.array(le) - np.array(re))
    center_x = int((le[0] + re[0]) / 2)
    center_y = int((le[1] + re[1]) / 2 + eye_dist * 0.4)
    
    axes = (int(eye_dist * 1.4), int(eye_dist * 1.8))
    cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
    
    eye_radius = int(eye_dist * 0.35)
    cv2.circle(mask, (int(le[0]), int(le[1])), eye_radius, 0, -1)
    cv2.circle(mask, (int(re[0]), int(re[1])), eye_radius, 0, -1)
    
    mouth_w = int(np.linalg.norm(np.array(ml) - np.array(mr)) / 2 * 1.2)
    mouth_h = int(eye_dist * 0.3)
    mouth_cx = int((ml[0] + mr[0]) / 2)
    mouth_cy = int((ml[1] + mr[1]) / 2)
    cv2.ellipse(mask, (mouth_cx, mouth_cy), (mouth_w, mouth_h), 0, 0, 360, 0, -1)
    
    return mask


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 - PREPROCESSING  (CLAHE + Grey-World)
# ═══════════════════════════════════════════════════════════════════

def _grey_world(arr):
    img  = arr.astype(np.float32)
    mean = img.mean()
    for c in range(3):
        cm = img[:, :, c].mean()
        img[:, :, c] *= mean / (cm + 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


def _clahe(arr):
    """CLAHE on L-channel of LAB - supervisor's recommended approach."""
    bgr      = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab      = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    l        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    bgr2     = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)


def preprocess_face(face_image, mask, target_size=(224, 224)):
    if isinstance(face_image, Image.Image):
        arr = np.array(face_image.convert('RGB').resize(target_size, Image.LANCZOS))
    else:
        arr = cv2.resize(face_image, target_size)
    
    arr = arr.astype(np.float32)
    for c in range(3):
        channel = arr[:, :, c]
        mean_val = np.mean(channel[mask > 0]) if np.any(mask > 0) else channel.mean()
        channel *= (128.0 / (mean_val + 1e-6))
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    bgr2 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    arr = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    
    # Stronger blur to remove webcam sensor noise before analysis
    arr = cv2.GaussianBlur(arr, (5, 5), sigmaX=1.0)
    return arr


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 - ACNE DETECTION  (Blob detection + HSV)
# ═══════════════════════════════════════════════════════════════════

def detect_acne(face_arr, mask):
    """Approach 2: Deep Learning (MobileNetV2) for Acne Detection"""
    try:
        # Import the new ML model function
        from app.processing.acne_detection_ml import detect_acne as detect_acne_ml
        
        # The ML function signature accepts (face_array, regions)
        # We pass mask as regions (it's ignored by the ML model anyway)
        result = detect_acne_ml(face_arr, mask)
        
        # If the ML model successfully ran
        if result.get('method') == 'mobilenet_v2':
            return dict(
                acne_count=result['acne_count'],
                acne_score=result['acne_score'],
                spot_mask=result['spot_mask'],
                annotated_image=result['annotated_image']
            )
    except Exception as e:
        logger.error(f"Error using ML acne detection: {e}")

    # --- FALLBACK TO TRADITIONAL CV ---
    if face_arr is None or face_arr.size == 0:
        return dict(acne_count=0.0, acne_score=50.0,
                    spot_mask=None, annotated_image=None)

    bgr = cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    m1 = cv2.inRange(hsv_blurred, np.array([0,   45,  50]), np.array([10,  255, 255]))
    m2 = cv2.inRange(hsv_blurred, np.array([160, 45,  50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)

    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 15, 12)
    combined = cv2.bitwise_or(red_mask, dm)
    combined = cv2.bitwise_and(combined, mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)

    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea       = True;  p.minArea = 35;  p.maxArea = 2000
    p.filterByCircularity = True; p.minCircularity = 0.2
    p.filterByConvexity  = False; p.filterByInertia = False
    kps = cv2.SimpleBlobDetector_create(p).detect(cv2.bitwise_not(combined))

    count = len(kps)
    bp = np.sum(combined > 0) / (np.sum(mask > 0) + 1e-6)
    
    score = round(max(0.0, 100.0 - count * 2.5) * 0.7 +
                  max(0.0, 100.0 - bp * 300)  * 0.3, 1)

    ann = face_arr.copy()
    for kp in kps:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(ann, (cx, cy), max(int(kp.size / 2), 4), (255, 80, 80), 2)

    return dict(acne_count=float(count), acne_score=score,
                spot_mask=combined, annotated_image=ann)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 - REDNESS ANALYSIS  (HSV hue range)
# ═══════════════════════════════════════════════════════════════════

def analyse_redness(face_arr, mask):
    if face_arr is None or face_arr.size == 0:
        return dict(redness_score=50.0, redness_pct=0.0, redness_map=None)

    bgr = cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    m1 = cv2.inRange(hsv_blurred, np.array([0,   35, 40]), np.array([12,  255, 255]))
    m2 = cv2.inRange(hsv_blurred, np.array([155, 35, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(m1, m2)
    red_mask = cv2.bitwise_and(red_mask, mask)

    bright   = (hsv_blurred[:, :, 2] > 40).astype(np.uint8) * 255
    red_mask = cv2.bitwise_and(red_mask, bright)

    redness_pct = np.sum(red_mask > 0) / (np.sum(cv2.bitwise_and(bright, mask) > 0) + 1e-6) * 100
    redness_score = round(max(0.0, 100.0 - (redness_pct / 30.0) * 100.0), 1)

    blurred = cv2.GaussianBlur((red_mask / 255.0).astype(np.float32), (21, 21), 0)
    hmap = cv2.cvtColor(cv2.applyColorMap((blurred * 255).astype(np.uint8), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)

    return dict(redness_score=redness_score,
                redness_pct=round(float(redness_pct), 2),
                redness_map=hmap)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 - TEXTURE ANALYSIS  (LBP - Ojala et al., 2002)
# ═══════════════════════════════════════════════════════════════════

def _lbp_numpy(gray, radius=1, n_points=8):
    """Pure-NumPy LBP - no scikit-image required."""
    h, w    = gray.shape
    lbp     = np.zeros((h, w), dtype=np.uint8)
    padded  = np.pad(gray, radius, mode='reflect').astype(np.float32)
    center  = gray.astype(np.float32)
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        dy    =  radius * np.sin(angle)
        dx    = -radius * np.cos(angle)
        yy, xx = np.meshgrid(
            np.arange(radius, radius + h, dtype=np.float32) + dy,
            np.arange(radius, radius + w, dtype=np.float32) + dx,
            indexing='ij')
        y0 = np.clip(np.floor(yy).astype(int), 0, padded.shape[0] - 1)
        x0 = np.clip(np.floor(xx).astype(int), 0, padded.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, padded.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, padded.shape[1] - 1)
        fy, fx = yy - np.floor(yy), xx - np.floor(xx)
        interp = (padded[y0, x0] * (1 - fy) * (1 - fx) +
                  padded[y1, x0] *      fy   * (1 - fx) +
                  padded[y0, x1] * (1 - fy)  *      fx  +
                  padded[y1, x1] *      fy   *      fx)
        lbp += ((interp >= center).astype(np.uint8) << i)
    return lbp


def analyse_texture(face_arr, mask):
    if face_arr is None or face_arr.size == 0:
        return dict(texture_score=50.0, uniformity=0.5, lbp_image=None)

    gray = cv2.cvtColor(face_arr, cv2.COLOR_RGB2GRAY)
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform').astype(np.uint8)
    except ImportError:
        lbp = _lbp_numpy(gray)

    lbp_masked = lbp[mask > 0]
    if len(lbp_masked) == 0:
        return dict(texture_score=50.0, uniformity=0.5, lbp_image=None)

    hist, _ = np.histogram(lbp_masked, bins=256, range=(0, 256), density=True)
    uniformity = float(np.sum(hist ** 2))
    var        = float(np.var(lbp_masked))
    # Adjusted scaling factor so texture doesn't always max out at 99
    score      = round(max(0.0, 100.0 - (var / 1500.0) * 100.0), 1)

    norm  = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
    norm[mask == 0] = 0
    lbp_c = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    return dict(texture_score=score,
                uniformity=round(uniformity, 4),
                lbp_image=lbp_c)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 - COMPOSITE SCORER
# ═══════════════════════════════════════════════════════════════════

WEIGHTS = dict(acne=0.4, redness=0.3, texture=0.3)

def composite_score(acne, redness, texture):
    """Supervisor's recommended weighted formula."""
    return round(acne * WEIGHTS['acne'] +
                 redness * WEIGHTS['redness'] +
                 texture * WEIGHTS['texture'], 1)


def compute_change(current, baseline):
    if baseline is None or baseline == 0:
        return 0.0
    return round(current - baseline, 1)


def analyse_image(image_path, baseline_scan=None):
    """Full pipeline: detect -> preprocess -> analyse -> score -> compare."""
    r = dict(face_detected=False, acne_count=0.0, acne_score=50.0,
             redness_score=50.0, texture_score=50.0, overall_score=50.0,
             acne_change=0.0, redness_change=0.0, texture_change=0.0,
             overall_change=0.0, analysis_status='failed', message='')

    det = detect_and_align_face(image_path)
    if not det['success']:
        r['message'] = det['message']
        return r

    r['face_detected'] = True
    mask = get_skin_mask((224, 224), det['landmarks'])
    face_arr = preprocess_face(det['face_image'], mask)

    ar = detect_acne(face_arr, mask)
    rr = analyse_redness(face_arr, mask)
    tr = analyse_texture(face_arr, mask)

    a_s     = ar['acne_score']
    red_s   = rr['redness_score']
    tex_s   = tr['texture_score']
    overall = composite_score(a_s, red_s, tex_s)

    ac = rc = tc = oc = 0.0
    if baseline_scan is not None:
        ac = compute_change(a_s,    baseline_scan.acne_score)
        rc = compute_change(red_s,  baseline_scan.redness_score)
        tc = compute_change(tex_s,  baseline_scan.texture_score)
        oc = compute_change(overall, baseline_scan.overall_score)

    r.update(acne_count=ar['acne_count'], acne_score=a_s,
             redness_score=red_s, texture_score=tex_s,
             overall_score=overall,
             acne_change=ac, redness_change=rc,
             texture_change=tc, overall_change=oc,
             analysis_status='complete', message='Analysis complete.')
    return r


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 - FLASK APP + MODELS
# ═══════════════════════════════════════════════════════════════════

from flask import (Flask, render_template_string, request, redirect,
                   url_for, flash, jsonify, abort, send_from_directory)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config.update(
    SECRET_KEY              = 'The AI Skincare Progress Tracker-skincare-secret-2026',
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DB_PATH}',
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    UPLOAD_FOLDER           = UPLOAD_DIR,
    MAX_CONTENT_LENGTH      = 16 * 1024 * 1024,
)

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view             = 'login_page'
login_manager.login_message          = 'Please log in to continue.'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))



#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   The AI Skincare Progress Tracker                          ║
║   COM668 Computing Project · Student ID: B00912171              ║
║   Project Title: AI Skincare Progress Tracker                   ║
║                                                                  ║
║   SINGLE FILE - run with:  python run.py                        ║
║   Then open:               http://127.0.0.1:5000                ║
╚══════════════════════════════════════════════════════════════════╝

Install dependencies first (one-time):
    pip install flask flask-sqlalchemy flask-login werkzeug \
                opencv-python-headless Pillow numpy scipy \
                scikit-image scikit-learn torch torchvision \
                facenet-pytorch matplotlib pandas
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS & SETUP
# ═══════════════════════════════════════════════════════════════════

import os, uuid, logging
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DB_PATH    = os.path.join(BASE_DIR, 'skincare.db')
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('The AI Skincare Progress Tracker')


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 - FACE DETECTION  (Zhang et al., 2016 - MTCNN)
# ═══════════════════════════════════════════════════════════════════

_mtcnn = None

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            import torch
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


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    consent_given = db.Column(db.Boolean,  default=False)
    scans         = db.relationship('SkinScan', backref='user', lazy=True,
                                    cascade='all, delete-orphan')

    def set_password(self, p):
        self.password_hash = generate_password_hash(p)

    def check_password(self, p):
        return check_password_hash(self.password_hash, p)


class SkinScan(db.Model):
    __tablename__   = 'skin_scans'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename  = db.Column(db.String(256), nullable=False)
    captured_at     = db.Column(db.DateTime, default=datetime.utcnow)
    notes           = db.Column(db.Text, default='')
    face_detected   = db.Column(db.Boolean, default=False)
    acne_count      = db.Column(db.Float, default=0.0)
    acne_score      = db.Column(db.Float, default=50.0)
    redness_score   = db.Column(db.Float, default=50.0)
    texture_score   = db.Column(db.Float, default=50.0)
    overall_score   = db.Column(db.Float, default=50.0)
    acne_change     = db.Column(db.Float, default=0.0)
    redness_change  = db.Column(db.Float, default=0.0)
    texture_change  = db.Column(db.Float, default=0.0)
    overall_change  = db.Column(db.Float, default=0.0)
    analysis_status = db.Column(db.String(20), default='pending')

    def to_dict(self):
        return dict(
            id=self.id,
            image_filename=self.image_filename,
            captured_at=self.captured_at.strftime('%Y-%m-%d %H:%M'),
            notes=self.notes,
            face_detected=self.face_detected,
            acne_count=round(self.acne_count, 1),
            acne_score=round(self.acne_score, 1),
            redness_score=round(self.redness_score, 1),
            texture_score=round(self.texture_score, 1),
            overall_score=round(self.overall_score, 1),
            acne_change=round(self.acne_change, 1),
            redness_change=round(self.redness_change, 1),
            texture_change=round(self.texture_change, 1),
            overall_change=round(self.overall_change, 1),
            analysis_status=self.analysis_status,
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 - CSS (shared across all pages)
# ═══════════════════════════════════════════════════════════════════

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700&display=swap');

:root {
  --bg-main:    #f8fafc;
  --bg-card:    #ffffff;
  --bg-subtle:  #f1f5f9;
  --primary:    #0ea5e9;
  --primary-hover: #0284c7;
  --primary-light: #e0f2fe;
  --text-main:  #0f172a;
  --text-muted: #64748b;
  --border:     #e2e8f0;
  --success:    #10b981;
  --warning:    #f59e0b;
  --danger:     #ef4444;
  --danger-light: #fee2e2;
  --radius:     16px;
  --radius-sm:  8px;
  --shadow:     0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  --shadow-lg:  0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --transition: all 0.2s ease-in-out;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; -webkit-font-smoothing: antialiased; }
body { font-family: 'Inter', sans-serif; background: var(--bg-main);
       color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }
h1,h2,h3,h4 { font-family: 'Outfit', sans-serif; font-weight: 600; line-height: 1.2; letter-spacing: -0.02em; }
a { color: var(--primary); text-decoration: none; transition: var(--transition); }
a:hover { color: var(--primary-hover); }
img { max-width: 100%; display: block; }
main { flex: 1; }

.navbar { background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 2rem; height: 64px; position: sticky; top: 0; z-index: 100; }
.nav-brand { font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: .5rem; letter-spacing: -0.01em; }
.nav-brand:hover { color: var(--primary); }
.brand-mark { color: var(--primary); display: flex; }
.nav-links { display: flex; gap: .25rem; margin-left: 2.5rem; }
.nav-links a { font-size: .875rem; font-weight: 500; color: var(--text-muted); padding: .5rem 1rem; border-radius: var(--radius-sm); transition: var(--transition); }
.nav-links a:hover, .nav-links a.active { background: var(--bg-subtle); color: var(--primary); }
.nav-user { margin-left: auto; display: flex; align-items: center; gap: 1rem; }
.user-chip { font-size: .875rem; font-weight: 500; background: var(--primary-light); padding: .35rem .85rem; border-radius: 20px; color: var(--primary-hover); display: flex; align-items: center; gap: .4rem; }
.btn-ghost-sm { font-size: .875rem; font-weight: 500; color: var(--text-muted); padding: .35rem .75rem; border-radius: var(--radius-sm); border: 1px solid var(--border); transition: var(--transition); display: flex; align-items: center; gap: .4rem; }
.btn-ghost-sm:hover { background: var(--bg-subtle); color: var(--text-main); }

.flash-container { position: fixed; top: 80px; right: 1.5rem; z-index: 200; display: flex; flex-direction: column; gap: .5rem; max-width: 380px; }
.flash { padding: 1rem 1.25rem; border-radius: var(--radius-sm); font-size: .875rem; font-weight: 500; display: flex; align-items: flex-start; justify-content: space-between; gap: .75rem; box-shadow: var(--shadow-lg); animation: slideIn .3s cubic-bezier(0.16, 1, 0.3, 1); line-height: 1.4; }
@keyframes slideIn { from { opacity:0; transform:translateX(20px); } to { opacity:1; transform:translateX(0); } }
.flash-success { background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; }
.flash-error   { background:#fef2f2; border:1px solid #fecaca; color:#991b1b; }
.flash-warning { background:#fffbeb; border:1px solid #fde68a; color:#92400e; }
.flash-info    { background:#eff6ff; border:1px solid #bfdbfe; color:#1e40af; }
.flash-close { background:none; border:none; cursor:pointer; color:inherit; opacity:.5; line-height:1; flex-shrink:0; display:flex; }
.flash-close:hover { opacity:1; }

.pc { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.5rem 4rem; }
.pc--n { max-width: 720px; }
.ph { display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }
.pt { font-size: 2.25rem; letter-spacing: -0.03em; color: var(--text-main); }
.ps { font-size: 1rem; color: var(--text-muted); margin-top: .4rem; }

.btn-primary { background: var(--primary); color: #fff; border: none; padding: .75rem 1.5rem; border-radius: var(--radius-sm); font-family: 'Inter', sans-serif; font-size: .875rem; font-weight: 500; cursor: pointer; transition: var(--transition); display: inline-flex; align-items: center; justify-content: center; gap: .5rem; }
.btn-primary:hover { background: var(--primary-hover); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25); color: #fff; }
.btn-primary.fw { width:100%; padding: .875rem; }
.btn-primary:disabled { background: var(--text-muted); cursor: not-allowed; transform: none; box-shadow: none; opacity: 0.7; }
.btn-secondary { background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border); padding: .75rem 1.5rem; border-radius: var(--radius-sm); font-family: 'Inter', sans-serif; font-size: .875rem; font-weight: 500; cursor: pointer; transition: var(--transition); display: inline-flex; align-items: center; justify-content: center; gap: .5rem; }
.btn-secondary:hover { background: var(--bg-subtle); border-color: var(--text-muted); }
.btn-ghost { background: transparent; color: var(--text-muted); border: 1px solid transparent; padding: .75rem 1.5rem; border-radius: var(--radius-sm); font-family: 'Inter', sans-serif; font-size: .875rem; font-weight: 500; cursor: pointer; transition: var(--transition); display: inline-flex; align-items: center; gap: .5rem;}
.btn-ghost:hover { background: var(--bg-subtle); color: var(--text-main); }
.btn-danger { background: var(--danger-light); color: var(--danger); border: 1px solid #fecaca; padding: .75rem 1.5rem; border-radius: var(--radius-sm); font-family: 'Inter', sans-serif; font-size: .875rem; font-weight: 500; cursor: pointer; transition: var(--transition); display: inline-flex; align-items: center; gap: .5rem; }
.btn-danger:hover { background: #fee2e2; border-color: var(--danger); }
.btn-dsm { background: none; border: 1px solid #fecaca; color: var(--danger); font-size: .75rem; font-weight: 500; padding: .35rem .75rem; border-radius: 6px; cursor: pointer; font-family: 'Inter', sans-serif; transition: var(--transition); display: inline-flex; align-items: center; gap: .3rem; }
.btn-dsm:hover { background: var(--danger-light); }

.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 2rem; box-shadow: var(--shadow); margin-bottom: 1.5rem; }
.ct { font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 600; color: var(--text-main); margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: .5rem; }

.sg { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
@media(max-width:900px){ .sg{ grid-template-columns: repeat(2, 1fr); } }
@media(max-width:560px){ .sg{ grid-template-columns: 1fr; } }
.sc { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; box-shadow: var(--shadow); display: flex; flex-direction: column; gap: .75rem; }
.sc--o { background: var(--primary); border-color: var(--primary); color: #fff; align-items: center; }
.sl { font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); display: flex; align-items: center; gap: .4rem; }
.sc--o .sl { color: var(--primary-light); }
.sring { position: relative; width: 88px; height: 88px; }
.rsv { width: 88px; height: 88px; }
.rv { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 600; color: #fff; }
.bscore { display: flex; align-items: center; gap: 1rem; }
.btrack { flex: 1; height: 8px; background: var(--bg-subtle); border-radius: 4px; overflow: hidden; }
.bfill { height: 100%; border-radius: 4px; transition: width 1s cubic-bezier(0.16, 1, 0.3, 1); }
.bnum { font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 600; min-width: 2.5rem; text-align: right; }
.schg { font-size: .875rem; font-weight: 500; display: flex; align-items: center; gap: .25rem; }
.schg.pos { color: var(--success); } .schg.neg { color: var(--danger); }
.sc--o .schg.pos { color: #a7f3d0; } .sc--o .schg.neg { color: #fecaca; }

.sr2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
@media(max-width:700px){ .sr2{ grid-template-columns: 1fr; } }
.sprev { display: flex; gap: 1.5rem; align-items: flex-start; }
.simg  { width: 140px; height: 140px; object-fit: cover; border-radius: var(--radius-sm); border: 1px solid var(--border); flex-shrink: 0; }
.smeta { font-size: .875rem; line-height: 1.6; color: var(--text-muted); display: flex; flex-direction: column; gap: .5rem; }
.smeta p { display: flex; align-items: flex-start; gap: .5rem; margin: 0; }
.smeta strong { color: var(--text-main); font-weight: 500; width: 90px; flex-shrink: 0;}

.compare-row { display: flex; align-items: center; gap: 1.5rem; justify-content: center; }
.compare-col { text-align: center; }
.compare-label { font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-bottom: .75rem; }
.compare-img { width: 110px; height: 110px; object-fit: cover; border-radius: var(--radius-sm); border: 1px solid var(--border); box-shadow: var(--shadow); }
.compare-date { font-size: .875rem; font-weight: 500; color: var(--text-main); margin-top: .75rem; }
.compare-arrow { color: var(--text-muted); opacity: 0.5; }

.chtabs { display: flex; gap: .5rem; margin-bottom: 1.5rem; flex-wrap: wrap; background: var(--bg-subtle); padding: .35rem; border-radius: var(--radius-sm); width: fit-content; }
.tb { background: transparent; border: none; color: var(--text-muted); padding: .5rem 1rem; border-radius: 6px; font-family: 'Inter', sans-serif; font-size: .875rem; font-weight: 500; cursor: pointer; transition: var(--transition); display: flex; align-items: center; gap: .4rem; }
.tb:hover { color: var(--text-main); }
.tb.active { background: var(--bg-card); color: var(--primary); box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.chtwrap canvas { max-height: 300px; width: 100%; }

.badge { display: inline-flex; align-items: center; gap: .3rem; font-size: .75rem; font-weight: 600; padding: .25rem .75rem; border-radius: 20px; text-transform: capitalize;}
.badge--complete { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0;}
.badge--pending  { background: #fffbeb; color: #d97706; border: 1px solid #fde68a;}
.badge--failed   { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca;}

.empty-state { text-align: center; padding: 5rem 1rem; color: var(--text-muted); background: var(--bg-card); border-radius: var(--radius); border: 1px dashed var(--border); }
.empty-icon  { color: var(--text-muted); margin-bottom: 1.5rem; opacity: 0.5; display: flex; justify-content: center; }
.empty-state h2 { font-size: 1.5rem; color: var(--text-main); margin-bottom: .75rem; }
.empty-state p  { margin-bottom: 2rem; font-size: 1rem; max-width: 400px; margin-inline: auto; }

.auth-layout { min-height: 100vh; display: grid; grid-template-columns: 1.2fr 1fr; }
@media(max-width:900px){ .auth-layout{ grid-template-columns: 1fr; } .auth-hero{ display:none; } }
.auth-hero { background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); display: flex; align-items: center; justify-content: center; padding: 4rem; position: relative; overflow: hidden; }
.auth-hero::before { content: ''; position: absolute; inset: 0; background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4xKSIvPjwvc3ZnPg==') repeat; opacity: 0.5; }
.hero-inner  { position: relative; z-index: 1; color: #fff; max-width: 480px; }
.hero-eyebrow { font-size: .875rem; font-weight: 600; text-transform: uppercase; letter-spacing: .1em; color: var(--primary-light); margin-bottom: 1.5rem; display: flex; align-items: center; gap: .5rem; }
.hero-title { font-family: 'Outfit', sans-serif; font-size: 4rem; font-weight: 700; color: #fff; line-height: 1.1; margin-bottom: 1.5rem; letter-spacing: -0.03em; }
.hero-sub  { font-size: 1.125rem; color: var(--primary-light); line-height: 1.6; margin-bottom: 3rem; font-weight: 400; }
.hero-feats { display: flex; flex-direction: column; gap: 1.25rem; }
.feat { display: flex; align-items: center; gap: 1rem; font-size: 1rem; font-weight: 500; color: #fff; }
.feat-icon { color: var(--primary-light); display: flex; }
.auth-forms { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem 2rem; background: var(--bg-main); }
.form-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 3rem; width: 100%; max-width: 440px; box-shadow: var(--shadow-lg); }
.form-title { font-family: 'Outfit', sans-serif; font-size: 1.875rem; font-weight: 600; margin-bottom: .5rem; color: var(--text-main); }
.form-sub   { font-size: 1rem; color: var(--text-muted); margin-bottom: 2rem; }
.form-switch { font-size: .875rem; color: var(--text-muted); text-align: center; margin-top: 1.5rem; }
.form-switch a { color: var(--primary); font-weight: 500; }
.disc-mini { font-size: .75rem; color: var(--text-muted); text-align: center; max-width: 360px; margin-top: 2rem; line-height: 1.6; }

.field { margin-bottom: 1.25rem; }
.field label { display: flex; justify-content: space-between; align-items: baseline; font-size: .875rem; font-weight: 500; color: var(--text-main); margin-bottom: .5rem; }
.opt { font-weight: 400; color: var(--text-muted); font-size: .75rem; }
.field input, .field textarea { width: 100%; padding: .75rem 1rem; border: 1px solid var(--border); border-radius: var(--radius-sm); font-family: 'Inter', sans-serif; font-size: .95rem; background: var(--bg-main); color: var(--text-main); transition: var(--transition); }
.field input:focus, .field textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-light); background: var(--bg-card); }
.consent-box { background: var(--bg-subtle); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 1rem; margin-bottom: 1.5rem; }
.consent-label { display: flex; gap: .75rem; align-items: flex-start; font-size: .875rem; color: var(--text-main); line-height: 1.5; cursor: pointer; }
.consent-label input[type=checkbox] { width: 1.1rem; height: 1.1rem; margin-top: .15rem; flex-shrink: 0; accent-color: var(--primary); cursor: pointer; }
.consent-reminder { font-size: .875rem; color: var(--text-muted); background: var(--bg-subtle); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 1rem; margin-bottom: 1.5rem; line-height: 1.5; display: flex; gap: .75rem; align-items: flex-start; }

.upload-zone { border: 2px dashed var(--border); border-radius: var(--radius); background: var(--bg-subtle); margin-bottom: 1.5rem; transition: var(--transition); min-height: 240px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
.upload-zone:hover, .upload-zone.drag-over { border-color: var(--primary); background: var(--primary-light); }
#uploadPlaceholder { text-align: center; padding: 3rem; cursor: pointer; width: 100%; }
.upload-icon { color: var(--primary); margin-bottom: 1rem; display: flex; justify-content: center; }
.upload-cta  { font-size: 1.125rem; font-weight: 600; color: var(--text-main); margin-bottom: .4rem; }
.upload-hint { font-size: .875rem; color: var(--text-muted); }
.preview-image  { width: 100%; max-height: 400px; object-fit: contain; border-radius: var(--radius-sm); }
.preview-actions { padding: 1rem; text-align: center; border-top: 1px solid var(--border); background: var(--bg-card); display: flex; gap: 1rem; justify-content: center; }

.tips-bar { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.tip { display: flex; align-items: center; gap: .5rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: 24px; padding: .5rem 1rem; font-size: .875rem; font-weight: 500; color: var(--text-main); box-shadow: var(--shadow); }
.tip i { color: var(--primary); width: 16px; height: 16px; }

.info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 2rem; }
@media(max-width:768px){ .info-grid{ grid-template-columns: 1fr; } }
.info-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; box-shadow: var(--shadow); }
.info-card h4 { font-size: 1rem; color: var(--text-main); margin-bottom: .75rem; display: flex; align-items: center; gap: .5rem; }
.info-card h4 i { color: var(--primary); width: 20px; height: 20px;}
.info-card p  { font-size: .875rem; color: var(--text-muted); line-height: 1.6; }

.history-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem; }
.hcard { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow); transition: var(--transition); }
.hcard:hover { box-shadow: var(--shadow-lg); transform: translateY(-3px); }
.hiw { position: relative; }
.himg { width: 100%; height: 180px; object-fit: cover; display: block; border-bottom: 1px solid var(--border); }
.hbadge { position: absolute; top: .75rem; right: .75rem; box-shadow: var(--shadow); }
.hbody { padding: 1.25rem; }
.hdate { font-size: .875rem; font-weight: 500; color: var(--text-main); margin-bottom: 1rem; display: flex; align-items: center; gap: .4rem; }
.hdate i { color: var(--text-muted); width: 16px; height: 16px;}
.hnote { font-size: .875rem; color: var(--danger); margin-top: .5rem; display: flex; align-items: flex-start; gap: .4rem; }
.hnotes { font-size: .875rem; color: var(--text-muted); font-style: italic; margin-top: .75rem; border-top: 1px dashed var(--border); padding-top: .75rem; display: flex; gap: .4rem; }
.hnotes i { flex-shrink:0; width:16px; height:16px; margin-top:.1rem; }
.hchg { font-size: .875rem; font-weight: 500; margin-top: .75rem; display: flex; align-items: center; gap: .3rem; }
.hchg.pos { color: var(--success); } .hchg.neg { color: var(--danger); }
.del-form { margin-top: 1.25rem; }
.mini-scores { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; }
.mini-score  { background: var(--bg-subtle); border-radius: var(--radius-sm); padding: .5rem; display: flex; flex-direction: column; gap: .25rem; }
.ms-label { font-size: .7rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; }
.ms-val   { font-family: 'Outfit', sans-serif; font-size: 1.125rem; font-weight: 600; color: var(--text-main); }

.data-rights { background: var(--bg-card); }
.data-rights h3 { font-size: 1.25rem; margin-bottom: .75rem; display: flex; align-items: center; gap: .5rem; }
.data-rights h3 i { color: var(--text-muted); }
.data-rights p  { font-size: .95rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 1.5rem; max-width: 700px; }

.site-footer { border-top: 1px solid var(--border); background: var(--bg-card); padding: 2rem; text-align: center; margin-top: auto; }
.disclaimer  { font-size: .875rem; font-weight: 500; color: var(--text-main); background: var(--bg-subtle); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 1rem 1.5rem; max-width: 800px; margin: 0 auto 1.5rem; line-height: 1.6; display: flex; align-items: flex-start; gap: .75rem; text-align: left; }
.disclaimer i { color: var(--warning); flex-shrink: 0; margin-top: .1rem; }
.footer-copy { font-size: .875rem; color: var(--text-muted); }
"""

# ── Shared JS ────────────────────────────────────────────────────────────────
_JS = """
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) { lucide.createIcons(); }
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(20px)';
      setTimeout(() => el.remove(), 300);
    }, 5000);
  });
  document.querySelectorAll('.bfill').forEach(b => {
    const t = b.style.width;
    b.style.width = '0';
    requestAnimationFrame(() => setTimeout(() => { b.style.width = t; }, 100));
  });
  document.querySelectorAll('.rsv circle:last-child').forEach(c => {
    const t = parseFloat(c.getAttribute('stroke-dashoffset') || '0');
    c.setAttribute('stroke-dashoffset', '314');
    c.style.transition = 'stroke-dashoffset 1s cubic-bezier(0.16, 1, 0.3, 1)';
    setTimeout(() => c.setAttribute('stroke-dashoffset', t), 200);
  });
});
"""

# ── Base wrapper ─────────────────────────────────────────────────────────────
def _base(title, active, head_extra, content, scripts_extra=''):
    nav = ''
    if True:  # always render nav structure (Jinja will hide if not auth)
        nav = f"""
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark"><i data-lucide="activity"></i></span> The AI Skincare Progress Tracker</a>
  <div class="nav-links">
    <a href="/dashboard" {'class="active"' if active=='dash' else ''}>Dashboard</a>
    <a href="/capture"   {'class="active"' if active=='cap'  else ''}>New Scan</a>
    <a href="/history"   {'class="active"' if active=='hist' else ''}>History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{% if current_user.is_authenticated %}}{{{{ current_user.username }}}}{{% endif %}}</span>
    <a href="/logout" class="btn-ghost-sm"><i data-lucide="log-out"></i> Log out</a>
  </div>
</nav>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{_CSS}</style>
  {head_extra}
  <script src="https://unpkg.com/lucide@latest"></script>\n</head>
<body>
{{% if current_user.is_authenticated %}}
{nav}
{{% endif %}}
<div class="flash-container">
  {{% with messages = get_flashed_messages(with_categories=true) %}}
  {{% for cat, msg in messages %}}
  <div class="flash flash-{{{{ cat }}}}">{{{{ msg }}}}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {{% endfor %}}{{% endwith %}}
</div>
<main>{content}</main>
<footer class="site-footer">
 
  <p class="footer-copy">© 2026 The AI Skincare Progress Tracker · COM668 Computing Project · B00912171</p>
</footer>
<script>{_JS}</script>
{scripts_extra}
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 - HTML TEMPLATES
# ═══════════════════════════════════════════════════════════════════

INDEX_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>The AI Skincare Progress Tracker - Skincare Progress Tracker</title>
  <style>""" + _CSS + """</style>
  <script src="https://unpkg.com/lucide@latest"></script>\n</head>
<body>
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>

<div class="auth-layout">
  <div class="auth-hero">
    <div class="hero-inner">
      <p class="hero-eyebrow">COM668 Computing Project</p>
      <h1 class="hero-title">The AI Skincare Progress Tracker</h1>
      <p class="hero-sub">Objective, AI-powered skincare progress tracking.<br>Measure what matters. See real results.</p>
      <div class="hero-feats">
        <div class="feat"><i data-lucide="check-circle-2" class="feat-icon"></i><span>Face detection &amp; alignment</span></div>
        <div class="feat"><i data-lucide="check-circle-2" class="feat-icon"></i><span>Acne, redness &amp; texture analysis</span></div>
        <div class="feat"><i data-lucide="check-circle-2" class="feat-icon"></i><span>Progress charts over time</span></div>
        <div class="feat"><i data-lucide="check-circle-2" class="feat-icon"></i><span>All data stays on your device</span></div>
      </div>
    </div>
  </div>

  <div class="auth-forms">
    <!-- LOGIN -->
    <div class="form-card" id="loginCard" {% if show_register %}style="display:none"{% endif %}>
      <h2 class="form-title">Welcome back</h2>
      <p class="form-sub">Sign in to your account</p>
      <form method="POST" action="/login">
        <div class="field"><label>Username</label>
          <input type="text" name="username" placeholder="your username" required autofocus></div>
        <div class="field"><label>Password</label>
          <input type="password" name="password" placeholder="••••••••" required></div>
        <button type="submit" class="btn-primary fw">Sign In</button>
      </form>
      <p class="form-switch">New here? <a href="#" onclick="toggleForms()">Create an account</a></p>
    </div>

    <!-- REGISTER -->
    <div class="form-card" id="registerCard" {% if not show_register %}style="display:none"{% endif %}>
      <h2 class="form-title">Create account</h2>
      <p class="form-sub">Start tracking your skincare journey</p>
      <form method="POST" action="/register">
        <div class="field"><label>Username</label>
          <input type="text" name="username" placeholder="choose a username" required></div>
        <div class="field"><label>Email</label>
          <input type="email" name="email" placeholder="you@example.com" required></div>
        <div class="field"><label>Password</label>
          <input type="password" name="password" placeholder="at least 6 characters" required minlength="6"></div>
        <div class="field"><label>Confirm Password</label>
          <input type="password" name="confirm_password" placeholder="repeat password" required></div>
        <div class="consent-box">
          <label class="consent-label">
            <input type="checkbox" name="consent" required>
            <span>I understand this application captures and locally stores facial images for skincare
            analysis. I consent to this use and acknowledge The AI Skincare Progress Tracker is <strong>not a medical device</strong>
            and provides no medical advice.</span>
          </label>
        </div>
        <button type="submit" class="btn-primary fw">Create Account</button>
      </form>
      <p class="form-switch">Already have an account? <a href="#" onclick="toggleForms()">Sign in</a></p>
    </div>

    <p class="disc-mini"></p>
  </div>
</div>

<script>
function toggleForms() {
  const l = document.getElementById('loginCard'),
        r = document.getElementById('registerCard');
  if (l.style.display === 'none') { l.style.display='block'; r.style.display='none'; }
  else { l.style.display='none'; r.style.display='block'; }
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => { el.style.transition='opacity .4s'; el.style.opacity='0';
      setTimeout(() => el.remove(), 400); }, 5000);
  });
});
</script>
</body>
</html>"""


DASHBOARD_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard - The AI Skincare Progress Tracker</title>
  <style>""" + _CSS + """</style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://unpkg.com/lucide@latest"></script>\n</head>
<body>
{% if current_user.is_authenticated %}
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark"><i data-lucide="activity"></i></span> The AI Skincare Progress Tracker</a>
  <div class="nav-links">
    <a href="/dashboard" class="active">Dashboard</a>
    <a href="/capture">New Scan</a>
    <a href="/history">History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{ current_user.username }}</span>
    <a href="/logout" class="btn-ghost-sm"><i data-lucide="log-out"></i> Log out</a>
  </div>
</nav>
{% endif %}
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>
<main>
<div class="pc">
  <div class="ph">
    <div>
      <h1 class="pt">Skin Health Dashboard</h1>
      <p class="ps">{{ total_scans }} scan{{ 's' if total_scans != 1 }} recorded</p>
    </div>
    <a href="/capture" class="btn-primary"><i data-lucide="plus"></i> New Scan</a>
  </div>

  {% if not latest %}
  <div class="empty-state">
    <div class="empty-icon"><i data-lucide="inbox" width="48" height="48"></i></div>
    <h2>No scans yet</h2>
    <p>Capture your first facial scan to start tracking your skincare progress.</p>
    <a href="/capture" class="btn-primary"><i data-lucide="camera"></i> Capture First Scan</a>
  </div>

  {% else %}

  <!-- Score Cards -->
  <div class="sg">
    <div class="sc sc--o">
      <p class="sl">Overall Score</p>
      <div class="sring">
        <svg viewBox="0 0 120 120" class="rsv">
          <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,.15)" stroke-width="10"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,.9)" stroke-width="10"
                  stroke-linecap="round" stroke-dasharray="314"
                  stroke-dashoffset="{{ 314 - (latest.overall_score / 100 * 314) }}"
                  transform="rotate(-90 60 60)"/>
        </svg>
        <span class="rv">{{ latest.overall_score | int }}</span>
      </div>
      <div class="schg {% if latest.overall_change >= 0 %}pos{% else %}neg{% endif %}">
        {% if latest.overall_change > 0 %}<i data-lucide="trending-up" width="14" height="14"></i>{% elif latest.overall_change < 0 %}<i data-lucide="trending-down" width="14" height="14"></i>{% else %}<i data-lucide="minus" width="14" height="14"></i>{% endif %}
        {{ latest.overall_change | abs | round(1) }}
      </div>
    </div>

    {% for label, score, change, color in [
      ('Skin Clarity', latest.acne_score,    latest.acne_change,    'var(--sage)'),
      ('Redness',      latest.redness_score, latest.redness_change, 'var(--rose)'),
      ('Texture',      latest.texture_score, latest.texture_change, 'var(--sky)'),
    ] %}
    <div class="sc">
      <p class="sl">{{ label }}</p>
      <div class="bscore">
        <div class="btrack"><div class="bfill" style="width:{{ score }}%; background:{{ color }}"></div></div>
        <span class="bnum">{{ score | int }}</span>
      </div>
      <div class="schg {% if change >= 0 %}pos{% else %}neg{% endif %}">
        {% if change > 0 %}<i data-lucide="trending-up" width="14" height="14"></i>{% elif change < 0 %}<i data-lucide="trending-down" width="14" height="14"></i>{% else %}<i data-lucide="minus" width="14" height="14"></i>{% endif %}
        {{ change | abs | round(1) }} vs baseline
      </div>
    </div>
    {% endfor %}
  </div>

  <!-- Latest & Compare -->
  <div class="sr2">
    <div class="card">
      <h3 class="ct">Latest Scan</h3>
      <div class="sprev">
        <img src="/uploads/{{ latest.image_filename }}" class="simg" alt="Latest scan">
        <div class="smeta">
          <p><i data-lucide="clock"></i><strong>Captured:</strong> {{ latest.captured_at.strftime('%d %b %Y, %H:%M') }}</p>
          <p><i data-lucide="hash"></i><strong>Acne spots:</strong> {{ latest.acne_count | int }}</p>
          <p><i data-lucide="activity"></i><strong>Status:</strong>
            <span class="badge badge--{{ latest.analysis_status }}">{{ latest.analysis_status }}</span>
          </p>
          {% if latest.notes %}<p><i data-lucide="file-text"></i><strong>Notes:</strong> {{ latest.notes }}</p>{% endif %}
        </div>
      </div>
    </div>

    {% if total_scans > 1 %}
    <div class="card">
      <h3 class="ct">Progress vs Baseline</h3>
      <div class="compare-row">
        <div class="compare-col">
          <p class="compare-label">Baseline</p>
          <img src="/uploads/{{ baseline.image_filename }}" class="compare-img" alt="Baseline">
          <p class="compare-date">{{ baseline.captured_at.strftime('%d %b %Y') }}</p>
        </div>
        <div class="compare-arrow"><i data-lucide="arrow-right"></i></div>
        <div class="compare-col">
          <p class="compare-label">Latest</p>
          <img src="/uploads/{{ latest.image_filename }}" class="compare-img" alt="Latest">
          <p class="compare-date">{{ latest.captured_at.strftime('%d %b %Y') }}</p>
        </div>
      </div>
    </div>
    {% endif %}
  </div>

  <!-- Charts -->
  {% if total_scans > 1 %}
  <div class="card">
    <h3 class="ct">Progress Over Time</h3>
    <div class="chtabs">
      <button class="tb active" onclick="showChart('cO', this)">Overall</button>
      <button class="tb"        onclick="showChart('cA', this)">Skin Clarity</button>
      <button class="tb"        onclick="showChart('cR', this)">Redness</button>
      <button class="tb"        onclick="showChart('cT', this)">Texture</button>
    </div>
    <div class="chtwrap">
      <canvas id="cO"></canvas>
      <canvas id="cA" style="display:none"></canvas>
      <canvas id="cR" style="display:none"></canvas>
      <canvas id="cT" style="display:none"></canvas>
    </div>
  </div>
  {% endif %}

  {% endif %}
</div>
</main>
<footer class="site-footer">
  <div class="disclaimer"><i data-lucide="alert-triangle"></i><strong>Medical Disclaimer:</strong> The AI Skincare Progress Tracker tracks skincare
  effectiveness only - not a medical device. Consult a dermatologist for skin health concerns.</div>
  <p class="footer-copy">© 2026 The AI Skincare Progress Tracker · COM668 Computing Project · B00912171</p>
</footer>
<script>""" + _JS + """</script>
{% if total_scans > 1 %}
<script>
const L = {{ chart_labels | tojson }};
const cfg = { type:'line', options:{ responsive:true,
  plugins:{ legend:{ display:false } },
  scales:{ x:{ grid:{ color:'rgba(0,0,0,.05)' } },
           y:{ min:0, max:100, grid:{ color:'rgba(0,0,0,.05)' } } },
  elements:{ point:{ radius:5, hoverRadius:7 } }
}};
function mk(id, data, c) {
  return new Chart(document.getElementById(id), { ...cfg,
    data:{ labels:L, datasets:[{ data, borderColor:c,
      backgroundColor: c+'22', fill:true, tension:.4, borderWidth:2.5 }] }
  });
}
mk('cO', {{ overall_data  | tojson }}, '#4a7c59');
mk('cA', {{ acne_data     | tojson }}, '#4a7c59');
mk('cR', {{ redness_data  | tojson }}, '#c17b6c');
mk('cT', {{ texture_data  | tojson }}, '#6c9ec1');

function showChart(id, btn) {
  ['cO','cA','cR','cT'].forEach(k =>
    document.getElementById(k).style.display = k === id ? 'block' : 'none');
  document.querySelectorAll('.tb').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}
</script>
{% endif %}
</body>
</html>"""


CAPTURE_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>New Scan - The AI Skincare Progress Tracker</title>
  <style>""" + _CSS + """</style>
  <script src="https://unpkg.com/lucide@latest"></script>\n</head>
<body>
{% if current_user.is_authenticated %}
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark"><i data-lucide="activity"></i></span> The AI Skincare Progress Tracker</a>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a>
    <a href="/capture" class="active">New Scan</a>
    <a href="/history">History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{ current_user.username }}</span>
    <a href="/logout" class="btn-ghost-sm"><i data-lucide="log-out"></i> Log out</a>
  </div>
</nav>
{% endif %}
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>
<main>
<div class="pc pc--n">
  <div class="ph">
    <div>
      <h1 class="pt">New Skin Scan</h1>
      <p class="ps">Upload a clear, frontal photo of your face for analysis</p>
    </div>
  </div>

  <div class="tips-bar">
    <div class="tip"><i data-lucide="user"></i> Face the camera directly</div>
    <div class="tip"><i data-lucide="sun"></i> Good natural or bright light</div>
    <div class="tip"><i data-lucide="ruler"></i> Keep consistent distance</div>
    <div class="tip"><i data-lucide="droplets"></i> Clean, make-up-free skin</div>
  </div>

  <div class="card">
    <form method="POST" action="/upload" enctype="multipart/form-data" id="uploadForm">
      <div class="capture-options" style="display: flex; gap: 10px; margin-bottom: 15px; justify-content: center;">
        <button type="button" class="btn-secondary" onclick="document.getElementById('imageInput').click()" style="padding: 0.6rem 1rem; border-radius: 8px; border: 1px solid #ccc; cursor: pointer; background: #fff;"><i data-lucide="image"></i> Upload Image</button>
        <button type="button" class="btn-secondary" onclick="startCamera()" style="padding: 0.6rem 1rem; border-radius: 8px; border: 1px solid #ccc; cursor: pointer; background: #fff;"><i data-lucide="video"></i> Use Camera</button>
      </div>
      <div class="upload-zone" id="uploadZone">
        <input type="file" name="image" id="imageInput" accept="image/*"
               style="display:none" onchange="previewImage(event)" required>
        <div id="uploadPlaceholder" onclick="document.getElementById('imageInput').click()">
          <div class="upload-icon"><i data-lucide="upload-cloud" width="48" height="48"></i></div>
          <p class="upload-cta">Click to choose an image</p>
          <p class="upload-hint">JPG, PNG or WEBP · max 16 MB</p>
        </div>
        <div id="cameraContainer" style="display:none; position: relative;">
          <video id="cameraStream" autoplay playsinline style="width: 100%; border-radius: 8px;"></video>
          <div class="preview-actions" style="margin-top: 10px;">
            <button type="button" class="btn-primary" onclick="capturePhoto()"><i data-lucide="camera"></i> Take Photo</button>
            <button type="button" class="btn-ghost" onclick="stopCamera()"><i data-lucide="x"></i> Cancel</button>
          </div>
          <canvas id="canvas" style="display:none;"></canvas>
        </div>
        <div id="previewContainer" style="display:none">
          <img id="previewImg" src="" alt="Preview" class="preview-image">
          <div class="preview-actions">
            <button type="button" class="btn-ghost" onclick="clearPreview()">Choose different</button>
          </div>
        </div>
      </div>

      <div class="field">
        <label for="notes">Notes <span class="opt">(optional)</span></label>
        <input type="text" name="notes" id="notes"
               placeholder="e.g. Week 3 of new moisturiser, morning routine…">
      </div>

      <div class="consent-reminder">
        By submitting this scan you confirm consent to facial image capture
        for personal skincare tracking. All data is stored locally only.
      </div>

      <button type="submit" class="btn-primary fw" id="submitBtn" disabled>
        <span id="btnText"><i data-lucide="scan-line"></i> Analyse Scan</span>
        <span id="btnLoader" style="display:none"><i data-lucide="loader-2" class="lucide-spin"></i> Analysing…</span>
      </button>
    </form>
  </div>

  <div class="info-grid">
    <div class="info-card">
      <h4><i data-lucide="microscope"></i> Acne Detection</h4>
      <p>Counts blemishes and measures skin clarity using blob detection
         and HSV colour analysis.</p>
    </div>
    <div class="info-card">
      <h4><i data-lucide="thermometer"></i> Redness Analysis</h4>
      <p>Measures inflammation by analysing colour distribution
         across all facial regions.</p>
    </div>
    <div class="info-card">
      <h4><i data-lucide="layers"></i> Texture Score</h4>
      <p>Evaluates skin smoothness using Local Binary Pattern (LBP)
         texture descriptors (Ojala et al., 2002).</p>
    </div>
  </div>
</div>
</main>
<footer class="site-footer">
  <div class="disclaimer"><i data-lucide="alert-triangle"></i><strong>Medical Disclaimer:</strong> The AI Skincare Progress Tracker tracks skincare
  effectiveness only - not a medical device.</div>
  <p class="footer-copy">© 2026 The AI Skincare Progress Tracker · COM668 Computing Project · B00912171</p>
</footer>
<script>""" + _JS + """</script>
<script>
let videoStream = null;

function previewImage(e) {
  const f = e.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = ev => {
    document.getElementById('previewImg').src = ev.target.result;
    document.getElementById('uploadPlaceholder').style.display = 'none';
    document.getElementById('cameraContainer').style.display = 'none';
    document.getElementById('previewContainer').style.display  = 'block';
    document.getElementById('submitBtn').disabled = false;
  };
  r.readAsDataURL(f);
}

function clearPreview() {
  document.getElementById('imageInput').value = '';
  document.getElementById('previewImg').src   = '';
  document.getElementById('uploadPlaceholder').style.display = 'block';
  document.getElementById('previewContainer').style.display  = 'none';
  document.getElementById('cameraContainer').style.display = 'none';
  document.getElementById('submitBtn').disabled = true;
  stopCamera();
}

async function startCamera() {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    const video = document.getElementById('cameraStream');
    video.srcObject = videoStream;
    document.getElementById('uploadPlaceholder').style.display = 'none';
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('cameraContainer').style.display = 'block';
    document.getElementById('submitBtn').disabled = true;
  } catch (err) {
    alert("Error accessing camera: " + err.message);
  }
}

function stopCamera() {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  document.getElementById('cameraContainer').style.display = 'none';
  if (!document.getElementById('imageInput').files.length) {
    document.getElementById('uploadPlaceholder').style.display = 'block';
  }
}

function capturePhoto() {
  const video = document.getElementById('cameraStream');
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  
  canvas.toBlob(blob => {
    const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById('imageInput').files = dt.files;
    
    document.getElementById('previewImg').src = canvas.toDataURL('image/jpeg');
    document.getElementById('cameraContainer').style.display = 'none';
    document.getElementById('previewContainer').style.display = 'block';
    document.getElementById('submitBtn').disabled = false;
    
    stopCamera();
  }, 'image/jpeg', 0.95);
}

document.getElementById('uploadForm').addEventListener('submit', () => {
  document.getElementById('btnText').style.display   = 'none';
  document.getElementById('btnLoader').style.display = 'inline';
  document.getElementById('submitBtn').disabled      = true;
});
const z = document.getElementById('uploadZone');
z.addEventListener('dragover', e => { e.preventDefault(); z.classList.add('drag-over'); });
z.addEventListener('dragleave', () => z.classList.remove('drag-over'));
z.addEventListener('drop', e => {
  e.preventDefault(); z.classList.remove('drag-over');
  const inp = document.getElementById('imageInput');
  const dt  = new DataTransfer();
  dt.items.add(e.dataTransfer.files[0]);
  inp.files = dt.files;
  previewImage({ target: inp });
});
</script>
</body>
</html>"""


HISTORY_T = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>History - The AI Skincare Progress Tracker</title>
  <style>""" + _CSS + """</style>
  <script src="https://unpkg.com/lucide@latest"></script>\n</head>
<body>
{% if current_user.is_authenticated %}
<nav class="navbar">
  <a class="nav-brand" href="/dashboard"><span class="brand-mark"><i data-lucide="activity"></i></span> The AI Skincare Progress Tracker</a>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a>
    <a href="/capture">New Scan</a>
    <a href="/history" class="active">History</a>
  </div>
  <div class="nav-user">
    <span class="user-chip">{{ current_user.username }}</span>
    <a href="/logout" class="btn-ghost-sm"><i data-lucide="log-out"></i> Log out</a>
  </div>
</nav>
{% endif %}
<div class="flash-container">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}
  <div class="flash flash-{{ cat }}">{{ msg }}
    <button class="flash-close" onclick="this.parentElement.remove()">×</button>
  </div>
  {% endfor %}{% endwith %}
</div>
<main>
<div class="pc">
  <div class="ph">
    <div>
      <h1 class="pt">Scan History</h1>
      <p class="ps">{{ scans | length }} total scan{{ 's' if scans | length != 1 }}</p>
    </div>
    <a href="/capture" class="btn-primary"><i data-lucide="plus"></i> New Scan</a>
  </div>

  {% if not scans %}
  <div class="empty-state">
    <div class="empty-icon"><i data-lucide="inbox" width="48" height="48"></i></div>
    <h2>No scans yet</h2>
    <p>Start your skincare tracking journey by capturing your first scan.</p>
    <a href="/capture" class="btn-primary"><i data-lucide="camera"></i> Capture First Scan</a>
  </div>

  {% else %}
  <div class="history-grid">
    {% for scan in scans %}
    <div class="hcard">
      <div class="hiw">
        <img src="/uploads/{{ scan.image_filename }}" class="himg" alt="Scan {{ scan.id }}">
        <span class="badge badge--{{ scan.analysis_status }} hbadge">{{ scan.analysis_status }}</span>
      </div>
      <div class="hbody">
        <p class="hdate"><i data-lucide="calendar"></i> {{ scan.captured_at.strftime('%d %b %Y · %H:%M') }}</p>

        {% if scan.analysis_status == 'complete' %}
        <div class="mini-scores">
          <div class="mini-score"><span class="ms-label">Overall</span>
            <span class="ms-val">{{ scan.overall_score | int }}</span></div>
          <div class="mini-score"><span class="ms-label">Clarity</span>
            <span class="ms-val">{{ scan.acne_score | int }}</span></div>
          <div class="mini-score"><span class="ms-label">Redness</span>
            <span class="ms-val">{{ scan.redness_score | int }}</span></div>
          <div class="mini-score"><span class="ms-label">Texture</span>
            <span class="ms-val">{{ scan.texture_score | int }}</span></div>
        </div>
        {% if scan.overall_change != 0 %}
        <p class="hchg {% if scan.overall_change > 0 %}pos{% else %}neg{% endif %}">
          {% if scan.overall_change > 0 %}<i data-lucide="trending-up" width="14" height="14"></i>{% else %}<i data-lucide="trending-down" width="14" height="14"></i>{% endif %}
          {{ scan.overall_change | abs | round(1) }} vs baseline
        </p>
        {% endif %}

        {% elif scan.analysis_status == 'failed' %}
        <p class="hnote"><i data-lucide="alert-circle"></i> No face detected - re-capture in better light.</p>
        {% else %}
        <p class="hnote">Pending analysis</p>
        {% endif %}

        {% if scan.notes %}<p class="hnotes"><i data-lucide="file-text"></i> {{ scan.notes }}</p>{% endif %}

        <form method="POST" action="/scan/{{ scan.id }}/delete" class="del-form"
              onsubmit="return confirm('Delete this scan permanently?')">
          <button type="submit" class="btn-dsm"><i data-lucide="trash"></i> Delete</button>
        </form>
      </div>
    </div>
    {% endfor %}
  </div>

  <div class="card data-rights">
    <h3><i data-lucide="shield"></i> Your Data Rights</h3>
    <p>All facial images and analysis data are stored locally on this device only.
       You may delete individual scans above, or permanently delete your entire
       account and all associated data below (GDPR right to erasure).</p>
    <form method="POST" action="/account/delete"
          onsubmit="return confirm('This will permanently delete your account and ALL data. Are you absolutely sure?')">
      <button type="submit" class="btn-danger"><i data-lucide="trash-2"></i> <i data-lucide="trash"></i> Delete My Account &amp; All Data</button>
    </form>
  </div>
  {% endif %}
</div>
</main>
<footer class="site-footer">
  <div class="disclaimer"><i data-lucide="alert-triangle"></i><strong>Medical Disclaimer:</strong> The AI Skincare Progress Tracker tracks skincare
  effectiveness only - not a medical device. Consult a dermatologist for skin concerns.</div>
  <p class="footer-copy">© 2026 The AI Skincare Progress Tracker · COM668 Computing Project · B00912171</p>
</footer>
<script>""" + _JS + """</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# SECTION 10 - ROUTES
# ═══════════════════════════════════════════════════════════════════

def _allowed(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'})


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template_string(INDEX_T, show_register=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        u  = request.form.get('username', '').strip()
        e  = request.form.get('email', '').strip().lower()
        p  = request.form.get('password', '')
        cp = request.form.get('confirm_password', '')
        c  = request.form.get('consent') == 'on'

        if not c:
            flash('You must provide informed consent to use this application.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if p != cp:
            flash('Passwords do not match.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if len(p) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if User.query.filter_by(username=u).first():
            flash('Username already taken.', 'error')
            return render_template_string(INDEX_T, show_register=True)
        if User.query.filter_by(email=e).first():
            flash('Email already registered.', 'error')
            return render_template_string(INDEX_T, show_register=True)

        user = User(username=u, email=e, consent_given=c)
        user.set_password(p)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash(f'Welcome, {u}! Your account has been created.', 'success')
        return redirect(url_for('dashboard'))

    return render_template_string(INDEX_T, show_register=True)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        u    = request.form.get('username', '').strip()
        p    = request.form.get('password', '')
        user = User.query.filter_by(username=u).first()
        if user and user.check_password(p):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template_string(INDEX_T, show_register=False)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    scans    = (SkinScan.query.filter_by(user_id=current_user.id)
                .order_by(SkinScan.captured_at.asc()).all())
    latest   = scans[-1] if scans else None
    baseline = scans[0]  if scans else None
    return render_template_string(
        DASHBOARD_T,
        scans         = scans,
        latest        = latest,
        baseline      = baseline,
        chart_labels  = [s.captured_at.strftime('%d %b') for s in scans],
        overall_data  = [s.overall_score  for s in scans],
        acne_data     = [s.acne_score     for s in scans],
        redness_data  = [s.redness_score  for s in scans],
        texture_data  = [s.texture_score  for s in scans],
        total_scans   = len(scans),
    )


@app.route('/capture')
@login_required
def capture_page():
    return render_template_string(CAPTURE_T)


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files:
        flash('No image provided.', 'error')
        return redirect(url_for('capture_page'))

    file  = request.files['image']
    notes = request.form.get('notes', '')

    if not file.filename or not _allowed(file.filename):
        flash('Unsupported file type. Use PNG, JPG or JPEG.', 'error')
        return redirect(url_for('capture_page'))

    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{current_user.id}_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    scan = SkinScan(user_id=current_user.id, image_filename=filename,
                    notes=notes, analysis_status='pending')
    db.session.add(scan)
    db.session.commit()

    bl = (SkinScan.query
          .filter_by(user_id=current_user.id, analysis_status='complete')
          .order_by(SkinScan.captured_at.asc()).first())
    if bl and bl.id == scan.id:
        bl = None

    try:
        res = analyse_image(filepath, baseline_scan=bl)
        scan.face_detected   = res['face_detected']
        scan.acne_count      = res['acne_count']
        scan.acne_score      = res['acne_score']
        scan.redness_score   = res['redness_score']
        scan.texture_score   = res['texture_score']
        scan.overall_score   = res['overall_score']
        scan.acne_change     = res['acne_change']
        scan.redness_change  = res['redness_change']
        scan.texture_change  = res['texture_change']
        scan.overall_change  = res['overall_change']
        scan.analysis_status = res['analysis_status']
        db.session.commit()

        if not res['face_detected']:
            flash(f'⚠ {res["message"]} - scan saved but not analysed.', 'warning')
        else:
            flash('✓ Scan analysed successfully!', 'success')
    except Exception as e:
        scan.analysis_status = 'failed'
        db.session.commit()
        flash(f'Analysis error: {e}', 'error')

    return redirect(url_for('dashboard'))


@app.route('/history')
@login_required
def history():
    scans = (SkinScan.query.filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.desc()).all())
    return render_template_string(HISTORY_T, scans=scans)


@app.route('/scan/<int:scan_id>/delete', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan = SkinScan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        abort(403)
    fp = os.path.join(UPLOAD_DIR, scan.image_filename)
    if os.path.exists(fp):
        os.remove(fp)
    db.session.delete(scan)
    db.session.commit()
    flash('Scan deleted.', 'info')
    return redirect(url_for('history'))


@app.route('/account/delete', methods=['POST'])
@login_required
def delete_account():
    """GDPR right to erasure - deletes all user data."""
    user = current_user
    for scan in user.scans:
        fp = os.path.join(UPLOAD_DIR, scan.image_filename)
        if os.path.exists(fp):
            os.remove(fp)
    logout_user()
    db.session.delete(user)
    db.session.commit()
    flash('Your account and all data have been permanently deleted.', 'info')
    return redirect(url_for('index'))


@app.route('/api/scans')
@login_required
def api_scans():
    scans = (SkinScan.query.filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.asc()).all())
    return jsonify([s.to_dict() for s in scans])


# ═══════════════════════════════════════════════════════════════════
# SECTION 11 - ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        logger.info("Database tables created / verified.")

    print("=" * 60)
    print("  The AI Skincare Progress Tracker — AI Skincare Progress Tracker")
    print("  COM668 Computing Project · B00912171")
    print("  ----------------------------------------")
    print("  Open your browser:  http://127.0.0.1:5000")
    print("=" * 60)

    pass # app.run(debug=True, host='0.0.0.0', port=5000)

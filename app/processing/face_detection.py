"""
Face Detection Module
Uses Mediapipe Face Detection for robust face and landmark localisation.
"""

import numpy as np
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

_mp_face_detection = None

def _get_detector():
    global _mp_face_detection
    if _mp_face_detection is None:
        try:
            import mediapipe as mp
            _mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
            logger.info("Mediapipe FaceDetection loaded")
        except Exception as e:
            logger.error(f"Failed to load Mediapipe: {e}")
            _mp_face_detection = None
    return _mp_face_detection


def detect_and_align_face(image_path, target_size=(224, 224)):
    result = {
        'success':    False,
        'face_image': None,
        'box':        None,
        'landmarks':  None,
        'confidence': 0.0,
        'message':    ''
    }

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        result['message'] = f'Could not open image: {e}'
        return result

    detector = _get_detector()
    if detector is not None:
        cv_img = np.array(img)
        # Mediapipe expects RGB numpy array
        results = detector.process(cv_img)
        
        if results.detections:
            # Get the highest confidence face
            detection = max(results.detections, key=lambda d: d.score[0])
            confidence = float(detection.score[0])
            bboxC = detection.location_data.relative_bounding_box
            ih, iw = cv_img.shape[:2]
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Padding 10%
            pad_x = w * 0.1
            pad_y = h * 0.1
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(iw, x + w + pad_x)
            y2 = min(ih, y + h + pad_y)

            face_crop = img.crop((x1, y1, x2, y2)).resize(target_size, Image.LANCZOS)
            
            result.update({
                'success':    True,
                'face_image': face_crop,
                'box':        [float(x1), float(y1), float(x2), float(y2)],
                'confidence': confidence,
                'message':    f'Face detected via Mediapipe (confidence: {confidence:.2f})'
            })
            return result
            
        else:
            result['message'] = 'No face detected in image.'
            return result

    result['message'] = 'Face detector not available.'
    return result


def get_face_regions(face_image_array):
    """
    Divides a face image into anatomical regions of interest (ROIs).
    Returns a dict of region_name -> numpy array crop.
    """
    h, w = face_image_array.shape[:2]

    regions = {
        'forehead':    face_image_array[0:int(h*0.30), int(w*0.2):int(w*0.8)],
        'left_cheek':  face_image_array[int(h*0.35):int(h*0.70), 0:int(w*0.40)],
        'right_cheek': face_image_array[int(h*0.35):int(h*0.70), int(w*0.60):w],
        'nose':        face_image_array[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)],
        'chin':        face_image_array[int(h*0.72):h,            int(w*0.25):int(w*0.75)],
    }
    return {k: v for k, v in regions.items() if v.size > 0}


def get_face_region_coords(h, w):
    """
    Returns a dictionary of region names to (x, y, w, h) coordinates 
    based on the face image dimensions.
    """
    return {
        'forehead':    (int(w*0.2), 0, int(w*0.8) - int(w*0.2), int(h*0.30)),
        'left_cheek':  (0, int(h*0.35), int(w*0.40), int(h*0.70) - int(h*0.35)),
        'right_cheek': (int(w*0.60), int(h*0.35), w - int(w*0.60), int(h*0.70) - int(h*0.35)),
        'nose':        (int(w*0.35), int(h*0.35), int(w*0.65) - int(w*0.35), int(h*0.65) - int(h*0.35)),
        'chin':        (int(w*0.25), int(h*0.72), int(w*0.75) - int(w*0.25), h - int(h*0.72)),
    }

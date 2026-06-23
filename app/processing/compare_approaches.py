"""
Compares traditional CV vs ML model on the same test images.
Run this to generate accuracy numbers for your AT3 demo video.
"""

import os
import numpy as np
from PIL import Image
from preprocessing   import preprocess_face
from acne_detection   import detect_acne as cv_detect
from acne_detection_ml import detect_acne as ml_detect

# Put 20 test images here with known severity (you label them manually)
# Format: (image_path, true_severity_0_to_3)
TEST_IMAGES = [
    # ('test_images/clear1.jpg',    0),
    # ('test_images/clear2.jpg',    0),
]

cv_correct = 0
ml_correct = 0

print(f"{'Image':<25} {'True':>6} {'CV pred':>8} {'ML pred':>8} {'CV':>4} {'ML':>4}")
print("-" * 60)

for path, true_label in TEST_IMAGES:
    try:
        img = Image.open(path).convert('RGB')
        arr = preprocess_face(img)

        cv_result = cv_detect(arr)
        ml_result = ml_detect(arr)

        # Convert CV score back to severity bucket for comparison
        cv_score = cv_result['acne_score']
        if cv_score >= 85:   cv_pred = 0
        elif cv_score >= 60: cv_pred = 1
        elif cv_score >= 35: cv_pred = 2
        else:                cv_pred = 3

        ml_pred = int(ml_result['acne_count'])  # 0-3 severity

        cv_ok = cv_pred == true_label
        ml_ok = ml_pred == true_label
        cv_correct += cv_ok
        ml_correct += ml_ok

        print(f"{os.path.basename(path):<25} {true_label:>6} {cv_pred:>8} {ml_pred:>8} "
              f"{'✓' if cv_ok else '✗':>4} {'✓' if ml_ok else '✗':>4}")
    except Exception as e:
        print(f"Failed to process {path}: {e}")

n = len(TEST_IMAGES)
if n > 0:
    print("-" * 60)
    print(f"{'ACCURACY':<25} {'':>6} {'':>8} {'':>8} "
          f"{cv_correct/n:>4.0%} {ml_correct/n:>4.0%}")
    print(f"\nTraditional CV: {cv_correct}/{n} correct ({cv_correct/n:.0%})")
    print(f"ML MobileNetV2: {ml_correct}/{n} correct ({ml_correct/n:.0%})")
else:
    print("No test images provided. Please add test images to TEST_IMAGES list.")

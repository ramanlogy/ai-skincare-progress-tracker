# The AI Skincare Progress Tracker — AI Skincare Progress Tracker
### COM668 Computing Project · Student ID: B00912171

An AI-powered web application that objectively tracks skincare product
effectiveness by analysing facial images over time, measuring acne count,
redness intensity, and skin texture using computer vision and machine learning.

---

## Quick Start

### 1. Prerequisites
- Python 3.9 or higher  
- pip  
- Webcam or smartphone for capturing facial images

### 2. Clone / Download the project
```bash
cd ai-skincare-tracker
```

### 3. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** First install takes 5–10 minutes — PyTorch and OpenCV are large packages.

### 5. Run the application
```bash
python run.py
```

### 6. Open in browser
Visit **http://127.0.0.1:5000**

---

## Project Structure

```
ai-skincare-tracker/
│
├── run.py                         ← Entry point
├── config.py                      ← App configuration
├── requirements.txt               ← Python dependencies
│
├── app/
│   ├── __init__.py                ← Flask app factory
│   ├── models.py                  ← SQLite database models (User, SkinScan)
│   ├── routes.py                  ← All URL routes and API endpoints
│   │
│   ├── processing/                ← AI/CV pipeline
│   │   ├── face_detection.py      ← MTCNN face detection (Zhang et al., 2016)
│   │   ├── preprocessing.py       ← CLAHE lighting normalisation
│   │   ├── acne_detection.py      ← Blob detection + HSV thresholding
│   │   ├── redness.py             ← HSV redness analysis
│   │   ├── texture.py             ← LBP texture scoring (Ojala et al., 2002)
│   │   └── scorer.py              ← Composite progress score engine
│   │
│   ├── templates/                 ← Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── index.html             ← Login / Register
│   │   ├── dashboard.html         ← Main dashboard with charts
│   │   ├── capture.html           ← Image upload
│   │   └── history.html           ← Scan history + GDPR deletion
│   │
│   └── static/
│       ├── css/style.css
│       ├── js/main.js
│       └── uploads/               ← User images (auto-created)
│
├── tests/
│   └── test_pipeline.py           ← Unit tests for CV pipeline
│
└── instance/
    └── skincare.db                ← SQLite database (auto-created)
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage report
pip install pytest-cov
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## System Architecture

```
User Image Upload
        │
        ▼
┌─────────────────────┐
│   Face Detection    │  ← MTCNN (Zhang et al., 2016)
│   + Alignment       │    OpenCV Haar fallback
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Preprocessing     │  ← CLAHE lighting normalisation
│                     │    Gray-world colour correction
└────────┬────────────┘
         │
    ┌────┴────────────────────┐
    │         │               │
    ▼         ▼               ▼
┌───────┐ ┌────────┐  ┌──────────┐
│ Acne  │ │Redness │  │ Texture  │
│Detect.│ │Analyse │  │  (LBP)   │
└───┬───┘ └───┬────┘  └────┬─────┘
    │         │             │
    └─────────┴─────────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │    Composite Scorer     │
    │  Score = Acne×0.4       │
    │        + Redness×0.3    │
    │        + Texture×0.3    │
    └────────────┬────────────┘
                 │
                 ▼
          SQLite Database
                 │
                 ▼
        Dashboard + Charts
```

---

## Composite Progress Score

The overall skin health score uses the weighted formula recommended by the
project supervisor:

```
Progress Score = (Acne_Score × 0.4) +
                 (Redness_Score × 0.3) +
                 (Texture_Score × 0.3)
```

All component scores are on a 0–100 scale where **100 = best possible**.
Changes are tracked relative to the user's first (baseline) scan.

---

## Ethical Considerations

<<<<<<< HEAD
- **Not a medical device.** The AI Skincare Progress Tracker tracks skincare product effectiveness only.
=======
- **Not a medical device.** SanjuAI tracks skincare product effectiveness only.
>>>>>>> 718b2949afcd5d59da1e8a69c1016927062668ff
  It does not diagnose, treat, or provide medical advice. A prominent disclaimer
  is displayed on every page.
- **Informed consent.** Users must explicitly consent to facial image capture
  during registration.
- **Data privacy.** All images and analysis data are stored locally (SQLite +
  filesystem). No data is sent to third parties.
- **Right to erasure (GDPR).** Users can delete individual scans or their
  entire account from the History page at any time.
- **Algorithmic fairness.** Testing should cover diverse skin tones
  (Fitzpatrick scale I–VI) to identify accuracy disparities.

---

## Key References (from AT1 Proposal)

- Zhang, K. et al. (2016) — MTCNN face detection/alignment
- Ojala, T. et al. (2002) — Local Binary Patterns for texture analysis
- Bradski, G. (2000) — OpenCV library
- Paszke, A. et al. (2019) — PyTorch

---

## AT2 Action Items Addressed

| Supervisor Requirement | Status |
|------------------------|--------|
| PyTorch vs TF decision | ✅ PyTorch (facenet-pytorch / MTCNN) |
| Platform specification | ✅ Web app (Flask) |
| Evaluation metrics | ✅ Accuracy tests in test_pipeline.py |
| Risk: no face detected | ✅ Quality check + user guidance message |
| GDPR / data privacy | ✅ Consent form + right-to-delete page |
| Non-medical disclaimer | ✅ Footer + capture page |
| Diverse skin tone testing | ✅ Planned in test_pipeline.py |

---

## Troubleshooting

**`facenet-pytorch` not installing:**
```bash
pip install facenet-pytorch --no-deps
pip install torch torchvision
```

**Face not detected:**  
Ensure the image has a clear, well-lit, frontal face.  
The app falls back to OpenCV Haar cascades automatically.

**Port already in use:**
```bash
python run.py  # change port in run.py if needed
```

"""
Unit Tests — AI Skincare Progress Tracker
Run with:  python -m pytest tests/ -v
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_face(h=224, w=224, skin_tone=(200, 170, 140)):
    """Create a synthetic skin-toned face array for testing."""
    arr         = np.ones((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = skin_tone[0]
    arr[:, :, 1] = skin_tone[1]
    arr[:, :, 2] = skin_tone[2]
    return arr


def _add_red_patch(arr, cx=112, cy=112, r=10):
    """Add a red circular patch (simulated blemish) to the array."""
    import cv2
    patched = arr.copy()
    cv2.circle(patched, (cx, cy), r, (200, 50, 50), -1)
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_normalize_lighting_shape(self):
        from app.processing.preprocessing import normalize_lighting
        face = _make_face()
        out  = normalize_lighting(face)
        assert out.shape == face.shape, "Output shape should match input"

    def test_normalize_lighting_dtype(self):
        from app.processing.preprocessing import normalize_lighting
        out = normalize_lighting(_make_face())
        assert out.dtype == np.uint8

    def test_preprocess_face_pil(self):
        from PIL import Image
        from app.processing.preprocessing import preprocess_face
        pil  = Image.fromarray(_make_face())
        out  = preprocess_face(pil, target_size=(224, 224))
        assert out.shape == (224, 224, 3)

    def test_preprocess_resize(self):
        from app.processing.preprocessing import preprocess_face
        big = _make_face(512, 512)
        out = preprocess_face(big, target_size=(128, 128))
        assert out.shape == (128, 128, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Acne detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAcneDetection:
    def test_returns_expected_keys(self):
        from app.processing.acne_detection_ml import detect_acne
        res = detect_acne(_make_face())
        assert 'acne_count' in res
        assert 'acne_score' in res

    def test_clear_skin_high_score(self):
        from app.processing.acne_detection_ml import detect_acne
        res = detect_acne(_make_face(skin_tone=(210, 180, 155)))
        assert res['acne_score'] >= 50, "Clean skin should score ≥ 50"

    def test_score_range(self):
        from app.processing.acne_detection_ml import detect_acne
        res = detect_acne(_make_face())
        assert 0 <= res['acne_score'] <= 100

    def test_acne_count_non_negative(self):
        from app.processing.acne_detection_ml import detect_acne
        res = detect_acne(_make_face())
        assert res['acne_count'] >= 0

    def test_empty_input(self):
        from app.processing.acne_detection_ml import detect_acne
        res = detect_acne(None)
        assert res['acne_count'] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Redness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRedness:
    def test_returns_expected_keys(self):
        from app.processing.redness import analyse_redness
        res = analyse_redness(_make_face())
        assert 'redness_score' in res
        assert 'redness_pct'   in res

    def test_score_range(self):
        from app.processing.redness import analyse_redness
        res = analyse_redness(_make_face())
        assert 0 <= res['redness_score'] <= 100

    def test_red_face_lower_score(self):
        from app.processing.redness import analyse_redness
        normal = _make_face(skin_tone=(200, 170, 140))
        red    = _make_face(skin_tone=(200, 60, 60))
        res_n  = analyse_redness(normal)
        res_r  = analyse_redness(red)
        assert res_r['redness_score'] < res_n['redness_score'], \
            "Red-toned face should have lower redness_score"

    def test_empty_input(self):
        from app.processing.redness import analyse_redness
        res = analyse_redness(None)
        assert res['redness_score'] == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Texture tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTexture:
    def test_returns_expected_keys(self):
        from app.processing.texture import analyse_texture
        res = analyse_texture(_make_face())
        assert 'texture_score' in res
        assert 'uniformity'    in res

    def test_score_range(self):
        from app.processing.texture import analyse_texture
        res = analyse_texture(_make_face())
        assert 0 <= res['texture_score'] <= 100

    def test_uniform_vs_noisy(self):
        from app.processing.texture import analyse_texture
        uniform = _make_face()
        noisy   = uniform.copy()
        noisy   = (noisy + np.random.randint(0, 80, noisy.shape, dtype=np.uint8)).clip(0, 255).astype(np.uint8)
        res_u   = analyse_texture(uniform)
        res_n   = analyse_texture(noisy)
        assert res_u['texture_score'] >= res_n['texture_score'], \
            "Uniform (smooth) face should have higher texture_score than noisy"

    def test_empty_input(self):
        from app.processing.texture import analyse_texture
        res = analyse_texture(None)
        assert res['texture_score'] == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Composite scorer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScorer:
    def test_composite_perfect(self):
        from app.processing.scorer import composite_score
        assert composite_score(100, 100, 100) == 100.0

    def test_composite_zero(self):
        from app.processing.scorer import composite_score
        assert composite_score(0, 0, 0) == 0.0

    def test_composite_weights(self):
        from app.processing.scorer import composite_score
        # Acne 0.4, redness 0.3, texture 0.3
        expected = 100*0.4 + 0*0.3 + 0*0.3
        assert composite_score(100, 0, 0) == round(expected, 1)

    def test_compute_change(self):
        from app.processing.scorer import compute_change
        assert compute_change(80, 70) ==  10.0
        assert compute_change(60, 70) == -10.0
        assert compute_change(70, 70) ==   0.0


# ─────────────────────────────────────────────────────────────────────────────
# Face region test
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceRegions:
    def test_regions_non_empty(self):
        from app.processing.face_detection import get_face_regions
        face  = _make_face()
        regs  = get_face_regions(face)
        assert len(regs) >= 4, "Should return at least 4 face regions"

    def test_region_names(self):
        from app.processing.face_detection import get_face_regions
        regs = get_face_regions(_make_face())
        for name in ('forehead', 'left_cheek', 'right_cheek', 'nose'):
            assert name in regs, f"Missing region: {name}"

    def test_region_arrays_have_content(self):
        from app.processing.face_detection import get_face_regions
        regs = get_face_regions(_make_face())
        for name, arr in regs.items():
            assert arr.size > 0, f"Region '{name}' is empty"


# ─────────────────────────────────────────────────────────────────────────────
# Flask app tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFlaskApp:
    def setup_method(self):
        os.environ['TESTING'] = '1'
        from app import create_app, db
        self.app = create_app()
        self.app.config.update({
            'TESTING':              True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'WTF_CSRF_ENABLED':     False,
        })
        self.client = self.app.test_client()
        with self.app.app_context():
            db.create_all()

    def test_index_redirects_to_login(self):
        res = self.client.get('/', follow_redirects=False)
        assert res.status_code in (200, 302)

    def test_login_page_loads(self):
        res = self.client.get('/login')
        assert res.status_code == 200

    def test_register_and_login(self):
        res = self.client.post('/register', data={
            'username':         'testuser',
            'email':            'test@example.com',
            'password':         'password123',
            'confirm_password': 'password123',
            'consent':          'on',
        }, follow_redirects=True)
        assert res.status_code == 200

    def test_dashboard_requires_login(self):
        res = self.client.get('/dashboard', follow_redirects=False)
        assert res.status_code == 302   # redirect to login

    def test_capture_requires_login(self):
        res = self.client.get('/capture', follow_redirects=False)
        assert res.status_code == 302

    def test_history_requires_login(self):
        res = self.client.get('/history', follow_redirects=False)
        assert res.status_code == 302

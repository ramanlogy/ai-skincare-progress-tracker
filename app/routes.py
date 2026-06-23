
import os
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, Response
from flask_login import login_user, logout_user, login_required, current_user
from app.models import User, SkinScan, db
from app.pipeline import analyse_image, _allowed

main_bp = Blueprint('main', __name__)

# --- START CSV EXPORT ---
import csv
@main_bp.route('/history/export')
@login_required
def export_csv():
    scans = SkinScan.query.filter_by(user_id=current_user.id).order_by(SkinScan.captured_at.asc()).all()
    
    def generate():
        yield "Date,Status,Overall Score,Clarity,Redness,Texture,Overall Change,Notes\n"
        for scan in scans:
            yield f"{scan.captured_at.strftime('%Y-%m-%d %H:%M')},{scan.analysis_status},{scan.overall_score},{scan.acne_score},{scan.redness_score},{scan.texture_score},{scan.overall_change},\"{scan.notes}\"\n"

    return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": "attachment; filename=skincare_history.csv"})
# --- END CSV EXPORT ---

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template_string(INDEX_T, show_register=False)


@main_bp.route('/register', methods=['GET', 'POST'])
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


@main_bp.route('/login', methods=['GET', 'POST'])
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


@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@main_bp.route('/dashboard')
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


@main_bp.route('/capture')
@login_required
def capture_page():
    return render_template('capture.html')


@main_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files:
        flash('No image provided.', 'error')
        return redirect(url_for('capture_page'))

    file  = request.files['image']
    notes = request.form.get('notes', '')
    skin_type = request.form.get('skin_type', 'Normal')

    if not file.filename or not _allowed(file.filename):
        flash('Unsupported file type. Use PNG, JPG or JPEG.', 'error')
        return redirect(url_for('capture_page'))

    ext      = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{current_user.id}_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    scan = SkinScan(user_id=current_user.id, image_filename=filename,
                    notes=notes, skin_type=skin_type, analysis_status='pending')
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


@main_bp.route('/history')
@login_required
def history():
    scans = (SkinScan.query.filter_by(user_id=current_user.id)
             .order_by(SkinScan.captured_at.desc()).all())
    return render_template('history.html', scans=scans)


@main_bp.route('/scan/<int:scan_id>/delete', methods=['POST'])
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


@main_bp.route('/account/delete', methods=['POST'])
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


@main_bp.route('/api/scans')
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

    app.run(debug=True, host='0.0.0.0', port=5000)

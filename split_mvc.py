import os
import re

with open('run.py', 'r') as f:
    lines = f.readlines()

index_start = 0
for i, line in enumerate(lines):
    if line.startswith('@app.route(\'/\')'):
        index_start = i
        break

routes_code = lines[index_start:]

# Replace @app.route with @main_bp.route
routes_code = [line.replace('@app.route', '@main_bp.route') for line in routes_code]
routes_code = "".join(routes_code)

# Replace render_template_string
routes_code = re.sub(r"render_template_string\(INDEX_T\)", "render_template('index.html')", routes_code)
routes_code = re.sub(r"render_template_string\(DASHBOARD_T,.*?\)", lambda m: m.group(0).replace('render_template_string(DASHBOARD_T,', 'render_template(\'dashboard.html\','), routes_code)
routes_code = re.sub(r"render_template_string\(CAPTURE_T\)", "render_template('capture.html')", routes_code)
routes_code = re.sub(r"render_template_string\(HISTORY_T,.*?\)", lambda m: m.group(0).replace('render_template_string(HISTORY_T,', 'render_template(\'history.html\','), routes_code)

# Write app/routes.py
header = """
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
        yield "Date,Status,Overall Score,Clarity,Redness,Texture,Overall Change,Notes\\n"
        for scan in scans:
            yield f"{scan.captured_at.strftime('%Y-%m-%d %H:%M')},{scan.analysis_status},{scan.overall_score},{scan.acne_score},{scan.redness_score},{scan.texture_score},{scan.overall_change},\\"{scan.notes}\\"\\n"

    return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": "attachment; filename=skincare_history.csv"})
# --- END CSV EXPORT ---

"""

with open('app/routes.py', 'w') as f:
    f.write(header + routes_code)

# Now extract Pipeline
user_start = 0
for i, line in enumerate(lines):
    if line.startswith('class User'):
        user_start = i
        break

pipeline_code = lines[50:user_start]
pipeline_str = "".join(pipeline_code)

pipeline_header = """
import os
import cv2
import numpy as np
from PIL import Image
from math import pi
import logging

logger = logging.getLogger(__name__)

"""
# Need facenet-pytorch if it's there
if 'facenet_pytorch' in "".join(lines[0:50]):
    pipeline_header += "from facenet_pytorch import MTCNN\n"

with open('app/pipeline.py', 'w') as f:
    f.write(pipeline_header + pipeline_str)

print("Split completed.")

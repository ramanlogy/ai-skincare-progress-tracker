from . import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    consent_given = db.Column(db.Boolean, default=False)

    scans = db.relationship('SkinScan', backref='user', lazy=True,
                            cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class SkinScan(db.Model):
    __tablename__ = 'skin_scans'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename  = db.Column(db.String(256), nullable=False)
    captured_at     = db.Column(db.DateTime, default=datetime.utcnow)
    notes           = db.Column(db.Text, default='')

    # Analysis Results
    face_detected   = db.Column(db.Boolean, default=False)
    acne_count      = db.Column(db.Float, default=0.0)
    acne_score      = db.Column(db.Float, default=50.0)
    redness_score   = db.Column(db.Float, default=50.0)
    texture_score   = db.Column(db.Float, default=50.0)
    overall_score   = db.Column(db.Float, default=50.0)

    # Comparison with baseline
    acne_change     = db.Column(db.Float, default=0.0)
    redness_change  = db.Column(db.Float, default=0.0)
    texture_change  = db.Column(db.Float, default=0.0)
    overall_change  = db.Column(db.Float, default=0.0)

    analysis_status = db.Column(db.String(20), default='pending')  # pending, complete, failed

    def to_dict(self):
        return {
            'id':              self.id,
            'image_filename':  self.image_filename,
            'captured_at':     self.captured_at.strftime('%Y-%m-%d %H:%M'),
            'notes':           self.notes,
            'face_detected':   self.face_detected,
            'acne_count':      round(self.acne_count, 1),
            'acne_score':      round(self.acne_score, 1),
            'redness_score':   round(self.redness_score, 1),
            'texture_score':   round(self.texture_score, 1),
            'overall_score':   round(self.overall_score, 1),
            'acne_change':     round(self.acne_change, 1),
            'redness_change':  round(self.redness_change, 1),
            'texture_change':  round(self.texture_change, 1),
            'overall_change':  round(self.overall_change, 1),
            'analysis_status': self.analysis_status,
        }

    def __repr__(self):
        return f'<SkinScan {self.id} user={self.user_id}>'

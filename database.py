from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import bcrypt

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    is_medical_professional = db.Column(db.Boolean, default=False)
    
    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

# ... e la classe IntubationData
class IntubationData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # Patient demographics
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)  # 'Male', 'Female'
    weight = db.Column(db.Float, nullable=False)  # kg
    height = db.Column(db.Float, nullable=False)  # cm
    bmi = db.Column(db.Float, nullable=False)
    
    # Airway assessment parameters (from literature)
    mallampati = db.Column(db.Integer, nullable=False)  # 1-4
    neck_circumference = db.Column(db.Float, nullable=False)  # cm
    thyromental_distance = db.Column(db.Float, nullable=False)  # cm (DTM)
    interincisor_distance = db.Column(db.Float, nullable=False)  # cm (DII)
    al_ganzuri_score = db.Column(db.Integer, nullable=False)  # 0-?
    stop_bang_score = db.Column(db.Integer, nullable=False)  # 0-8
    
    # Additional airway parameters
    mouth_opening = db.Column(db.Float)  # cm
    upper_lip_bite_test = db.Column(db.String(20))  # Class I, II, III
    neck_mobility = db.Column(db.String(20))  # Normal, Limited
    
    # Outcome
    cormack_grade = db.Column(db.Integer, nullable=False)  # 1-4
    difficult_intubation = db.Column(db.Boolean, nullable=False)  # True if Cormack 3-4
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    verified = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'age': self.age,
            'gender': self.gender,
            'weight': self.weight,
            'height': self.height,
            'bmi': self.bmi,
            'mallampati': self.mallampati,
            'neck_circumference': self.neck_circumference,
            'thyromental_distance': self.thyromental_distance,
            'interincisor_distance': self.interincisor_distance,
            'al_ganzuri_score': self.al_ganzuri_score,
            'stop_bang_score': self.stop_bang_score,
            'mouth_opening': self.mouth_opening,
            'upper_lip_bite_test': self.upper_lip_bite_test,
            'neck_mobility': self.neck_mobility,
            'cormack_grade': self.cormack_grade,
            'difficult_intubation': self.difficult_intubation,
            'created_at': self.created_at.isoformat(),
            'contributor': self.contributor.username,
            'verified': self.verified
        }
    
    def calculate_bmi(self):
        """Calculate BMI from weight and height"""
        if self.height > 0:
            return self.weight / ((self.height / 100) ** 2)
        return 0

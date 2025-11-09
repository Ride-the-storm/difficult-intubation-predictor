import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-123')

# Database setup
database_url = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Simple Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

class IntubationData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    weight = db.Column(db.Float)
    height = db.Column(db.Float)
    bmi = db.Column(db.Float)
    mallampati = db.Column(db.Integer)
    neck_circumference = db.Column(db.Float)
    thyromental_distance = db.Column(db.Float)
    interincisor_distance = db.Column(db.Float)
    stop_bang_score = db.Column(db.Integer)
    cormack_grade = db.Column(db.Integer)
    difficult_intubation = db.Column(db.Boolean)
    user_id = db.Column(db.Integer)
    verified = db.Column(db.Boolean, default=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Simple ML placeholder
class SimpleModel:
    def __init__(self):
        self.is_trained = True  # Always return a prediction
    
    def predict(self, features):
        # Simple rule-based prediction for now
        risk_factors = 0
        if features.get('mallampati', 1) >= 3:
            risk_factors += 1
        if features.get('neck_circumference', 0) > 40:
            risk_factors += 1
        if features.get('stop_bang_score', 0) >= 3:
            risk_factors += 1
            
        probability = min(risk_factors * 0.3, 0.9)
        prediction = risk_factors >= 2
        
        if probability > 0.7:
            risk_level = 'High'
        elif probability > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
            
        return {
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level
        }

model = SimpleModel()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password_hash, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_pw = generate_password_hash(request.form['password'])
        user = User(username=request.form['username'], password_hash=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    data_count = IntubationData.query.count()
    return render_template('dashboard.html', 
                         data_count=data_count, 
                         model_trained=model.is_trained)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            features = {
                'age': int(request.form['age']),
                'mallampati': int(request.form['mallampati']),
                'neck_circumference': float(request.form['neck_circumference']),
                'stop_bang_score': int(request.form['stop_bang_score'])
            }
            result = model.predict(features)
            return render_template('predict_result.html', prediction=result)
        except Exception as e:
            flash(f'Error: {str(e)}')
    return render_template('predict.html')

@app.route('/input-data', methods=['GET', 'POST'])
@login_required
def input_data():
    if request.method == 'POST':
        data = IntubationData(
            age=int(request.form['age']),
            gender=request.form['gender'],
            weight=float(request.form['weight']),
            height=float(request.form['height']),
            bmi=float(request.form['weight']) / ((float(request.form['height'])/100) ** 2),
            mallampati=int(request.form['mallampati']),
            neck_circumference=float(request.form['neck_circumference']),
            thyromental_distance=float(request.form.get('thyromental_distance', 6.5)),
            interincisor_distance=float(request.form.get('interincisor_distance', 4.0)),
            stop_bang_score=int(request.form['stop_bang_score']),
            cormack_grade=int(request.form['cormack_grade']),
            difficult_intubation=int(request.form['cormack_grade']) >= 3,
            user_id=current_user.id
        )
        db.session.add(data)
        db.session.commit()
        flash('Data saved successfully!')
        return redirect(url_for('dashboard'))
    return render_template('input_data.html')

# Initialize app
def init_app():
    with app.app_context():
        db.create_all()
        # Create admin user if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created: admin/admin123")

init_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

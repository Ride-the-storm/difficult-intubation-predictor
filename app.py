import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_login import LoginManager, current_user, login_required
from database import db, User, IntubationData
from model import DifficultIntubationModel
from auth import auth_bp  # AGGIUNGI QUESTA RIGA

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Database configuration
database_url = os.environ.get('DATABASE_URL')
if database_url:
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'  # CAMBIA IN 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize model
model = DifficultIntubationModel(retrain_threshold=5)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth_bp)  # AGGIUNGI QUESTA RIGA

# Le tue route esistenti rimangono uguali...
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    model_info = model.get_model_info()
    user_data_count = IntubationData.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html', 
                         model_info=model_info, 
                         user_data_count=user_data_count)

# ... tutte le altre route rimangono uguali

def init_db():
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                is_admin=True,
                is_medical_professional=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created: admin/admin123")
        
        # Initialize model
        model.load_model()
        if not model.is_trained:
            print("Training initial model...")
            model.train(force_retrain=True)

# Initialize when app starts
init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

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

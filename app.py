import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_login import LoginManager, current_user, login_required
from database import db, User, IntubationData
from model import DifficultIntubationModel
from auth import auth_bp

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
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize model
model = DifficultIntubationModel(retrain_threshold=5)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth_bp)

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

@app.route('/input-data', methods=['GET', 'POST'])
@login_required
def input_data():
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form['age'])
            gender = request.form['gender']
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            mallampati = int(request.form['mallampati'])
            neck_circumference = float(request.form['neck_circumference'])
            thyromental_distance = float(request.form['thyromental_distance'])
            interincisor_distance = float(request.form['interincisor_distance'])
            al_ganzuri_score = int(request.form['al_ganzuri_score'])
            stop_bang_score = int(request.form['stop_bang_score'])
            mouth_opening = float(request.form.get('mouth_opening', 4.0))
            upper_lip_bite_test = request.form['upper_lip_bite_test']
            neck_mobility = request.form['neck_mobility']
            cormack_grade = int(request.form['cormack_grade'])
            
            # Calculate BMI
            bmi = weight / ((height / 100) ** 2)
            
            # Determine difficult intubation
            difficult_intubation = cormack_grade >= 3
            
            # Create new data point
            new_data = IntubationData(
                age=age,
                gender=gender,
                weight=weight,
                height=height,
                bmi=bmi,
                mallampati=mallampati,
                neck_circumference=neck_circumference,
                thyromental_distance=thyromental_distance,
                interincisor_distance=interincisor_distance,
                al_ganzuri_score=al_ganzuri_score,
                stop_bang_score=stop_bang_score,
                mouth_opening=mouth_opening,
                upper_lip_bite_test=upper_lip_bite_test,
                neck_mobility=neck_mobility,
                cormack_grade=cormack_grade,
                difficult_intubation=difficult_intubation,
                user_id=current_user.id,
                verified=current_user.is_medical_professional
            )
            
            db.session.add(new_data)
            db.session.commit()
            
            flash('Data submitted successfully!', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            flash(f'Error submitting data: {str(e)}', 'error')
    
    return render_template('input_data.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form['age'])
            gender = 1 if request.form['gender'] == 'Male' else 0
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            mallampati = int(request.form['mallampati'])
            neck_circumference = float(request.form['neck_circumference'])
            thyromental_distance = float(request.form['thyromental_distance'])
            interincisor_distance = float(request.form['interincisor_distance'])
            al_ganzuri_score = int(request.form['al_ganzuri_score'])
            stop_bang_score = int(request.form['stop_bang_score'])
            mouth_opening = float(request.form.get('mouth_opening', 4.0))
            upper_lip_bite_test = 1 if request.form['upper_lip_bite_test'] == 'Class I' else 2 if request.form['upper_lip_bite_test'] == 'Class II' else 3
            neck_mobility = 1 if request.form['neck_mobility'] == 'Normal' else 0
            
            # Calculate BMI
            bmi = weight / ((height / 100) ** 2)
            
            # Prepare features
            features = {
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height,
                'bmi': bmi,
                'mallampati': mallampati,
                'neck_circumference': neck_circumference,
                'thyromental_distance': thyromental_distance,
                'interincisor_distance': interincisor_distance,
                'al_ganzuri_score': al_ganzuri_score,
                'stop_bang_score': stop_bang_score,
                'mouth_opening': mouth_opening,
                'upper_lip_bite_test': upper_lip_bite_test,
                'neck_mobility': neck_mobility
            }
            
            # Make prediction
            prediction_result = model.predict(features)
            
            return render_template('predict_result.html', 
                                prediction=prediction_result,
                                features=request.form)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

# SIMPLE DEBUG ROUTES
@app.route('/debug-model')
@login_required
def debug_model():
    if not current_user.is_admin:
        return "Admin access required"
    
    model_info = model.get_model_info()
    from database import IntubationData
    total = IntubationData.query.count()
    verified = IntubationData.query.filter_by(verified=True).count()
    
    return f"""
    Model Trained: {model_info['is_trained']}<br>
    Model Version: v{model_info['model_version']}<br>
    Total Data: {total}<br>
    Verified Data: {verified}<br>
    <a href="/train-now">TRAIN MODEL</a>
    """

@app.route('/train-now')
@login_required
def train_now():
    if not current_user.is_admin:
        return "Admin access required"
    
    result = model.train(force_retrain=True)
    if result:
        return f"Model trained! Accuracy: {result.get('accuracy', 'N/A')}"
    else:
        return "Training failed - check Heroku logs"

@app.route('/fix-data')
@login_required
def fix_data():
    if not current_user.is_admin:
        return "Admin access required"
    
    from database import IntubationData, db
    # Verify all data
    IntubationData.query.update({'verified': True})
    db.session.commit()
    return "All data verified! <a href='/train-now'>Train Model</a>"

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

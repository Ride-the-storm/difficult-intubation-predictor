import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_login import LoginManager, current_user, login_required
from database import db, User, IntubationData
from model import DifficultIntubationModel
from auth import auth_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Database configuration
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace("postgres://", "postgresql://", 1)
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)


# After this line in app.py:

# ADD THIS:
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Create admin user if doesn't exist
        from database import User
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
            print("✅ Admin user created")
            
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        
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
            
            # Determine difficult intubation (Cormack 3-4 considered difficult)
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
                verified=current_user.is_medical_professional  # Auto-verify if medical professional
            )
            
            db.session.add(new_data)
            db.session.commit()
            
            # Check if retraining is needed
            if model.check_retrain_needed():
                flash('New training data available! Model will be retrained.', 'info')
            
            flash('Data submitted successfully! Thank you for your contribution.', 'success')
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
            
            # Prepare features dictionary
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
            
            print(f"Making prediction with features: {features}")  # Debug log
            
            # Make prediction
            prediction_result = model.predict(features)
            
            print(f"Prediction result: {prediction_result}")  # Debug log
            
            return render_template('predict_result.html', 
                                prediction=prediction_result,
                                features=request.form)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")  # Debug log
            flash(f'Error making prediction: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')
@app.route('/model-info')
@login_required
def model_info():
    info = model.get_model_info()
    return jsonify(info)

@app.route('/user-data')
@login_required
def user_data():
    user_data_points = IntubationData.query.filter_by(user_id=current_user.id).all()
    return jsonify([dp.to_dict() for dp in user_data_points])

@app.route('/retrain-model', methods=['POST'])
@login_required
def retrain_model():
    if not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        performance = model.train(force_retrain=True)
        if performance:
            return jsonify({
                'success': True,
                'performance': performance
            })
        else:
            return jsonify({'error': 'Training failed or not needed'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Admin routes
@app.route('/admin/pending-data')
@login_required
def pending_data():
    if not current_user.is_admin:
        flash('Admin access required', 'error')
        return redirect(url_for('dashboard'))
    
    pending_data = IntubationData.query.filter_by(verified=False).all()
    return jsonify([dp.to_dict() for dp in pending_data])

@app.route('/admin/verify-data/<int:data_id>', methods=['POST'])
@login_required
def verify_data(data_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    data_point = IntubationData.query.get_or_404(data_id)
    data_point.verified = True
    db.session.commit()
    
    # Check if retraining is needed
    if model.check_retrain_needed():
        model.train()
    
    return jsonify({'success': True})

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
        
        # Load or train initial model
        model.load_model()
        if not model.is_trained:
            print("Training initial model...")
            model.train(force_retrain=True)

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

@app.route('/train-now')
def train_now():
    try:
        from database import IntubationData
        
        # Check data stats
        total_data = IntubationData.query.count()
        verified_data = IntubationData.query.filter_by(verified=True).count()
        
        # Force train the model
        result = model.train(force_retrain=True)
        
        if result:
            return f"""
            ✅ Model trained successfully!<br>
            <b>Accuracy:</b> {result.get('accuracy', 'N/A')}<br>
            <b>Training Data:</b> {result.get('training_size', 0)} points<br>
            <b>Total Data:</b> {total_data}<br>
            <b>Verified Data:</b> {verified_data}<br>
            <a href="/predict">Test Prediction Now</a>
            """
        else:
            return f"""
            ❌ Training failed<br>
            <b>Possible reasons:</b><br>
            - Not enough verified data (need 20+, have {verified_data})<br>
            - Data imbalance<br>
            - Technical issue<br>
            <a href="/debug-data">Check Data</a>
            """
    except Exception as e:
        return f"Training error: {str(e)}"

@app.route('/debug-data')
def debug_data():
    from database import IntubationData
    import pandas as pd
    
    # Get all verified data
    data = IntubationData.query.filter_by(verified=True).all()
    
    if not data:
        return "No verified data found!"
    
    # Convert to DataFrame for analysis
    records = []
    for record in data:
        records.append({
            'age': record.age,
            'gender': record.gender,
            'bmi': record.bmi,
            'mallampati': record.mallampati,
            'neck_circumference': record.neck_circumference,
            'thyromental_distance': record.thyromental_distance,
            'interincisor_distance': record.interincisor_distance,
            'stop_bang_score': record.stop_bang_score,
            'cormack_grade': record.cormack_grade,
            'difficult_intubation': record.difficult_intubation
        })
    
    df = pd.DataFrame(records)
    
    # Basic stats
    stats = f"""
    <h3>Data Analysis</h3>
    <b>Total Verified Records:</b> {len(data)}<br>
    <b>Easy Intubations:</b> {sum(1 for r in data if not r.difficult_intubation)}<br>
    <b>Difficult Intubations:</b> {sum(1 for r in data if r.difficult_intubation)}<br>
    <b>Data Balance:</b> {'✅ Good' if (sum(1 for r in data if r.difficult_intubation) >= 5) else '❌ Need more difficult cases'}<br>
    """
    
    return stats
@app.route('/force-train-debug')
def force_train_debug():
    try:
        from database import IntubationData
        
        # Check data before training
        total_data = IntubationData.query.count()
        verified_data = IntubationData.query.filter_by(verified=True).count()
        
        # Check data balance
        easy_cases = IntubationData.query.filter_by(verified=True, difficult_intubation=False).count()
        difficult_cases = IntubationData.query.filter_by(verified=True, difficult_intubation=True).count()
        
        # Debug: Check model current state
        current_state = f"""
        <h3>Pre-Training Check</h3>
        <b>Total Data:</b> {total_data}<br>
        <b>Verified Data:</b> {verified_data}<br>
        <b>Easy Cases:</b> {easy_cases}<br>
        <b>Difficult Cases:</b> {difficult_cases}<br>
        <b>Model Trained:</b> {model.is_trained}<br>
        <b>Model Version:</b> {model.model_version}<br>
        """
        
        # Force train
        result = model.train(force_retrain=True)
        
        if result:
            return current_state + f"""
            <h3>✅ Training Successful!</h3>
            <b>Accuracy:</b> {result.get('accuracy', 'N/A')}<br>
            <b>Training Size:</b> {result.get('training_size', 'N/A')}<br>
            <b>New Version:</b> v{model.model_version}<br>
            <a href="/dashboard">Check Dashboard</a> | 
            <a href="/test-prediction">Test Prediction</a>
            """
        else:
            return current_state + f"""
            <h3>❌ Training Failed</h3>
            <p>Training returned None. Possible issues:</p>
            <ul>
                <li>Not enough verified data (have {verified_data})</li>
                <li>Data imbalance (need both easy and difficult cases)</li>
                <li>Training error (check logs)</li>
            </ul>
            <a href="/debug-data">Check Data Details</a>
            """
            
    except Exception as e:
        return f"Training error: {str(e)}"

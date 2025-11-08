# Auto-create tables on startup
with app.app_context():
    try:
        db.create_all()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating tables: {e}")


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
            
            # Make prediction
            prediction_result = model.predict(features)
            
            return render_template('predict_result.html', 
                                prediction=prediction_result,
                                features=request.form)
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
    
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

# Add these debug routes anywhere in your app.py

@app.route('/debug')
def debug():
    import os
    db_url = os.environ.get('DATABASE_URL', 'Not set')
    # Hide password for security
    if 'postgres' in db_url:
        parts = db_url.split('@')
        if len(parts) > 1:
            db_url = 'postgres://[hidden]@' + parts[1]
    
    return f"""
    <h3>App Debug Info</h3>
    <b>Database URL:</b> {db_url}<br>
    <b>Secret Key Set:</b> {bool(os.environ.get('SECRET_KEY'))}<br>
    <b>Python Version:</b> {os.environ.get('PYTHON_VERSION', 'Not set')}<br>
    <a href="/debug-db">Check Database</a><br>
    <a href="/debug-tables">Check Tables</a>
    """

@app.route('/debug-db')
def debug_db():
    from database import User
    try:
        user_count = User.query.count()
        return f"✅ Database accessible. Users in database: {user_count}"
    except Exception as e:
        return f"❌ Database error: {str(e)}"

@app.route('/debug-tables')
def debug_tables():
    from database import db, User, IntubationData
    try:
        # Try to query all tables
        users = User.query.count()
        data_points = IntubationData.query.count()
        return f"""
        ✅ Tables accessible:<br>
        - Users: {users}<br>
        - Intubation Data: {data_points}
        """
    except Exception as e:
        return f"❌ Table error: {str(e)}"

@app.route('/create-admin')
def create_admin():
    from database import User, db
    try:
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
            return "✅ Admin user created: admin/admin123"
        else:
            return "✅ Admin user already exists"
    except Exception as e:
        return f"❌ Error creating admin: {str(e)}"

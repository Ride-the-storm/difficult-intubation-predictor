import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime
from database import IntubationData, db

class DifficultIntubationModel:
    def __init__(self, retrain_threshold=5):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.retrain_threshold = retrain_threshold
        self.last_training_size = 0
        self.model_version = 1
        self.model_performance = {}
        
    def load_data_from_db(self, only_verified=True):
        """Load training data from database"""
        if only_verified:
            data_points = IntubationData.query.filter_by(verified=True).all()
        else:
            data_points = IntubationData.query.all()
        
        if len(data_points) < 20:
            # If not enough data, generate initial synthetic data based on medical literature
            return self.generate_initial_medical_data()
        
        data = []
        for point in data_points:
            data.append([
                point.age,
                1 if point.gender == 'Male' else 0,  # Encode gender
                point.weight,
                point.height,
                point.bmi,
                point.mallampati,
                point.neck_circumference,
                point.thyromental_distance,
                point.interincisor_distance,
                point.al_ganzuri_score,
                point.stop_bang_score,
                point.mouth_opening or 4.0,  # Default value
                1 if point.upper_lip_bite_test == 'Class I' else 2 if point.upper_lip_bite_test == 'Class II' else 3,
                1 if point.neck_mobility == 'Normal' else 0,
                point.difficult_intubation  # Target
            ])
        
        df = pd.DataFrame(data, columns=[
            'age', 'gender', 'weight', 'height', 'bmi', 'mallampati', 
            'neck_circumference', 'thyromental_distance', 'interincisor_distance',
            'al_ganzuri_score', 'stop_bang_score', 'mouth_opening',
            'upper_lip_bite_test', 'neck_mobility', 'difficult_intubation'
        ])
        return df
    
    def generate_initial_medical_data(self):
        """Generate initial synthetic data based on medical literature"""
        print("Generating initial synthetic medical data...")
        np.random.seed(42)
        n_samples = 200
        
        # Based on medical literature prevalence
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),  # 0: Female, 1: Male
            'weight': np.random.normal(70, 15, n_samples),  # kg
            'height': np.random.normal(170, 10, n_samples),  # cm
            'mallampati': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'neck_circumference': np.random.normal(38, 4, n_samples),  # cm
            'thyromental_distance': np.random.normal(6.5, 1.5, n_samples),  # cm
            'interincisor_distance': np.random.normal(4.5, 1.0, n_samples),  # cm
            'al_ganzuri_score': np.random.randint(0, 4, n_samples),
            'stop_bang_score': np.random.randint(0, 9, n_samples),
            'mouth_opening': np.random.normal(4.0, 0.8, n_samples),
            'upper_lip_bite_test': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
            'neck_mobility': np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
        }
        
        df = pd.DataFrame(data)
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        
        # Generate difficult intubation outcome based on known risk factors
        # Based on medical literature: higher mallampati, neck circumference, stop-bang, lower thyromental distance
        difficulty_prob = (
            (df['mallampati'] - 1) * 0.15 +  # Mallampati 3-4 increases risk
            (df['neck_circumference'] > 40) * 0.2 +  # Large neck
            (df['thyromental_distance'] < 6) * 0.15 +  # Short thyromental distance
            (df['stop_bang_score'] >= 3) * 0.2 +  # High STOP-BANG
            (df['upper_lip_bite_test'] == 3) * 0.15 +  # Class III ULBT
            (df['neck_mobility'] == 0) * 0.1 +  # Limited neck mobility
            (df['interincisor_distance'] < 3) * 0.15  # Small mouth opening
        )
        
        # Add some noise and ensure probabilities are between 0 and 1
        difficulty_prob = np.clip(difficulty_prob + np.random.normal(0, 0.1, n_samples), 0, 0.9)
        df['difficult_intubation'] = np.random.binomial(1, difficulty_prob)
        
        # Calculate Cormack grade based on difficulty (simplified)
        df['cormack_grade'] = np.where(
            df['difficult_intubation'] == 1, 
            np.random.choice([3, 4], n_samples, p=[0.7, 0.3]),
            np.random.choice([1, 2], n_samples, p=[0.6, 0.4])
        )
        
        # Add these to database as verified synthetic data
        for _, row in df.iterrows():
            data_point = IntubationData(
                age=int(row['age']),
                gender='Male' if row['gender'] == 1 else 'Female',
                weight=row['weight'],
                height=row['height'],
                bmi=row['bmi'],
                mallampati=int(row['mallampati']),
                neck_circumference=row['neck_circumference'],
                thyromental_distance=row['thyromental_distance'],
                interincisor_distance=row['interincisor_distance'],
                al_ganzuri_score=int(row['al_ganzuri_score']),
                stop_bang_score=int(row['stop_bang_score']),
                mouth_opening=row['mouth_opening'],
                upper_lip_bite_test='Class I' if row['upper_lip_bite_test'] == 1 else 'Class II' if row['upper_lip_bite_test'] == 2 else 'Class III',
                neck_mobility='Normal' if row['neck_mobility'] == 1 else 'Limited',
                cormack_grade=int(row['cormack_grade']),
                difficult_intubation=bool(row['difficult_intubation']),
                user_id=1,  # System user
                verified=True
            )
            db.session.add(data_point)
        
        db.session.commit()
        return df
    
    def check_retrain_needed(self):
        """Check if model needs retraining based on new data"""
        current_data_size = IntubationData.query.filter_by(verified=True).count()
        new_data_points = current_data_size - self.last_training_size
        
        return new_data_points >= self.retrain_threshold
    
    def train(self, force_retrain=False):
        """Train or retrain the model"""
        if not force_retrain and not self.check_retrain_needed():
            print("No retraining needed yet.")
            return None
        
        print("Training difficult intubation prediction model...")
        df = self.load_data_from_db()
        
        if len(df) < 20:
            print("Not enough data for training.")
            return None
        
        # Prepare features and target
        feature_columns = [
            'age', 'gender', 'weight', 'height', 'bmi', 'mallampati', 
            'neck_circumference', 'thyromental_distance', 'interincisor_distance',
            'al_ganzuri_score', 'stop_bang_score', 'mouth_opening',
            'upper_lip_bite_test', 'neck_mobility'
        ]
        
        X = df[feature_columns]
        y = df['difficult_intubation']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model - Using Random Forest for medical data
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            class_weight='balanced'  # Important for imbalanced medical data
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importance
        feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        
        self.is_trained = True
        self.last_training_size = len(df)
        self.model_version += 1
        
        # Save performance metrics
        self.model_performance = {
            'accuracy': accuracy,
            'training_size': len(df),
            'positive_cases': sum(y),
            'negative_cases': len(y) - sum(y),
            'last_trained': datetime.now().isoformat(),
            'version': self.model_version,
            'feature_importance': feature_importance
        }
        
        # Save the model
        self.save_model()
        
        print(f"Model v{self.model_version} trained with Accuracy: {accuracy:.3f}")
        print(f"Feature importance: {feature_importance}")
        return self.model_performance
    
    def predict(self, features_dict):
    """Make prediction for difficult intubation"""
    if not self.is_trained:
        self.load_model()
    
    if not self.is_trained:
        raise Exception("Model not trained and no saved model found.")
    
    # Convert features to array in correct order
    feature_columns = [
        'age', 'gender', 'weight', 'height', 'bmi', 'mallampati', 
        'neck_circumference', 'thyromental_distance', 'interincisor_distance',
        'al_ganzuri_score', 'stop_bang_score', 'mouth_opening',
        'upper_lip_bite_test', 'neck_mobility'
    ]
    
    # Ensure all features are present and in correct order
    features_array = []
    for col in feature_columns:
        if col in features_dict:
            features_array.append(features_dict[col])
        else:
            # Provide default value if feature is missing
            if col == 'mouth_opening':
                features_array.append(4.0)  # Default mouth opening
            else:
                features_array.append(0)  # Default for other features
    
    features_array = np.array(features_array).reshape(1, -1)
    
    try:
        features_scaled = self.scaler.transform(features_array)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Determine risk level
        prob_difficult = float(probability[1])
        if prob_difficult > 0.7:
            risk_level = 'High'
        elif prob_difficult > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'prediction': bool(prediction),
            'probability': prob_difficult,
            'risk_level': risk_level
        }
    
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")    
    def save_model(self):
        """Save model and scaler to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'last_training_size': self.last_training_size,
            'model_version': self.model_version,
            'model_performance': self.model_performance
        }
        joblib.dump(model_data, 'difficult_intubation_model.joblib')
    
    def load_model(self):
        """Load model and scaler from disk"""
        try:
            model_data = joblib.load('difficult_intubation_model.joblib')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.last_training_size = model_data['last_training_size']
            self.model_version = model_data['model_version']
            self.model_performance = model_data['model_performance']
            print(f"Model v{self.model_version} loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Please train the model first.")
    
    def get_model_info(self):
        """Get model information and performance"""
        return {
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'last_training_size': self.last_training_size,
            'performance': self.model_performance,
            'total_data_points': IntubationData.query.count(),
            'verified_data_points': IntubationData.query.filter_by(verified=True).count()
        }

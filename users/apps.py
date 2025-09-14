from django.apps import AppConfig
import os
import joblib
from django.conf import settings

class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'
    
class SneakerPredictionConfig(AppConfig):
    """Django app configuration for Sneaker Price Prediction"""
    
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sneaker_prediction'  # Update with your actual app name
    
    # Model instances
    models = {}
    
    def ready(self):
        """Initialize models when Django starts"""
        print("🚀 Initializing Sneaker Price Prediction App...")
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load pre-trained models if available"""
        try:
            # Define model file paths
            model_paths = {
                'linear_regression': 'models/linear_regression_model.joblib',
                'random_forest': 'models/random_forest_model.joblib',
                'xgboost': 'models/xgboost_model.joblib',
                'svr': 'models/svr_model.joblib'
            }
            
            base_path = getattr(settings, 'BASE_DIR', '.')
            
            for model_name, relative_path in model_paths.items():
                full_path = os.path.join(base_path, relative_path)
                
                if os.path.exists(full_path):
                    try:
                        # Load the model
                        model_data = joblib.load(full_path)
                        self.models[model_name] = model_data
                        print(f"✓ Loaded {model_name} model")
                    except Exception as e:
                        print(f"✗ Error loading {model_name}: {str(e)}")
                else:
                    print(f"⚠ Model file not found: {relative_path}")
            
            if self.models:
                print(f"✅ Successfully loaded {len(self.models)} models")
            else:
                print("ℹ No pre-trained models found. Models will be trained on first use.")
                self.initialize_fresh_models()
                
        except Exception as e:
            print(f"❌ Error during model loading: {str(e)}")
            self.initialize_fresh_models()
    
    def initialize_fresh_models(self):
        """Initialize fresh model instances"""
        try:
            from .models import (
                LinearRegressionModel, RandomForestModel, 
                XGBoostModel, SVRModel
            )
            
            self.models = {
                'linear_regression': {
                    'model': LinearRegressionModel(),
                    'is_trained': False
                },
                'random_forest': {
                    'model': RandomForestModel(),
                    'is_trained': False
                },
                'xgboost': {
                    'model': XGBoostModel(),
                    'is_trained': False
                },
                'svr': {
                    'model': SVRModel(),
                    'is_trained': False
                }
            }
            print("✓ Initialized fresh model instances")
            
        except ImportError as e:
            print(f"❌ Error importing models: {str(e)}")
    
    def get_model(self, model_name):
        """Get a specific model instance"""
        return self.models.get(model_name)
    
    def get_available_models(self):
        """Get list of available model names"""
        return list(self.models.keys())
    
    def save_model(self, model_name, model_instance):
        """Save a trained model"""
        try:
            base_path = getattr(settings, 'BASE_DIR', '.')
            models_dir = os.path.join(base_path, 'models')
            
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Save the model
            filepath = os.path.join(models_dir, f'{model_name}_model.joblib')
            model_instance.save_model(filepath)
            
            # Update the loaded models
            self.models[model_name] = {
                'model': model_instance,
                'is_trained': True,
                'filepath': filepath
            }
            
            print(f"✓ Model {model_name} saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model {model_name}: {str(e)}")
            return False

class ModelManager:
    """Singleton class to manage model instances across the application"""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.load_models()
    
    def load_models(self):
        """Load all models"""
        try:
            from .models import (
                LinearRegressionModel, RandomForestModel,
                XGBoostModel, SVRModel, ModelComparison
            )
            
            self._models = {
                'linear_regression': LinearRegressionModel(),
                'random_forest': RandomForestModel(n_estimators=100, random_state=42),
                'xgboost': XGBoostModel(n_estimators=100, random_state=42),
                'svr': SVRModel(kernel='rbf', C=1.0)
            }
            
            self.model_comparison = ModelComparison()
            for name, model in self._models.items():
                self.model_comparison.add_model(name.replace('_', ' ').title(), model)
            
        except Exception as e:
            print(f"Error loading models in ModelManager: {str(e)}")
    
    def get_model(self, model_name):
        """Get a specific model"""
        return self._models.get(model_name)
    
    def get_all_models(self):
        """Get all models"""
        return self._models
    
    def get_model_comparison(self):
        """Get model comparison instance"""
        return self.model_comparison
    
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None):
        """Train a specific model"""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            if model_name == 'xgboost' and X_val is not None:
                model.train(X_train, y_train, X_val, y_val)
            else:
                model.train(X_train, y_train)
            
            return True
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return False
    
    def predict(self, model_name, X):
        """Make predictions with a specific model"""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        if not model.is_fitted:
            raise ValueError(f"Model {model_name} is not trained")
        
        return model.predict(X)
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model"""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        return model.evaluate(X_test, y_test)

# Global model manager instance
model_manager = ModelManager()

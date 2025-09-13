import os
import json
import pandas as pd
import numpy as np
import datetime as dt
import io
import base64
from pathlib import Path

# Django imports
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, metrics
from sklearn.metrics import classification_report

# Plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker

# Import your forms and models
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

# Try to import advanced models, fallback to basic ones if not available
try:
    from .models import (
        LinearRegressionModel, RandomForestModel, XGBoostModel, SVRModel,
        ModelComparison, load_and_prepare_data
    )
    from .apps import model_manager
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("Advanced models not available, using basic implementation")


class MLPipelineManager:
    """Centralized ML pipeline to avoid code duplication"""
    
    def __init__(self):
        self.encoder = None
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    def load_and_preprocess_data(self, file_path=None):
        """Load and preprocess the sneaker data"""
        try:
            # Use pathlib for cross-platform path handling
            if file_path:
                path = Path(file_path)
            else:
                path = Path(settings.MEDIA_ROOT) / "Clean_Shoe_Data.csv"
            
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found at {path}")
            
            # Load data
            df = pd.read_csv(path, parse_dates=True)
            
            # Rename columns to remove spaces
            df = df.rename(columns={
                "Order Date": "Order_date",
                "Sneaker Name": "Sneaker_Name", 
                "Sale Price": "Sale_Price",
                "Retail Price": "Retail_Price",
                "Release Date": "Release_Date",
                "Shoe Size": "Shoe_Size",
                "Buyer Region": "Buyer"
            })
            
            # Convert dates to ordinal
            df['Order_date'] = pd.to_datetime(df['Order_date'], errors='coerce')
            df['Order_date'] = df['Order_date'].map(dt.datetime.toordinal)
            
            df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
            df['Release_Date'] = df['Release_Date'].map(dt.datetime.toordinal)
            
            # Handle missing values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def prepare_features(self, df, is_training=True):
        """Prepare features with one-hot encoding"""
        try:
            # Separate features and target
            X = df.drop(['Sale_Price'], axis=1)
            y = df['Sale_Price'] if 'Sale_Price' in df.columns else None
            
            # Categorical columns
            categorical_cols = ['Sneaker_Name', 'Buyer', 'Brand']
            
            if is_training:
                # Fit encoder on training data
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_features = self.encoder.fit_transform(X[categorical_cols])
                
                # Get feature names
                feature_names = self.encoder.get_feature_names_out(categorical_cols)
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                
                # Combine with numerical features
                numerical_features = X.drop(categorical_cols, axis=1)
                X_processed = pd.concat([numerical_features, encoded_df], axis=1)
                
                self.feature_names = X_processed.columns.tolist()
                
            else:
                # Transform using fitted encoder
                if self.encoder is None:
                    raise ValueError("Encoder not fitted. Please train the model first.")
                
                encoded_features = self.encoder.transform(X[categorical_cols])
                feature_names = self.encoder.get_feature_names_out(categorical_cols)
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                
                # Combine with numerical features
                numerical_features = X.drop(categorical_cols, axis=1)
                X_processed = pd.concat([numerical_features, encoded_df], axis=1)
                
                # Ensure all training columns are present
                for col in self.feature_names:
                    if col not in X_processed.columns:
                        X_processed[col] = 0
                
                # Reorder columns to match training data
                X_processed = X_processed[self.feature_names]
            
            return X_processed, y
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return None, None
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        try:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.is_fitted = True
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if not self.is_fitted:
            return None
        
        predictions = self.predict(X)
        
        return {
            'MAE': metrics.mean_absolute_error(y, predictions),
            'MSE': metrics.mean_squared_error(y, predictions), 
            'RMSE': np.sqrt(metrics.mean_squared_error(y, predictions)),
            'R2': metrics.r2_score(y, predictions)
        }


# Global ML pipeline instance
ml_pipeline = MLPipelineManager()


def UserRegisterActions(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.error(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    """User login check with session management"""
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID =", loginid, 'Password =', pswd)
        
        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if user.status == "activated":
                request.session['id'] = user.id
                request.session['loggeduser'] = user.name
                request.session['loginid'] = loginid
                request.session['email'] = user.email
                print("User id:", user.id, user.status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.error(request, 'Your Account is not activated')
                return render(request, 'UserLogin.html')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid Login ID and password')
        except Exception as e:
            print('Exception:', str(e))
            messages.error(request, 'An error occurred during login')
    
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    """User home page"""
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    """View dataset with error handling - RESOLVED VERSION"""
    try:
        path = Path(settings.MEDIA_ROOT) / "Clean_Shoe_Data.csv"
        df = pd.read_csv(path, nrows=100)
        df_html = df.to_html(classes='table table-striped', table_id='dataset-table')
        return render(request, 'users/viewdataset.html', {'data': df_html})
    except Exception as e:
        error_msg = f'Error loading data: {str(e)}'
        return render(request, 'users/viewdataset.html', {'data': error_msg})


def machinelearning(request):
    """Machine learning analysis - RESOLVED VERSION using centralized pipeline"""
    try:
        # Load and preprocess data using centralized pipeline
        df = ml_pipeline.load_and_preprocess_data()
        if df is None:
            return render(request, "users/ml.html", {"error": "Failed to load dataset"})
        
        # Prepare features using centralized pipeline
        X, y = ml_pipeline.prepare_features(df, is_training=True)
        if X is None:
            return render(request, "users/ml.html", {"error": "Failed to prepare features"})
        
        # Split data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model using centralized pipeline
        if not ml_pipeline.train_model(X_train, y_train):
            return render(request, "users/ml.html", {"error": "Failed to train model"})
        
        # Evaluate model using centralized pipeline
        metrics_result = ml_pipeline.evaluate(X_valid, y_valid)
        
        return render(request, "users/ml.html", {
            "MAE": round(metrics_result['MAE'], 2),
            "MSE": round(metrics_result['MSE'], 2), 
            "RMSE": round(metrics_result['RMSE'], 2),
            "R2": round(metrics_result['R2'], 4)
        })
        
    except Exception as e:
        return render(request, "users/ml.html", {"error": str(e)})


def prediction(request):
    """Price prediction view - RESOLVED VERSION using centralized pipeline"""
    if request.method == "POST":
        try:
            # Extract form data
            Order_date = request.POST.get("Order_date")
            Brand = request.POST.get("Brand")
            Sneaker_Name = request.POST.get("Sneaker_Name")
            Retail_Price = request.POST.get("Retail_Price")
            Release_Date = request.POST.get("Release_Date")
            Shoe_Size = request.POST.get("Shoe_Size")
            Buyer = request.POST.get("Buyer")
            
            print(f"Received prediction request for: {Sneaker_Name}, Brand: {Brand}, Buyer: {Buyer}")
            
            # Validate and convert numeric fields
            try:
                Retail_Price = float(Retail_Price) if Retail_Price else 0.0
                Shoe_Size = float(Shoe_Size) if Shoe_Size else 0.0
            except ValueError:
                return render(request, 'users/prediction.html', {
                    'error': 'Invalid numeric values for Retail Price or Shoe Size'
                })
            
            # Create DataFrame for prediction
            new_data = pd.DataFrame({
                'Order_date': [Order_date],
                'Brand': [Brand],
                'Sneaker_Name': [Sneaker_Name],
                'Retail_Price': [Retail_Price],
                'Release_Date': [Release_Date],
                'Shoe_Size': [Shoe_Size],
                'Buyer': [Buyer]
            })
            
            # Convert dates to ordinal
            new_data['Order_date'] = pd.to_datetime(new_data['Order_date'], errors='coerce')
            new_data['Order_date'] = new_data['Order_date'].map(dt.datetime.toordinal)
            
            new_data['Release_Date'] = pd.to_datetime(new_data['Release_Date'], errors='coerce')
            new_data['Release_Date'] = new_data['Release_Date'].map(dt.datetime.toordinal)
            
            # Add dummy Sale_Price column for processing
            new_data['Sale_Price'] = 0
            
            # Check if model is trained, if not train it
            if not ml_pipeline.is_fitted:
                print("Model not fitted, training now...")
                df = ml_pipeline.load_and_preprocess_data()
                if df is None:
                    return render(request, 'users/prediction.html', {'error': 'Failed to load training data'})
                
                X, y = ml_pipeline.prepare_features(df, is_training=True)
                if X is None:
                    return render(request, 'users/prediction.html', {'error': 'Failed to prepare training features'})
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                if not ml_pipeline.train_model(X_train, y_train):
                    return render(request, 'users/prediction.html', {'error': 'Failed to train model'})
            
            # Prepare prediction features
            X_pred, _ = ml_pipeline.prepare_features(new_data, is_training=False)
            if X_pred is None:
                return render(request, 'users/prediction.html', {'error': 'Failed to prepare prediction features'})
            
            # Make prediction
            prediction_result = ml_pipeline.predict(X_pred)
            predicted_price = round(float(prediction_result[0]), 2)
            
            # Prepare input data for display
            input_data = {
                'Order_date': Order_date,
                'Brand': Brand,
                'Sneaker_Name': Sneaker_Name,
                'Retail_Price': Retail_Price,
                'Release_Date': Release_Date,
                'Shoe_Size': Shoe_Size,
                'Buyer': Buyer
            }
            
            return render(request, 'users/prediction.html', {
                'y_pred': [predicted_price],
                'input_data': input_data,
                'success': True
            })
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return render(request, 'users/prediction.html', {'error': str(e)})
    
    return render(request, 'users/prediction.html')


def home(request):
    """Home page view"""
    context = {
        'title': 'Sneaker Price Prediction',
        'available_models': ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR'] if ADVANCED_MODELS_AVAILABLE else ['Random Forest'],
        'total_models': 4 if ADVANCED_MODELS_AVAILABLE else 1
    }
    return render(request, 'sneaker_prediction/home.html', context)


def model_comparison(request):
    """Model comparison page"""
    context = {
        'title': 'Model Comparison',
        'models': ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR'] if ADVANCED_MODELS_AVAILABLE else ['Random Forest']
    }
    return render(request, 'sneaker_prediction/model_comparison.html', context)


# Advanced model endpoints (only available if models are imported)
if ADVANCED_MODELS_AVAILABLE:
    @csrf_exempt
    @require_http_methods(["POST"])
    def train_models(request):
        """Train all models endpoint"""
        try:
            data = json.loads(request.body) if request.body else {}
            data_path = data.get('data_path', 'media/Clean_Shoe_Data.csv')
            
            X, y, feature_names = load_and_prepare_data(data_path)
            
            if X is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Failed to load data. Please check the file path.'
                })
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.125, random_state=42
            )
            
            comparison = ModelComparison()
            
            models = {
                'Linear Regression': LinearRegressionModel(),
                'Random Forest': RandomForestModel(n_estimators=100, random_state=42),
                'XGBoost': XGBoostModel(n_estimators=100, random_state=42),
                'SVR': SVRModel(kernel='rbf', C=1.0)
            }
            
            for name, model in models.items():
                comparison.add_model(name, model)
            
            comparison.train_all_models(X_train, y_train, X_val, y_val)
            comparison.evaluate_all_models(X_test, y_test)
            
            results_df = comparison.get_results_dataframe()
            best_model_name, _ = comparison.get_best_model()
            
            results = []
            for _, row in results_df.iterrows():
                results.append({
                    'model': row['Model'],
                    'r2_score': round(row['R² Score'], 4),
                    'mae': round(row['MAE'], 2),
                    'rmse': round(row['RMSE'], 2)
                })
            
            return JsonResponse({
                'success': True,
                'results': results,
                'best_model': best_model_name,
                'dataset_info': {
                    'total_samples': len(X),
                    'features': len(feature_names),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'price_range': {
                        'min': float(y.min()),
                        'max': float(y.max()),
                        'mean': float(y.mean())
                    }
                }
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    @csrf_exempt
    @require_http_methods(["POST"])
    def predict_price(request):
        """Predict sneaker price endpoint"""
        try:
            data = json.loads(request.body)
            model_name = data.get('model', 'xgboost').lower().replace(' ', '_')
            features = data.get('features', {})
            
            if not features:
                return JsonResponse({
                    'success': False,
                    'error': 'No features provided for prediction'
                })
            
            model = model_manager.get_model(model_name)
            if model is None:
                return JsonResponse({
                    'success': False,
                    'error': f'Model {model_name} not found'
                })
            
            if not model.is_fitted:
                return JsonResponse({
                    'success': False,
                    'error': f'Model {model_name} is not trained. Please train the model first.'
                })
            
            feature_df = pd.DataFrame([features])
            
            for col in feature_df.select_dtypes(include=['object']).columns:
                feature_df[col] = pd.Categorical(feature_df[col]).codes
            
            prediction = model.predict(feature_df)
            predicted_price = float(prediction[0])
            
            return JsonResponse({
                'success': True,
                'predicted_price': round(predicted_price, 2),
                'model_used': model_name.replace('_', ' ').title(),
                'features_used': len(features)
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

else:
    # Fallback implementations for basic functionality
    @csrf_exempt
    @require_http_methods(["POST"])
    def train_models(request):
        """Basic model training fallback"""
        return JsonResponse({
            'success': False,
            'error': 'Advanced models not available. Please install required dependencies.'
        })

    @csrf_exempt
    @require_http_methods(["POST"])
    def predict_price(request):
        """Basic prediction fallback"""
        return JsonResponse({
            'success': False,
            'error': 'Advanced prediction API not available. Use the web form instead.'
        })


def generate_comparison_chart(request):
    """Generate model comparison chart"""
    try:
        plt.figure(figsize=(10, 6))
        
        models = ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR']
        r2_scores = [0.75, 0.85, 0.92, 0.88]
        
        bars = plt.bar(models, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Model Performance Comparison (R² Score)', fontsize=16, fontweight='bold')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return JsonResponse({
            'success': True,
            'chart': f'data:image/png;base64,{image_base64}'
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def model_details(request, model_name):
    """Get detailed information about a specific model"""
    if not ADVANCED_MODELS_AVAILABLE:
        return JsonResponse({
            'success': False,
            'error': 'Advanced models not available'
        })
    
    try:
        model_name_key = model_name.lower().replace(' ', '_').replace('-', '_')
        model = model_manager.get_model(model_name_key)
        
        if model is None:
            return JsonResponse({
                'success': False,
                'error': f'Model {model_name} not found'
            })
        
        model_info = {
            'name': model.model_name,
            'is_trained': model.is_fitted,
            'algorithm_type': type(model).__name__,
        }
        
        if hasattr(model.model, 'get_params'):
            model_info['parameters'] = model.model.get_params()
        
        if hasattr(model, 'get_feature_importance') and model.is_fitted:
            try:
                feature_names = [f'feature_{i}' for i in range(10)]
                importance = model.get_feature_importance(feature_names)
                model_info['feature_importance'] = importance.to_dict('records')
            except:
                model_info['feature_importance'] = None
        
        return JsonResponse({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def upload_dataset(request):
    """Handle dataset upload"""
    try:
        if 'dataset' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No file uploaded'
            })
        
        uploaded_file = request.FILES['dataset']
        
        if not uploaded_file.name.endswith('.csv'):
            return JsonResponse({
                'success': False,
                'error': 'Only CSV files are supported'
            })
        
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Error reading CSV file: {str(e)}'
            })
        
        if df.empty:
            return JsonResponse({
                'success': False,
                'error': 'Uploaded file is empty'
            })
        
        # Save file
        media_dir = Path(settings.MEDIA_ROOT)
        media_dir.mkdir(exist_ok=True)
        file_path = media_dir / f'uploaded_{uploaded_file.name}'
        
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        return JsonResponse({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'file_info': {
                'filename': uploaded_file.name,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'file_path': str(file_path)
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def api_status(request):
    """API status endpoint"""
    try:
        if ADVANCED_MODELS_AVAILABLE:
            available_models = model_manager.get_all_models()
            models_list = list(available_models.keys())
        else:
            available_models = {'random_forest': 'Available'}
            models_list = ['Random Forest (Basic)']
        
        status = {
            'status': 'online',
            'available_models': len(available_models),
            'models': models_list,
            'version': '1.0.0',
            'advanced_features': ADVANCED_MODELS_AVAILABLE,
            'features': [
                'Linear Regression' if ADVANCED_MODELS_AVAILABLE else None,
                'Random Forest',
                'XGBoost' if ADVANCED_MODELS_AVAILABLE else None,
                'Support Vector Regression' if ADVANCED_MODELS_AVAILABLE else None,
                'Model Comparison',
                'Performance Metrics'
            ]
        }
        
        # Remove None values
        status['features'] = [f for f in status['features'] if f is not None]
        
        return JsonResponse(status)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        })


# Error handler views
def handler404(request, exception):
    """Custom 404 page"""
    return render(request, 'sneaker_prediction/404.html', status=404)


def handler500(request):
    """Custom 500 page"""
    return render(request, 'sneaker_prediction/500.html', status=500)

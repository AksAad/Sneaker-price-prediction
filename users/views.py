# SNEAKER-PRICE-PR/views.py
import os
import json
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

# Django imports
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Plotting imports (for potential future use, though chart.js is used now)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fixed imports - Use absolute imports from the users app
try:
    from users.forms import UserRegistrationForm
    from users.models import UserRegistrationModel
    from users.models import (
        LinearRegressionModel, RandomForestModel, XGBoostModel, SVRModel,
        ModelComparison, load_and_prepare_data
    )
    from users.apps import model_manager
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    # Fallback for when running as standalone script or if advanced models are not available
    from forms import UserRegistrationForm
    from models import UserRegistrationModel
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
            if file_path:
                path = Path(file_path)
            else:
                if hasattr(settings, 'MEDIA_ROOT'):
                    path = Path(settings.MEDIA_ROOT) / "Clean_Shoe_Data.csv"
                else:
                    path = Path("media") / "Clean_Shoe_Data.csv"
                    
                alt_paths = [
                    Path("Clean_Shoe_Data.csv"),
                    Path("data") / "Clean_Shoe_Data.csv",
                    Path("../data") / "Clean_Shoe_Data.csv"
                ]
                
                if not path.exists():
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            path = alt_path
                            break
            
            if not path.exists():
                print(f"Dataset not found at {path}, creating dummy data for testing")
                return self._create_dummy_data()
            
            df = pd.read_csv(path, parse_dates=True)
            df = df.rename(columns={
                "Order Date": "Order_date",
                "Sneaker Name": "Sneaker_Name",
                "Sale Price": "Sale_Price",
                "Retail Price": "Retail_Price",
                "Release Date": "Release_Date",
                "Shoe Size": "Shoe_Size",
                "Buyer Region": "Buyer"
            })
            
            def safe_date_convert(date_series):
                try:
                    converted = pd.to_datetime(date_series, errors='coerce')
                    converted = converted.fillna(pd.Timestamp('2023-01-01'))
                    return converted.map(dt.datetime.toordinal)
                except Exception as e:
                    print(f"Date conversion error: {e}")
                    return pd.Series([dt.datetime(2023, 1, 1).toordinal()] * len(date_series))
            
            df['Order_date'] = safe_date_convert(df['Order_date'])
            df['Release_Date'] = safe_date_convert(df['Release_Date'])
            
            df = df.dropna(subset=['Sale_Price'])
            df = df.fillna({
                'Brand': 'Unknown',
                'Sneaker_Name': 'Unknown',
                'Buyer': 'Unknown',
                'Retail_Price': df['Retail_Price'].median() if 'Retail_Price' in df else 100,
                'Shoe_Size': df['Shoe_Size'].median() if 'Shoe_Size' in df else 10
            })
            
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for testing when real data is not available"""
        np.random.seed(42)
        n_samples = 1000
        brands = ['Nike', 'Adidas', 'Jordan', 'Puma', 'New Balance']
        sneaker_names = [f'Sneaker_{i}' for i in range(1, 21)]
        buyers = ['US', 'Europe', 'Asia', 'Canada', 'Australia']
        
        dummy_data = pd.DataFrame({
            'Order_date': [dt.datetime(2023, 1, 1).toordinal() + np.random.randint(0, 365) for _ in range(n_samples)],
            'Brand': np.random.choice(brands, n_samples),
            'Sneaker_Name': np.random.choice(sneaker_names, n_samples),
            'Retail_Price': np.random.normal(150, 50, n_samples),
            'Release_Date': [dt.datetime(2022, 1, 1).toordinal() + np.random.randint(0, 365) for _ in range(n_samples)],
            'Shoe_Size': np.random.normal(10, 2, n_samples),
            'Buyer': np.random.choice(buyers, n_samples),
            'Sale_Price': np.random.normal(200, 75, n_samples)
        })
        
        dummy_data['Retail_Price'] = np.abs(dummy_data['Retail_Price'])
        dummy_data['Sale_Price'] = np.abs(dummy_data['Sale_Price'])
        dummy_data['Shoe_Size'] = np.abs(dummy_data['Shoe_Size'])
        print("Created dummy dataset with 1000 samples for testing")
        return dummy_data
    
    def prepare_features(self, df, is_training=True):
        """Prepare features with one-hot encoding"""
        try:
            required_cols = ['Sale_Price', 'Sneaker_Name', 'Buyer', 'Brand']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols and 'Sale_Price' in missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                return None, None
            
            feature_cols = [col for col in df.columns if col != 'Sale_Price']
            X = df[feature_cols].copy()
            y = df['Sale_Price'] if 'Sale_Price' in df.columns else None
            
            available_categorical_cols = [col for col in ['Sneaker_Name', 'Buyer', 'Brand'] if col in X.columns]
            if not available_categorical_cols:
                print("Warning: No categorical columns found")
                return X, y
            
            if is_training:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_features = self.encoder.fit_transform(X[available_categorical_cols])
                feature_names = self.encoder.get_feature_names_out(available_categorical_cols)
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                
                numerical_cols = [col for col in X.columns if col not in available_categorical_cols]
                if numerical_cols:
                    numerical_features = X[numerical_cols]
                    X_processed = pd.concat([numerical_features, encoded_df], axis=1)
                else:
                    X_processed = encoded_df
                
                self.feature_names = X_processed.columns.tolist()
            else:
                if self.encoder is None:
                    raise ValueError("Encoder not fitted. Please train the model first.")
                
                encoded_features = self.encoder.transform(X[available_categorical_cols])
                feature_names = self.encoder.get_feature_names_out(available_categorical_cols)
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                
                numerical_cols = [col for col in X.columns if col not in available_categorical_cols]
                if numerical_cols:
                    numerical_features = X[numerical_cols]
                    X_processed = pd.concat([numerical_features, encoded_df], axis=1)
                else:
                    X_processed = encoded_df
                
                for col in self.feature_names:
                    if col not in X_processed.columns:
                        X_processed[col] = 0
                
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
            print("Model trained successfully")
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
            return render(request, 'users/UserRegistrations.html', {'form': form})
        else:
            messages.error(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'users/UserRegistrations.html', {'form': form})

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
    """View dataset with error handling"""
    try:
        data_loaded = False
        possible_paths = [
            Path(settings.MEDIA_ROOT) / "Clean_Shoe_Data.csv" if hasattr(settings, 'MEDIA_ROOT') else None,
            Path("media") / "Clean_Shoe_Data.csv",
            Path("Clean_Shoe_Data.csv"),
            Path("data") / "Clean_Shoe_Data.csv"
        ]
        
        df = None
        for path in possible_paths:
            if path and path.exists():
                try:
                    df = pd.read_csv(path, nrows=100)
                    data_loaded = True
                    break
                except Exception as e:
                    continue
        
        if not data_loaded:
            df = ml_pipeline._create_dummy_data().head(100)
            messages.info(request, 'Showing sample data as original dataset was not found')
        
        df_html = df.to_html(classes='table table-striped', table_id='dataset-table')
        return render(request, 'users/viewdataset.html', {'data': df_html})
    except Exception as e:
        error_msg = f'Error loading data: {str(e)}'
        return render(request, 'users/viewdataset.html', {'data': error_msg})

def machinelearning(request):
    """Machine learning analysis"""
    try:
        df = ml_pipeline.load_and_preprocess_data()
        if df is None:
            return render(request, "users/ml.html", {"error": "Failed to load dataset"})
        
        X, y = ml_pipeline.prepare_features(df, is_training=True)
        if X is None:
            return render(request, "users/ml.html", {"error": "Failed to prepare features"})
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if not ml_pipeline.train_model(X_train, y_train):
            return render(request, "users/ml.html", {"error": "Failed to train model"})
        
        metrics_result = ml_pipeline.evaluate(X_valid, y_valid)
        
        return render(request, "users/ml.html", {
            "MAE": round(metrics_result['MAE'], 2),
            "MSE": round(metrics_result['MSE'], 2),
            "RMSE": round(metrics_result['RMSE'], 2),
            "R2": round(metrics_result['R2'], 4),
            "data_info": {
                "total_samples": len(df),
                "features": len(X.columns),
                "train_samples": len(X_train),
                "test_samples": len(X_valid)
            }
        })
    except Exception as e:
        return render(request, "users/ml.html", {"error": str(e)})

def prediction(request):
    """Price prediction view"""
    if request.method == "POST":
        try:
            form_data = {}
            required_fields = ["Order_date", "Brand", "Sneaker_Name", "Retail_Price", "Release_Date", "Shoe_Size", "Buyer"]
            for field in required_fields:
                value = request.POST.get(field)
                if not value:
                    return render(request, 'users/prediction.html', {'error': f'Missing required field: {field}'})
                form_data[field] = value
            
            print(f"Received prediction request: {form_data}")
            
            try:
                retail_price = float(form_data['Retail_Price']) if form_data['Retail_Price'] else 0.0
                shoe_size = float(form_data['Shoe_Size']) if form_data['Shoe_Size'] else 0.0
                if retail_price < 0 or shoe_size < 0:
                    return render(request, 'users/prediction.html', {'error': 'Retail Price and Shoe Size must be positive numbers'})
            except ValueError:
                return render(request, 'users/prediction.html', {'error': 'Invalid numeric values for Retail Price or Shoe Size'})
            
            new_data = pd.DataFrame({
                'Order_date': [form_data['Order_date']],
                'Brand': [form_data['Brand']],
                'Sneaker_Name': [form_data['Sneaker_Name']],
                'Retail_Price': [retail_price],
                'Release_Date': [form_data['Release_Date']],
                'Shoe_Size': [shoe_size],
                'Buyer': [form_data['Buyer']]
            })
            
            def safe_single_date_convert(date_str):
                try:
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                    if pd.isna(parsed_date):
                        parsed_date = pd.Timestamp('2023-01-01')
                    return parsed_date.toordinal()
                except:
                    return dt.datetime(2023, 1, 1).toordinal()
            
            new_data['Order_date'] = new_data['Order_date'].apply(safe_single_date_convert)
            new_data['Release_Date'] = new_data['Release_Date'].apply(safe_single_date_convert)
            new_data['Sale_Price'] = 0
            
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
            
            X_pred, _ = ml_pipeline.prepare_features(new_data, is_training=False)
            if X_pred is None:
                return render(request, 'users/prediction.html', {'error': 'Failed to prepare prediction features'})
            
            prediction_result = ml_pipeline.predict(X_pred)
            predicted_price = round(float(prediction_result[0]), 2)
            if predicted_price < 0:
                predicted_price = abs(predicted_price)
            
            return render(request, 'users/prediction.html', {
                'y_pred': [predicted_price],
                'input_data': form_data,
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
    """Model comparison page with performance plot"""
    if not ADVANCED_MODELS_AVAILABLE:
        return render(request, 'sneaker_prediction/model_comparison.html', {
            'title': 'Model Comparison',
            'error': 'Advanced models not available. Please install required dependencies.',
            'models': ['Random Forest']
        })

    try:
        data_path = Path(settings.MEDIA_ROOT) / "Clean_Shoe_Data.csv" if hasattr(settings, 'MEDIA_ROOT') else Path("media/Clean_Shoe_Data.csv")
        X, y, feature_names = load_and_prepare_data(data_path)

        if X is None:
            return render(request, 'sneaker_prediction/model_comparison.html', {
                'title': 'Model Comparison',
                'error': 'Failed to load dataset.',
                'models': ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR']
            })

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

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

        chart_data = {
            'labels': results_df['Model'].tolist(),
            'r2_scores': results_df['R² Score'].tolist(),
            'mae': results_df['MAE'].tolist(),
            'rmse': results_df['RMSE'].tolist()
        }

        return render(request, 'sneaker_prediction/model_comparison.html', {
            'title': 'Model Comparison',
            'models': list(models.keys()),
            'results': results_df.to_dict('records'),
            'best_model': best_model_name,
            'chart_data': chart_data
        })

    except Exception as e:
        return render(request, 'sneaker_prediction/model_comparison.html', {
            'title': 'Model Comparison',
            'error': f'Error during model comparison: {str(e)}',
            'models': ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR']
        })

@csrf_exempt
@require_http_methods(["POST"])
def train_models(request):
    """Train all models endpoint with comparison plot data"""
    if not ADVANCED_MODELS_AVAILABLE:
        return JsonResponse({
            'success': False,
            'error': 'Advanced models not available. Please install required dependencies.'
        })

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

        chart_data = {
            'labels': results_df['Model'].tolist(),
            'r2_scores': results_df['R² Score'].tolist(),
            'mae': results_df['MAE'].tolist(),
            'rmse': results_df['RMSE'].tolist()
        }

        return JsonResponse({
            'success': True,
            'results': results,
            'best_model': best_model_name,
            'chart_data': chart_data,
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

def handler404(request, exception):
    """Custom 404 page"""
    return render(request, 'sneaker_prediction/404.html', status=404)

def handler500(request):
    """Custom 500 page"""
    return render(request, 'sneaker_prediction/500.html', status=500)

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import our models and utilities
from .models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, SVRModel,
    ModelComparison, load_and_prepare_data
)
from .apps import model_manager

def home(request):
    """Home page view"""
    context = {
        'title': 'Sneaker Price Prediction',
        'available_models': ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR'],
        'total_models': 4
    }
    return render(request, 'sneaker_prediction/home.html', context)

def model_comparison(request):
    """Model comparison page"""
    context = {
        'title': 'Model Comparison',
        'models': ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR']
    }
    return render(request, 'sneaker_prediction/model_comparison.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def train_models(request):
    """Train all models endpoint"""
    try:
        # Get data path from request or use default
        data = json.loads(request.body) if request.body else {}
        data_path = data.get('data_path', 'media/Clean_Shoe_Data.csv')
        
        # Load and prepare data
        X, y, feature_names = load_and_prepare_data(data_path)
        
        if X is None:
            return JsonResponse({
                'success': False,
                'error': 'Failed to load data. Please check the file path.'
            })
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42  # 0.1 of total
        )
        
        # Initialize model comparison
        comparison = ModelComparison()
        
        # Add models
        models = {
            'Linear Regression': LinearRegressionModel(),
            'Random Forest': RandomForestModel(n_estimators=100, random_state=42),
            'XGBoost': XGBoostModel(n_estimators=100, random_state=42),
            'SVR': SVRModel(kernel='rbf', C=1.0)
        }
        
        for name, model in models.items():
            comparison.add_model(name, model)
        
        # Train models
        comparison.train_all_models(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        comparison.evaluate_all_models(X_test, y_test)
        
        # Get results
        results_df = comparison.get_results_dataframe()
        best_model_name, _ = comparison.get_best_model()
        
        # Prepare response data
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
        
        # Get model
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
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Handle categorical variables if needed
        for col in feature_df.select_dtypes(include=['object']).columns:
            feature_df[col] = pd.Categorical(feature_df[col]).codes
        
        # Make prediction
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

def generate_comparison_chart(request):
    """Generate model comparison chart"""
    try:
        # This would typically use cached results from a previous training
        # For now, return a placeholder response
        
        # Create a sample comparison chart
        plt.figure(figsize=(10, 6))
        
        # Sample data (in real implementation, this would come from actual results)
        models = ['Linear Regression', 'Random Forest', 'XGBoost', 'SVR']
        r2_scores = [0.75, 0.85, 0.92, 0.88]  # Example scores
        
        bars = plt.bar(models, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Model Performance Comparison (R² Score)', fontsize=16, fontweight='bold')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert plot to base64 string
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
    try:
        model_name_key = model_name.lower().replace(' ', '_').replace('-', '_')
        model = model_manager.get_model(model_name_key)
        
        if model is None:
            return JsonResponse({
                'success': False,
                'error': f'Model {model_name} not found'
            })
        
        # Get model information
        model_info = {
            'name': model.model_name,
            'is_trained': model.is_fitted,
            'algorithm_type': type(model).__name__,
        }
        
        # Add specific model parameters
        if hasattr(model.model, 'get_params'):
            model_info['parameters'] = model.model.get_params()
        
        # Add feature importance if available
        if hasattr(model, 'get_feature_importance') and model.is_fitted:
            try:
                # This would need actual feature names from training
                feature_names = [f'feature_{i}' for i in range(10)]  # Placeholder
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
        
        # Validate file type
        if not uploaded_file.name.endswith('.csv'):
            return JsonResponse({
                'success': False,
                'error': 'Only CSV files are supported'
            })
        
        # Read and validate the CSV
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Error reading CSV file: {str(e)}'
            })
        
        # Basic validation
        if df.empty:
            return JsonResponse({
                'success': False,
                'error': 'Uploaded file is empty'
            })
        
        # Save file (in production, use proper file handling)
        file_path = f'media/uploaded_{uploaded_file.name}'
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
                'file_path': file_path
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
        available_models = model_manager.get_all_models()
        
        status = {
            'status': 'online',
            'available_models': len(available_models),
            'models': list(available_models.keys()),
            'version': '1.0.0',
            'features': [
                'Linear Regression',
                'Random Forest', 
                'XGBoost',
                'Support Vector Regression',
                'Model Comparison',
                'Feature Importance Analysis',
                'Performance Metrics'
            ]
        }
        
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
   
        


    

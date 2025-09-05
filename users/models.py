from django.db import models

# Create your models here.
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'UserRegistrations'

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class BaseModel:
    """Base class for all models"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def train(self, X_train, y_train):
        raise NotImplementedError
    
    def predict(self, X_test):
        raise NotImplementedError
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']

class LinearRegressionModel(BaseModel):
    """Linear Regression Model"""
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.model_name = "Linear Regression"
    
    def train(self, X_train, y_train):
        """Train the Linear Regression model"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X_test)

class RandomForestModel(BaseModel):
    """Random Forest Model"""
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.model_name = "Random Forest"
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """Train the Random Forest model"""
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, 
                scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class XGBoostModel(BaseModel):
    """XGBoost Model"""
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        super().__init__()
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective='reg:squarederror'
        )
        self.model_name = "XGBoost"
    
    def train(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparameters=False):
        """Train the XGBoost model"""
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5,
                scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use early stopping if validation data is provided
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class SVRModel(BaseModel):
    """Support Vector Regression Model"""
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        super().__init__()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.scaler = StandardScaler()
        self.model_name = "Support Vector Regression"
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """Train the SVR model"""
        # SVR requires feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'kernel': ['rbf', 'poly', 'linear']
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5,
                scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

class ModelComparison:
    """Class for comparing multiple models"""
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model):
        """Add a model to comparison"""
        self.models[name] = model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models"""
        print("Training all models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                if name == "XGBoost" and X_val is not None:
                    model.train(X_train, y_train, X_val, y_val)
                else:
                    model.train(X_train, y_train)
                print(f"✓ {name} training completed")
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models and store results"""
        print("\nEvaluating all models...")
        
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    results = model.evaluate(X_test, y_test)
                    self.results[name] = results
                    print(f"✓ {name} evaluation completed")
                    print(f"  R² Score: {results['r2_score']:.4f}")
                    print(f"  MAE: {results['mae']:.4f}")
                    print(f"  RMSE: {results['rmse']:.4f}")
                except Exception as e:
                    print(f"✗ Error evaluating {name}: {str(e)}")
    
    def get_best_model(self):
        """Get the best performing model based on R² score"""
        if not self.results:
            return None, None
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['r2_score'])
        best_model = self.models[best_model_name]
        
        return best_model_name, best_model
    
    def get_results_dataframe(self):
        """Get results as a pandas DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        df_data = []
        for name, metrics in self.results.items():
            df_data.append({
                'Model': name,
                'R² Score': metrics['r2_score'],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse']
            })
        
        return pd.DataFrame(df_data).sort_values('R² Score', ascending=False)

def create_models():
    """Factory function to create all model instances"""
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel(),
        'XGBoost': XGBoostModel(),
        'SVR': SVRModel()
    }
    return models

def load_and_prepare_data(filepath):
    """Load and prepare data for modeling"""
    try:
        df = pd.read_csv(filepath)
        
        # Assuming the target column is named 'price' or similar
        # Adjust column names based on your actual dataset
        target_cols = ['price', 'Price', 'retail_price', 'sale_price']
        target_col = None
        
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("Price column not found in dataset")
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Handle categorical variables (basic encoding)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        return X, y, list(X.columns)
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

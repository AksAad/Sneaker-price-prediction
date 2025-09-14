# SNEAKER-PRICE-PR/users/views.py
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from django.conf import settings

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn import metrics

# Model classes for modularity
class BaseModel:
    def __init__(self):
        self.model = None
        self.is_fitted = False
    
    def train(self, X_train, y_train):
        raise NotImplementedError
    
    def predict(self, X_test):
        raise NotImplementedError
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return {
            'r2_score': metrics.r2_score(y_test, y_pred),
            'mae': metrics.mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        }

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return self.model.predict(X_test)

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return self.model.predict(X_test)

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return self.model.predict(X_test)

class SVRModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVR(kernel='rbf', C=1.0)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return self.model.predict(X_test)

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model):
        self.models[name] = model
    
    def train_all_models(self, X_train, y_train):
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.train(X_train, y_train)
                print(f"✓ {name} training completed")
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
    
    def evaluate_all_models(self, X_test, y_test):
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    self.results[name] = model.evaluate(X_test, y_test)
                    print(f"✓ {name} evaluation completed")
                except Exception as e:
                    print(f"✗ Error evaluating {name}: {str(e)}")

def UserRegisterActions(request):
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
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.error(request, 'Your Account Not activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
        messages.error(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'Clean_Shoe_Data.csv'
    try:
        df = pd.read_csv(path, nrows=100)
        df_html = df.to_html()
        return render(request, 'users/viewdataset.html', {'data': df_html})
    except Exception as e:
        return render(request, 'users/viewdataset.html', {'data': f'Error loading data: {str(e)}'})

def machinelearning(request):
    path = settings.MEDIA_ROOT + "//" + "Clean_Shoe_Data.csv"
    try:
        df = pd.read_csv(path, parse_dates=True)

    df = pd.read_csv(path, nrows=100)
    df = df.to_html()
    return render(request, 'users/viewdataset.html', {'data': df})

def machinelearning(request):
    # Reading in the data
    path = settings.MEDIA_ROOT + "\\" + "Clean_Shoe_Data.csv"

    shoe_data = pd.read_csv(path, parse_dates = True)
    df = shoe_data.copy()
    df
    # Checking for missing values in the dataset
    nulls = pd.concat([df.isnull().sum()], axis=1)
    nulls[nulls.sum(axis=1) > 0]
    
    # Renaming columns to get rid of spaces 
    df = df.rename(columns={
    "Order Date": "Order_date",
    "Sneaker Name": "Sneaker_Name",
    "Sale Price": "Sale_Price",
    "Retail Price": "Retail_Price",
    "Release Date": "Release_Date",
    "Shoe Size": "Shoe_Size",
    "Buyer Region": "Buyer"
    })
    df['Order_date'] = pd.to_datetime(df['Order_date'])
    df['Order_date']=df['Order_date'].map(dt.datetime.toordinal)

    df['Release_Date'] = pd.to_datetime(df['Release_Date'])
    df['Release_Date']=df['Release_Date'].map(dt.datetime.toordinal)
    # Starting the linear regression


    X = df.drop(['Sale_Price'], axis=1)
    y = df.Sale_Price
    #X = X.columns.astype(str)
    print(X.columns)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    object_cols = ['Sneaker_Name', 'Buyer', 'Brand']
    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Adding the column names after one hot encoding
    OH_cols_train.columns = OH_encoder.get_feature_names_out(object_cols)
    OH_cols_valid.columns = OH_encoder.get_feature_names_out(object_cols)

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    lm = RandomForestRegressor()
    lm.fit(OH_X_train,y_train)
    predictions = lm.predict(OH_X_valid)
    MAE = metrics.mean_absolute_error(y_valid, predictions)
    MSE =  metrics.mean_squared_error(y_valid, predictions)
    RMSE =  np.sqrt(metrics.mean_squared_error(y_valid, predictions))
   
    
    return render(request,"users/ml.html",{"MAE":MAE,"MSE":MSE,"RMSE":RMSE})



def prediction(request):
    if request.method == "POST":
        import pandas as pd
        from django.conf import settings

        Order_date = request.POST.get("Order_date")
        Brand = request.POST.get("Brand")
        Sneaker_Name = request.POST.get("Sneaker_Name")
        Retail_Price = request.POST.get("Retail_Price")
        Release_Date = request.POST.get("Release_Date")
        Shoe_Size = request.POST.get("Shoe_Size")
        Buyer = request.POST.get("Buyer")
        print(Buyer)

        path = settings.MEDIA_ROOT + "\\" + "Clean_Shoe_Data.csv"

        shoe_data = pd.read_csv(path, parse_dates = True)
        df = shoe_data.copy()
        df
        # Checking for missing values in the dataset
        nulls = pd.concat([df.isnull().sum()], axis=1)
        nulls[nulls.sum(axis=1) > 0]
        
        # Renaming columns to get rid of spaces 
main
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
            'Retail_Price': df['Retail_Price'].median(),
            'Shoe_Size': df['Shoe_Size'].median()
        })

        X = df.drop(['Sale_Price'], axis=1)
        y = df['Sale_Price']
        
        object_cols = ['Sneaker_Name', 'Buyer', 'Brand']
main
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Apply one-hot encoder to each column with categorical data
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
 main
        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
        
        OH_cols_train.index = X_train.index
        OH_cols_valid.index = X_valid.index
        
        OH_cols_train.columns = OH_encoder.get_feature_names_out(object_cols)
        OH_cols_valid.columns = OH_encoder.get_feature_names_out(object_cols)
        
        num_X_train = X_train.drop(object_cols, axis=1)
        num_X_valid = X_valid.drop(object_cols, axis=1)
        
        OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

        # Initialize and train models
        models = {
            'Linear Regression': LinearRegressionModel(),
            'Random Forest': RandomForestModel(),
            'XGBoost': XGBoostModel(),
            'SVR': SVRModel()
        }
        
        comparison = ModelComparison()
        for name, model in models.items():
            comparison.add_model(name, model)
        
        comparison.train_all_models(OH_X_train, y_train)
        comparison.evaluate_all_models(OH_X_valid, y_valid)
        
        # Prepare chart data
        chart_data = {
            'labels': list(comparison.results.keys()),
            'r2_scores': [comparison.results[model]['r2_score'] for model in comparison.results],
            'mae': [comparison.results[model]['mae'] for model in comparison.results],
            'rmse': [comparison.results[model]['rmse'] for model in comparison.results]
        }
        
        return render(request, "users/ml.html", {
            'results': comparison.results,
            'chart_data': chart_data,
            'best_model': max(comparison.results, key=lambda x: comparison.results[x]['r2_score'])
        })
    except Exception as e:
        return render(request, "users/ml.html", {'error': str(e)})

def prediction(request):
    if request.method == "POST":
        try:
            path = settings.MEDIA_ROOT + "//" + "Clean_Shoe_Data.csv"
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
            
            def safe_date_convert(date_str):
                try:
                    return pd.to_datetime(date_str, errors='coerce').toordinal()
                except:
                    return dt.datetime(2023, 1, 1).toordinal()
            
            Order_date = safe_date_convert(request.POST.get("Order_date"))
            Release_Date = safe_date_convert(request.POST.get("Release_Date"))
            Retail_Price = float(request.POST.get("Retail_Price") or 0)
            Shoe_Size = float(request.POST.get("Shoe_Size") or 0)
            Brand = request.POST.get("Brand", "Unknown")
            Sneaker_Name = request.POST.get("Sneaker_Name", "Unknown")
            Buyer = request.POST.get("Buyer", "Unknown")

            X = df.drop(['Sale_Price'], axis=1)
            y = df['Sale_Price']
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            
            object_cols = ['Sneaker_Name', 'Buyer', 'Brand']
            OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
            OH_cols_train.index = X_train.index
            OH_cols_train.columns = OH_encoder.get_feature_names_out(object_cols)
            
            num_X_train = X_train.drop(object_cols, axis=1)
            OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
            
            lm = RandomForestRegressor(n_estimators=100, random_state=42)
            lm.fit(OH_X_train, y_train)
            
            new_data = pd.DataFrame({
                'Order_date': [Order_date],
                'Brand': [Brand],
                'Sneaker_Name': [Sneaker_Name],
                'Retail_Price': [Retail_Price],
                'Release_Date': [Release_Date],
                'Shoe_Size': [Shoe_Size],
                'Buyer': [Buyer]
            })
            
            new_data_object_cols = new_data[object_cols]
            OH_cols_new = pd.DataFrame(OH_encoder.transform(new_data_object_cols))
            OH_cols_new.index = new_data.index
            OH_cols_new.columns = OH_encoder.get_feature_names_out(object_cols)
            
            num_X_new = new_data.drop(object_cols, axis=1)
            OH_X_new = pd.concat([num_X_new, OH_cols_new], axis=1)
            
            # Align columns with training data
            for col in OH_X_train.columns:
                if col not in OH_X_new.columns:
                    OH_X_new[col] = 0
            OH_X_new = OH_X_new[OH_X_train.columns]
            
            y_pred = lm.predict(OH_X_new)
            predicted_price = max(round(float(y_pred[0]), 2), 0)  # Ensure non-negative price
            
            return render(request, 'users/prediction.html', {
                'y_pred': [predicted_price],
                'input_data': {
                    'Order_date': request.POST.get("Order_date"),
                    'Brand': Brand,
                    'Sneaker_Name': Sneaker_Name,
                    'Retail_Price': Retail_Price,
                    'Release_Date': request.POST.get("Release_Date"),
                    'Shoe_Size': Shoe_Size,
                    'Buyer': Buyer
                }
            })
        except Exception as e:
            return render(request, 'users/prediction.html', {'error': str(e)})
    
main
    return render(request, 'users/prediction.html')


 main

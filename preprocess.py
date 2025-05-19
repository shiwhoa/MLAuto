import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def load_and_preprocess_data(file_path, target_column='motor_power', test_size=0.2):
    """
    Load and preprocess the data for model training
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    target_column : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler, feature_names : tuple
        Preprocessed and split data with feature names
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Get all columns except target
    feature_names = [col for col in df.columns if col != target_column]
    
    X = df[feature_names]
    y = df[target_column]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

def get_feature_importance_scores(X, y, feature_names):
    """
    Get feature importance scores using multiple methods
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary containing feature importance scores from different methods
    """
    # 1. F-regression scores
    f_scores, _ = f_regression(X, y)
    f_importance = dict(zip(feature_names, f_scores))
    
    # 2. Mutual Information scores
    mi_scores = mutual_info_regression(X, y)
    mi_importance = dict(zip(feature_names, mi_scores))
    
    # 3. Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = dict(zip(feature_names, rf.feature_importances_))
    
    # 4. XGBoost importance
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X, y)
    xgb_importance = dict(zip(feature_names, xgb_model.feature_importances_))
    
    return {
        'F-Regression': f_importance,
        'Mutual Information': mi_importance,
        'Random Forest': rf_importance,
        'XGBoost': xgb_importance
    }

def select_top_features(X, y, feature_names, method='f_regression', k=10):
    """
    Select top k features using specified method
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        List of feature names
    method : str
        Feature selection method ('f_regression' or 'mutual_info')
    k : int
        Number of top features to select
        
    Returns:
    --------
    tuple
        Selected features and their names
    """
    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=k)
    else:
        selector = SelectKBest(mutual_info_regression, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    return X_selected, selected_features 
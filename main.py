import pandas as pd
import numpy as np
from preprocess import (
    load_and_preprocess_data,
    get_feature_importance_scores,
    select_top_features
)
from xgboost_model import XGBoostRegressor
from linear_regression_model import LinearRegressor
from nonlinear_regression_model import NonLinearRegressor

def print_feature_importance(importance_dict, method_name):
    """Print feature importance scores for a given method"""
    print(f"\n{method_name} Feature Importance:")
    print("-" * 50)
    # Sort features by importance score
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feature, score in sorted_features:
        print(f"{feature}: {score:.4f}")

def main():
    # Load and preprocess data
    file_path = 'your_data.csv'  # Replace with your data file path
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(file_path)
    
    # Get feature importance scores using multiple methods
    print("\nAnalyzing Feature Importance...")
    importance_scores = get_feature_importance_scores(X_train, y_train, feature_names)
    
    # Print feature importance from all methods
    for method_name, scores in importance_scores.items():
        print_feature_importance(scores, method_name)
    
    # Select top 10 features using F-regression
    print("\nSelecting top 10 features using F-regression...")
    X_train_selected, selected_features = select_top_features(
        X_train, y_train, feature_names, method='f_regression', k=10
    )
    print("\nTop 10 selected features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # Train models with selected features
    print("\nTraining models with selected features...")
    
    # Train and evaluate XGBoost model
    print("\nTraining XGBoost Model...")
    xgb_model = XGBoostRegressor()
    xgb_model.train(X_train_selected, y_train)
    xgb_metrics = xgb_model.evaluate(X_test[:, [feature_names.index(f) for f in selected_features]], y_test)
    print("XGBoost Results:")
    print(f"RMSE: {xgb_metrics['RMSE']:.2f}")
    print(f"R2 Score: {xgb_metrics['R2 Score']:.2f}")
    
    # Train and evaluate Linear Regression model
    print("\nTraining Linear Regression Model...")
    lr_model = LinearRegressor()
    lr_model.train(X_train_selected, y_train)
    lr_metrics = lr_model.evaluate(X_test[:, [feature_names.index(f) for f in selected_features]], y_test)
    print("Linear Regression Results:")
    print(f"RMSE: {lr_metrics['RMSE']:.2f}")
    print(f"R2 Score: {lr_metrics['R2 Score']:.2f}")
    
    # Train and evaluate Non-linear Regression model
    print("\nTraining Non-linear Regression Model...")
    nl_model = NonLinearRegressor()
    nl_model.train(X_train_selected, y_train)
    nl_metrics = nl_model.evaluate(X_test[:, [feature_names.index(f) for f in selected_features]], y_test)
    print("Non-linear Regression Results:")
    print(f"RMSE: {nl_metrics['RMSE']:.2f}")
    print(f"R2 Score: {nl_metrics['R2 Score']:.2f}")

if __name__ == "__main__":
    main() 
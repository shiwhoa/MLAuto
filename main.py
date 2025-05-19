import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import (
    load_and_preprocess_data,
    get_feature_importance_scores,
    select_top_features,
    analyze_feature_correlations,
    apply_tsne,
    analyze_temperature_patterns
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

def print_cluster_stats(cluster_analysis):
    """Print statistics for each temperature cluster"""
    print("\nTemperature Cluster Analysis:")
    print("=" * 50)
    for cluster_name, cluster_data in cluster_analysis.items():
        stats = cluster_data['stats']
        print(f"\n{cluster_name}:")
        print(f"Temperature Range: {stats['temperature_range'][0]:.1f}°C to {stats['temperature_range'][1]:.1f}°C")
        print(f"Mean Temperature: {stats['mean_temperature']:.1f}°C")
        print(f"Mean Power: {stats['mean_power']:.2f}")
        print(f"Power Std Dev: {stats['std_power']:.2f}")
        print(f"Number of Samples: {stats['sample_size']}")

def main():
    # Load and preprocess data
    file_path = 'your_data.csv'  # Replace with your data file path
    
    # First, analyze correlations and feature importance with original data
    print("\nAnalyzing original data...")
    X_train, X_test, y_train, y_test, scaler, feature_names, _ = load_and_preprocess_data(
        file_path, use_pca=False
    )
    
    # Analyze feature correlations
    analyze_feature_correlations(X_train, feature_names, y_train)
    
    # Analyze temperature-based patterns
    print("\nAnalyzing temperature-based patterns...")
    cluster_analysis = analyze_temperature_patterns(
        X_train, y_train, feature_names, temp_column='ambient_temperature', n_clusters=3
    )
    
    # Print cluster statistics
    print_cluster_stats(cluster_analysis)
    
    # Train separate models for each temperature cluster
    print("\nTraining models for each temperature cluster...")
    cluster_models = {}
    
    for cluster_name, cluster_data in cluster_analysis.items():
        print(f"\nTraining models for {cluster_name}...")
        X_cluster, y_cluster = cluster_data['data']
        
        # Train XGBoost model
        xgb_model = XGBoostRegressor()
        xgb_model.train(X_cluster, y_cluster)
        
        # Train Linear Regression model
        lr_model = LinearRegressor()
        lr_model.train(X_cluster, y_cluster)
        
        # Train Non-linear Regression model
        nl_model = NonLinearRegressor()
        nl_model.train(X_cluster, y_cluster)
        
        cluster_models[cluster_name] = {
            'xgb': xgb_model,
            'linear': lr_model,
            'nonlinear': nl_model
        }
        
        # Print feature importance for this cluster
        print(f"\nFeature Importance for {cluster_name}:")
        importance_scores = cluster_data['importance']
        for method_name, scores in importance_scores.items():
            print_feature_importance(scores, method_name)
    
    # Apply PCA and analyze
    print("\nApplying PCA...")
    X_train_pca, X_test_pca, y_train, y_test, scaler, feature_names, pca = load_and_preprocess_data(
        file_path, use_pca=True, n_components=0.95
    )
    
    # Apply t-SNE for visualization
    print("\nApplying t-SNE for visualization...")
    X_tsne = apply_tsne(X_train)
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='viridis')
    plt.colorbar(scatter, label='Motor Power')
    plt.title('t-SNE Visualization of Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('tsne_visualization.png')
    plt.close()
    
    print("\nAnalysis complete! Check the generated visualization files:")
    print("- correlation_matrix.png: Feature correlation heatmap")
    print("- feature_importance.png: Feature importance plots")
    print("- pca_variance.png: PCA explained variance plot")
    print("- tsne_visualization.png: t-SNE visualization of the data")
    print("- temperature_clusters.png: Temperature distribution and clusters")
    print("- power_vs_temperature.png: Power vs Temperature by cluster")

if __name__ == "__main__":
    main() 
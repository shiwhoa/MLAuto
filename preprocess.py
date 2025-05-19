import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path, target_column='motor_power', test_size=0.2, use_pca=False, n_components=None):
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
    use_pca : bool
        Whether to apply PCA transformation
    n_components : int or float
        Number of components for PCA (if float, represents explained variance ratio)
        
    Returns:
    --------
    tuple
        Preprocessed and split data with feature names and transformers
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
    
    # Apply PCA if requested
    pca = None
    if use_pca:
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"\nPCA Results:")
        print(f"Number of components: {pca.n_components_}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.savefig('pca_variance.png')
        plt.close()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, feature_names, pca

def analyze_feature_correlations(X, feature_names, target=None):
    """
    Analyze and visualize feature correlations
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    feature_names : list
        List of feature names
    target : array-like, optional
        Target vector for correlation with features
    """
    # Create correlation matrix
    df = pd.DataFrame(X, columns=feature_names)
    if target is not None:
        df['target'] = target
    
    corr_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def analyze_temperature_patterns(X, y, feature_names, temp_column='ambient_temperature', n_clusters=3):
    """
    Analyze patterns in different temperature ranges
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    feature_names : list
        List of feature names
    temp_column : str
        Name of the temperature column
    n_clusters : int
        Number of temperature clusters to create
        
    Returns:
    --------
    dict
        Dictionary containing data and analysis for each temperature cluster
    """
    # Get temperature values
    temp_idx = feature_names.index(temp_column)
    temperatures = X[:, temp_idx]
    
    # Create temperature clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    temp_clusters = kmeans.fit_predict(temperatures.reshape(-1, 1))
    
    # Plot temperature distribution and clusters
    plt.figure(figsize=(12, 6))
    plt.hist(temperatures, bins=50, alpha=0.5, label='Temperature Distribution')
    for i in range(n_clusters):
        cluster_temps = temperatures[temp_clusters == i]
        plt.axvline(np.mean(cluster_temps), color=f'C{i}', linestyle='--',
                   label=f'Cluster {i+1} Mean: {np.mean(cluster_temps):.1f}°C')
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('Frequency')
    plt.title('Temperature Distribution and Clusters')
    plt.legend()
    plt.savefig('temperature_clusters.png')
    plt.close()
    
    # Analyze patterns for each temperature cluster
    cluster_analysis = {}
    for i in range(n_clusters):
        # Get data for this cluster
        cluster_mask = temp_clusters == i
        X_cluster = X[cluster_mask]
        y_cluster = y[cluster_mask]
        
        # Calculate statistics
        cluster_stats = {
            'temperature_range': (np.min(temperatures[cluster_mask]), np.max(temperatures[cluster_mask])),
            'mean_temperature': np.mean(temperatures[cluster_mask]),
            'mean_power': np.mean(y_cluster),
            'std_power': np.std(y_cluster),
            'sample_size': np.sum(cluster_mask)
        }
        
        # Calculate feature importance for this cluster
        importance_scores = get_feature_importance_scores(
            X_cluster, y_cluster, feature_names, plot=False
        )
        
        cluster_analysis[f'cluster_{i+1}'] = {
            'data': (X_cluster, y_cluster),
            'stats': cluster_stats,
            'importance': importance_scores
        }
    
    # Plot power vs temperature for each cluster
    plt.figure(figsize=(12, 6))
    for i in range(n_clusters):
        cluster_mask = temp_clusters == i
        plt.scatter(temperatures[cluster_mask], y[cluster_mask], 
                   alpha=0.5, label=f'Cluster {i+1}')
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('Motor Power')
    plt.title('Motor Power vs Temperature by Cluster')
    plt.legend()
    plt.savefig('power_vs_temperature.png')
    plt.close()
    
    return cluster_analysis

def get_feature_importance_scores(X, y, feature_names, plot=True):
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
    plot : bool
        Whether to create importance plots
        
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
    
    if plot:
        # Plot feature importance
        plot_feature_importance({
            'F-Regression': f_importance,
            'Mutual Information': mi_importance,
            'Random Forest': rf_importance,
            'XGBoost': xgb_importance
        })
    
    return {
        'F-Regression': f_importance,
        'Mutual Information': mi_importance,
        'Random Forest': rf_importance,
        'XGBoost': xgb_importance
    }

def plot_feature_importance(importance_dict):
    """
    Plot feature importance scores from different methods
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary containing feature importance scores
    """
    n_methods = len(importance_dict)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 5*n_methods))
    
    for (method_name, scores), ax in zip(importance_dict.items(), axes):
        # Sort features by importance
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        # Plot
        ax.barh(range(len(features)), importance)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_title(f'{method_name} Feature Importance')
        ax.set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

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

def apply_tsne(X, n_components=2):
    """
    Apply t-SNE for dimensionality reduction and visualization
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    n_components : int
        Number of components for t-SNE
        
    Returns:
    --------
    array-like
        Transformed data
    """
    tsne = TSNE(n_components=n_components, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    return X_tsne 
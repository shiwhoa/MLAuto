from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class NonLinearRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def train(self, X_train, y_train):
        """Train the non-linear regression model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions using the trained model"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'RMSE': rmse,
            'R2 Score': r2
        }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance scores"""
        importance = self.model.feature_importances_
        return dict(zip(feature_names, importance)) 
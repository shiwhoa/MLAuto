import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class XGBoostRegressor:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
    def train(self, X_train, y_train):
        """Train the XGBoost model"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions using the trained model"""
<<<<<<< HEAD
        return self.model.predict(X) 
=======
        return self.model.predict(X)
>>>>>>> ce991b43fb940c9c2283de0c417254b1b7fda415
    
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
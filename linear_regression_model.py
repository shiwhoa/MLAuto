from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class LinearRegressor:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """Train the linear regression model"""
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
    
    def get_coefficients(self, feature_names):
        """Get model coefficients"""
        return dict(zip(feature_names, self.model.coef_)) 
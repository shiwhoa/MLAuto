# Motor Power Prediction Models

This project implements three different regression models to predict motor power based on various input features:
- XGBoost Regression
- Linear Regression
- Non-linear Regression (Random Forest)

## Features Used
- 4 different mass flows
- Ambient temperature
- Gas turbine power

## Project Structure
- `preprocess.py`: Data preprocessing and preparation
- `xgboost_model.py`: XGBoost regression implementation
- `linear_regression_model.py`: Linear regression implementation
- `nonlinear_regression_model.py`: Non-linear regression implementation
- `main.py`: Main script to run all models
- `requirements.txt`: Project dependencies

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in a CSV file with the following columns:
   - mass_flow_1
   - mass_flow_2
   - mass_flow_3
   - mass_flow_4
   - ambient_temperature
   - gas_turbine_power
   - motor_power (target variable)

2. Update the `file_path` in `main.py` to point to your data file.

3. Run the main script:
```bash
python main.py
```

The script will:
- Preprocess the data
- Train all three models
- Evaluate their performance
- Display feature importance and coefficients

## Model Evaluation
Each model is evaluated using:
- Root Mean Square Error (RMSE)
- R-squared (R²) Score

## Feature Importance
- XGBoost and Non-linear models provide feature importance scores
- Linear Regression provides coefficient values for each feature 
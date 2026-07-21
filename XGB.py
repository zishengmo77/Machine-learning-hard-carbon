import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import itertools

# Read Excel data
file_path = 'datasetclean.xlsx'
data = pd.read_excel(file_path)

# Data preprocessing
print("Data types:")
print(data.dtypes)

# Convert all columns to numeric type
data = data.apply(pd.to_numeric, errors='coerce')

# Check and handle NaN values
print("\nNaN value statistics:")
nan_counts = data.isnull().sum()
print(nan_counts)

print("\nData structure:")
print(data.head())
print("Number of samples after cleaning:", data.shape[0])

# Define input features and target variables
X = data[['Temp', 'La', 'Lc', 'd002', 'ID_IG', 'SSA', 'Vt', 'I', 'CN']]
y_CE = data['CE']
y_SC = data['SC']
y_PC = data['PC']
y_C = data['C']

# Define hyperparameter search space
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Generate all parameter combinations
param_combinations = list(itertools.product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['learning_rate'],
    param_grid['subsample'],
    param_grid['colsample_bytree']
))

# Dictionary to store all optimization results
all_results = {
    'CE': [],
    'SC': [],
    'PC': [],
    'C': []
}

# Set target variables
targets = {
    'CE': y_CE,
    'SC': y_SC,
    'PC': y_PC,
    'C': y_C
}

# Perform hyperparameter optimization for each target
for target_name, y in targets.items():
    print(f"\nOptimizing hyperparameters for {target_name} model...")
    best_rmse = float('inf')
    best_params = None

    for params in param_combinations:
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree = params

        # Initialize XGBoost regressor
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )

        # 10-fold cross validation
        cv_scores = cross_val_score(
            model, X, y, cv=10,
            scoring='neg_mean_squared_error'
        )

        # Calculate average RMSE
        mse_cv = -cv_scores.mean()
        rmse_cv = np.sqrt(mse_cv)

        # Record results of current parameter combination
        result = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'RMSE': rmse_cv,
            'is_best': False
        }
        all_results[target_name].append(result)

        # Update the best parameter set
        if rmse_cv < best_rmse:
            best_rmse = rmse_cv
            best_params = params

    # Mark the optimal parameter combination
    for result in all_results[target_name]:
        if (result['n_estimators'] == best_params[0] and
                result['max_depth'] == best_params[1] and
                result['learning_rate'] == best_params[2] and
                result['subsample'] == best_params[3] and
                result['colsample_bytree'] == best_params[4]):
            result['is_best'] = True

    print(f"Best hyperparameters found for {target_name} model")

# Train final models with optimal parameters
best_models = {}

for target_name, y in targets.items():
    print(f"\nTraining {target_name} model with optimal parameters...")

    # Get the best parameter set
    best_result = next(result for result in all_results[target_name] if result['is_best'])

    # Create model with best parameters
    model = XGBRegressor(
        n_estimators=best_result['n_estimators'],
        max_depth=best_result['max_depth'],
        learning_rate=best_result['learning_rate'],
        subsample=best_result['subsample'],
        colsample_bytree=best_result['colsample_bytree'],
        random_state=42
    )

    # Train the model
    model.fit(X, y)

    # Store trained model
    best_models[target_name] = model
    print(f"{target_name} model training completed")

# Save all optimal models
joblib.dump(best_models, 'best_xgboost_models.pkl')
print("\nAll best models saved as best_xgboost_models.pkl")
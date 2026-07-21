import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
import itertools
import joblib

# Read Excel data
file_path = 'datasetclean.xlsx'
data = pd.read_excel(file_path)

# Data preprocessing
print("Data types:")
print(data.dtypes)

# Convert all columns to numeric type
data = data.apply(pd.to_numeric, errors='coerce')

# Check and count NaN values
print("\nNaN value statistics:")
nan_counts = data.isnull().sum()
print(nan_counts)

# Fill missing values with median
data = data.fillna(data.median())

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

# Fixed parameters for LightGBM
fixed_params = {
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Store all optimization results
all_results = {
    'CE': [],
    'SC': [],
    'PC': [],
    'C': []
}

# Collection of target variables
targets = {
    'CE': y_CE,
    'SC': y_SC,
    'PC': y_PC,
    'C': y_C
}

# Optimize hyperparameters for each target variable
for target_name, y in targets.items():
    print(f"\nOptimizing hyperparameters for {target_name} model...")
    best_rmse = float('inf')
    best_params = None

    for params in param_combinations:
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree = params

        # Initialize LightGBM regressor
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=fixed_params['random_state'],
            n_jobs=fixed_params['n_jobs'],
            verbose=fixed_params['verbose']
        )

        # 5-fold cross validation
        cv_scores = cross_val_score(
            model, X, y, cv=5,
            scoring='neg_mean_squared_error'
        )
        mse_cv = -cv_scores.mean()
        rmse_cv = np.sqrt(mse_cv)

        # Record results of current parameter set
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
    best_result = next(result for result in all_results[target_name] if result['is_best'])

    model = LGBMRegressor(
        n_estimators=best_result['n_estimators'],
        max_depth=best_result['max_depth'],
        learning_rate=best_result['learning_rate'],
        subsample=best_result['subsample'],
        colsample_bytree=best_result['colsample_bytree'],
        random_state=fixed_params['random_state'],
        n_jobs=fixed_params['n_jobs'],
        verbose=fixed_params['verbose']
    )

    model.fit(X, y)
    best_models[target_name] = model
    print(f"{target_name} model training completed")

# Save all trained optimal models
joblib.dump(best_models, 'best_lgbm_models.pkl')
print("\nAll optimal LightGBM models saved as best_lgbm_models.pkl")
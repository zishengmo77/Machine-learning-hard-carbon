import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
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

# Check and handle NaN values
print("\nNaN value statistics:")
nan_counts = data.isnull().sum()
print(nan_counts)

# Handle missing values (fill with median)
data = data.fillna(data.median())

print("\nData structure:")
print(data.head())
print("Number of samples after cleaning:", data.shape[0])

# Define input and output variables
X = data[['Temp', 'La', 'Lc', 'd002', 'ID_IG', 'SSA', 'Vt', 'I', 'CN']]
y_CE = data['CE']
y_SC = data['SC']
y_PC = data['PC']
y_C = data['C']

# Define hyperparameter search space
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.8]
}

# Generate all parameter combinations
param_combinations = list(itertools.product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_split'],
    param_grid['min_samples_leaf'],
    param_grid['max_features']
))

# Fixed parameters
fixed_params = {
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 0
}

# Store all optimization results
all_results = {
    'CE': [],
    'SC': [],
    'PC': [],
    'C': []
}

# Target variables collection
targets = {
    'CE': y_CE,
    'SC': y_SC,
    'PC': y_PC,
    'C': y_C
}

# Iterate over each target variable for hyperparameter optimization
for target_name, y in targets.items():
    print(f"\nOptimizing hyperparameters for {target_name} model...")
    best_rmse = float('inf')
    best_params = None

    for params in param_combinations:
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = params

        # Build Random Forest regressor model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=fixed_params['bootstrap'],
            oob_score=fixed_params['oob_score'],
            n_jobs=fixed_params['n_jobs'],
            random_state=fixed_params['random_state'],
            verbose=fixed_params['verbose']
        )

        # 5-fold cross-validation
        cv_scores = cross_val_score(
            model, X, y, cv=5,
            scoring='neg_mean_squared_error'
        )
        mse_cv = -cv_scores.mean()
        rmse_cv = np.sqrt(mse_cv)

        # Save results of current parameter combination
        result = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'RMSE': rmse_cv,
            'is_best': False
        }
        all_results[target_name].append(result)

        # Update best parameters if current RMSE is lower
        if rmse_cv < best_rmse:
            best_rmse = rmse_cv
            best_params = params

    # Mark the best parameter combination
    for result in all_results[target_name]:
        if (result['n_estimators'] == best_params[0] and
                result['max_depth'] == best_params[1] and
                result['min_samples_split'] == best_params[2] and
                result['min_samples_leaf'] == best_params[3] and
                result['max_features'] == best_params[4]):
            result['is_best'] = True

    print(f"Best hyperparameters found for {target_name} model")

# Train and save final models with best parameters
best_models = {}
for target_name, y in targets.items():
    print(f"\nTraining {target_name} model with best parameters...")
    best_result = next(result for result in all_results[target_name] if result['is_best'])

    model = RandomForestRegressor(
        n_estimators=best_result['n_estimators'],
        max_depth=best_result['max_depth'],
        min_samples_split=best_result['min_samples_split'],
        min_samples_leaf=best_result['min_samples_leaf'],
        max_features=best_result['max_features'],
        bootstrap=fixed_params['bootstrap'],
        oob_score=fixed_params['oob_score'],
        n_jobs=fixed_params['n_jobs'],
        random_state=fixed_params['random_state'],
        verbose=fixed_params['verbose']
    )

    model.fit(X, y)
    best_models[target_name] = model
    print(f"{target_name} model training completed")

# Save all best models
joblib.dump(best_models, 'best_rf_models.pkl')
print("\nAll best Random Forest models saved as best_rf_models.pkl")
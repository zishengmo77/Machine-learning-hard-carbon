import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing
import warnings


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

CPU_CORES = multiprocessing.cpu_count()
torch.set_num_threads(CPU_CORES)
torch.set_num_interop_threads(CPU_CORES // 2)
torch.backends.mkldnn.enabled = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Device: {DEVICE} | CPU Cores: {CPU_CORES}")

# Plot configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.switch_backend('Agg')


# ===================== TabPFN Model Definition =====================
class TabPFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(num_layers)
        ])
        self.out_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.input_fc(x))
        for layer in self.layers:
            x = x + layer(x)
        return self.out_fc(x)


class TabPFNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=None, hidden_dim=128, num_layers=4, dropout=0.1,
                 lr=0.001, epochs=80, batch_size=32, random_state=42, verbose=0,
                 patience=10, val_split=0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.patience = patience
        self.val_split = val_split
        self.model = None
        self.X_scaler = None  # Feature standard scaler
        self.y_scaler = None  # Target standard scaler
        torch.manual_seed(random_state)

    def fit(self, X, y):
        # Standardize features
        if self.X_scaler is None:
            self.X_scaler = StandardScaler()
            X_scaled = self.X_scaler.fit_transform(X)
        else:
            X_scaled = self.X_scaler.transform(X)

        # Standardize target values
        if self.y_scaler is None:
            self.y_scaler = StandardScaler()
            y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
        else:
            y_scaled = self.y_scaler.transform(y.values.reshape(-1, 1)).ravel()

        if self.input_dim is None:
            self.input_dim = X_scaled.shape[1]

        self.model = TabPFN(self.input_dim, self.hidden_dim, self.num_layers, self.dropout).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=self.val_split, random_state=self.random_state
        )

        X_tr = torch.from_numpy(X_train).float().to(DEVICE)
        y_tr = torch.from_numpy(y_train).float().unsqueeze(-1).to(DEVICE)
        X_va = torch.from_numpy(X_val).float().to(DEVICE)
        y_va = torch.from_numpy(y_val).float().unsqueeze(-1).to(DEVICE)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        best_val_loss = float('inf')
        patience_counter = 0
        self.model.train()

        for epoch in range(self.epochs):
            train_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_va), y_va).item()

            if self.verbose and (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1:3d} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stop at epoch {epoch + 1}")
                    break
            self.model.train()
        return self

    def predict(self, X):
        # Standardize input features
        if self.X_scaler is not None:
            X_scaled = self.X_scaler.transform(X)
        else:
            X_scaled = X

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
            pred_scaled = self.model(x_tensor)
            pred = pred_scaled.cpu().numpy()
            # Inverse transform to restore original scale
            if self.y_scaler is not None:
                pred = self.y_scaler.inverse_transform(pred)
            return pred.ravel()


# ===================== Load Dataset =====================
file_path = 'datasetclean.xlsx'
use_cols = ['Temp', 'La', 'Lc', 'd002', 'ID_IG', 'SSA', 'Vt', 'I', 'CN', 'CE', 'SC', 'PC', 'C']
data = pd.read_excel(file_path, usecols=use_cols)
data = data.apply(pd.to_numeric, errors='coerce')

feature_cols = ['Temp', 'La', 'Lc', 'd002', 'ID_IG', 'SSA', 'Vt', 'I', 'CN']
X = data[feature_cols]
y_CE = data['CE']
y_SC = data['SC']
y_PC = data['PC']
y_C = data['C']

results_list = []

# ===================== Hyperparameter Grid =====================
param_grid = {
    'hidden_dim': [64, 128],
    'num_layers': [2, 3],
    'dropout': [0.05, 0.1],
    'lr': [1e-3],
    'batch_size': [16, 32]
}


# ===================== Model Training & Evaluation Function =====================
def train_and_evaluate_model(X, y, model_name, error_threshold=20):
    print(f"\n========== {model_name} Hyperparameter Tuning & Training =========")
    base_model = TabPFNRegressor(epochs=150, random_state=42, verbose=1)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=CPU_CORES,
        verbose=1
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    print(f"{model_name} Best Params: {grid_search.best_params_} | Best CV Neg MSE: {grid_search.best_score_:.4f}")

    # 5-fold cross validation
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=CPU_CORES)
    mse_cv = -cv_scores.mean()
    print(f"{model_name} 5-fold CV MSE: {mse_cv:.4f}")

    # Split train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    best_model.fit(X_train, y_train)

    # Make predictions (auto inverse standardization)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate relative error percentage on original data scale
    train_error_pct = np.abs(y_train_pred - y_train) / (np.abs(y_train) + 1e-8) * 100
    test_error_pct = np.abs(y_test_pred - y_test) / (np.abs(y_test) + 1e-8) * 100
    high_error_mask_train = train_error_pct > error_threshold
    high_error_mask_test = test_error_pct > error_threshold
    high_error_count_train = np.sum(high_error_mask_train)
    high_error_count_test = np.sum(high_error_mask_test)

    # Calculate regression metrics
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    results_list.extend([
        {'Target': model_name, 'Dataset': 'Train', 'R2': round(r2_train, 4), 'MSE': round(mse_train, 4),
         'RMSE': round(rmse_train, 4), 'MAE': round(mae_train, 4)},
        {'Target': model_name, 'Dataset': 'Test', 'R2': round(r2_test, 4), 'MSE': round(mse_test, 4),
         'RMSE': round(rmse_test, 4), 'MAE': round(mae_test, 4)}
    ])

    # Plot prediction results
    plt.figure(figsize=(7, 6))

    plt.scatter(y_train[~high_error_mask_train], y_train_pred[~high_error_mask_train],
                color='gray', label=f'Train (≤{error_threshold}% Error)', alpha=0.6)
    plt.scatter(y_test[~high_error_mask_test], y_test_pred[~high_error_mask_test],
                color='#7030A0', label=f'Test (≤{error_threshold}% Error)', alpha=0.6)

    plt.scatter(y_train[high_error_mask_train], y_train_pred[high_error_mask_train],
                color='blue', marker='*', s=100,
                label=f'Train Error Points (≥{error_threshold}%): {high_error_count_train}', alpha=0.6)
    plt.scatter(y_test[high_error_mask_test], y_test_pred[high_error_mask_test],
                color='red', marker='*', s=100,
                label=f'Test Error Points (≥{error_threshold}%): {high_error_count_test}', alpha=0.6)

    lims = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
    plt.plot(lims, lims, 'k--', lw=2)

    # Confidence interval on original data scale
    all_actual = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    errors = all_pred - all_actual
    mean_error, std_error = np.mean(errors), np.std(errors)
    mean_actual = np.mean(all_actual)
    std_dev_min, std_dev_max = std_error * 0.5, std_error * 2.5
    x_range = np.linspace(all_actual.min(), all_actual.max(), 100)
    std_devs = std_dev_min + (std_dev_max - std_dev_min) * np.abs(x_range - mean_actual) / (lims[1] - lims[0])
    std_devs = np.clip(std_devs, std_dev_min, std_dev_max)

    plt.fill_between(x_range, x_range + mean_error - 2 * std_devs, x_range + mean_error + 2 * std_devs,
                     color='lightgrey', alpha=0.7, zorder=0)
    plt.text(0.05, 0.85, 'Train', transform=plt.gca().transAxes, fontsize=15, weight='bold')
    plt.text(0.05, 0.80, f'$R^2$: {r2_train:.2f}', transform=plt.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.75, f'RMSE: {rmse_train:.2f}', transform=plt.gca().transAxes, fontsize=14)
    plt.text(0.05, 0.65, 'Test', transform=plt.gca().transAxes, fontsize=15, weight='bold', color='blue')
    plt.text(0.05, 0.60, f'$R^2$: {r2_test:.2f}', transform=plt.gca().transAxes, fontsize=14, color='blue')
    plt.text(0.05, 0.55, f'RMSE: {rmse_test:.2f}', transform=plt.gca().transAxes, fontsize=14, color='blue')
    plt.text(0.35, 0.90, 'TabPFN', transform=plt.gca().transAxes, fontsize=24, weight='bold')

    plt.xlabel(f'Experimental {model_name}', fontsize=18, fontweight='bold')
    plt.ylabel(f'Predicted {model_name}', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(linestyle='--', alpha=0.7)
    plt.savefig(f"D:/PycharmProjects/2026+SIB1/plots/TabPFN__R2_{model_name}.png", dpi=600, transparent=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    return best_model


# ===================== Main Program =====================
if __name__ == "__main__":
    tabfpn_CE = train_and_evaluate_model(X, y_CE, "CE", error_threshold=5)
    tabfpn_SC = train_and_evaluate_model(X, y_SC, "SC", error_threshold=30)
    tabfpn_PC = train_and_evaluate_model(X, y_PC, "PC", error_threshold=20)
    tabfpn_C = train_and_evaluate_model(X, y_C, "C", error_threshold=25)

    # Save evaluation metrics to Excel
    results_df = pd.DataFrame(results_list)
    results_df.to_excel('D:/PycharmProjects/2026+SIB1/R2/metrics_TabPFN_tuned.xlsx', index=False)
    print("\nMetrics saved successfully")

    # Save trained models
    models = {'tabfpn_CE': tabfpn_CE, 'tabfpn_SC': tabfpn_SC, 'tabfpn_PC': tabfpn_PC, 'tabfpn_C': tabfpn_C}
    joblib.dump(models, '../models_tabfpn_tuned.pkl')
    print("Models saved successfully")
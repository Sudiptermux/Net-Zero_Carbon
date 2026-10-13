"""
train_models.py
----------------
Starter modeling pipeline for the "Net-Zero Smart Energy Zones" project.

Implements the fixes described in Section 5 of the Master Blueprint:
  - Walk-forward / TimeSeriesSplit validation (NOT a random 80/20 split)
  - Linear Regression baseline + Random Forest + XGBoost + LightGBM
  - RMSE / MAE / R^2 / MAPE per fold
  - SHAP TreeExplainer values for the best ensemble model
  - A simple paired significance test (Diebold-Mariano-style) between the top two models

Usage:
    python train_models.py --data data/processed/sambalpur_grid_telemetry.csv

Dependencies:
    pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib scipy joblib
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import shap
except Exception as e:
    shap = None
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

TARGET = "Net_Carbon_Balance"

# IMPORTANT: Net_Carbon_Balance is *exactly* Grid_Load_MW * Emission_Factor_kgCO2_per_MWh
# - Sequestration_Rate_kgCO2 at the SAME timestamp (see data_pipeline.py). If those three
# same-timestamp columns are used as features to predict the same-timestamp target, the
# model is just doing arithmetic, not forecasting -- R^2 will look near-perfect and mean
# nothing. To make this a genuine forecasting problem (matching the blueprint's own stated
# goal of predicting "the impending net carbon balance for the next hour"), the target is
# shifted forward by FORECAST_HORIZON hours, so the model only ever uses information
# available *up to now* to predict a balance it cannot yet compute directly.
FORECAST_HORIZON = 1  # hours ahead; raise to 24 for a day-ahead forecasting variant

FEATURE_COLS = [
    "Grid_Load_MW", "Emission_Factor_kgCO2_per_MWh", "Ambient_Temp_C", "Solar_Rad",
    "Zone_Soil_Moisture_Pct", "Carbon_Credit_Price_INR", "Sequestration_Rate_kgCO2",
    "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos", "Is_Weekend", "Is_COVID_Lockdown",
    "Region_East", "Region_West", "Region_South",
]


# --------------------------------------------------------------------------------------
# Feature engineering: lag features (grouped by Region)
# --------------------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, target: str = TARGET, lags=(1, 24)) -> pd.DataFrame:
    df_list = []
    for region, group in df.groupby("Region"):
        group = group.sort_values("Datetime").reset_index(drop=True)
        for lag in lags:
            group[f"{target}_lag{lag}"] = group[target].shift(lag)
        df_list.append(group)
    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    return df


def add_forecast_target(df: pd.DataFrame, target: str = TARGET, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Shifts the target forward by `horizon` hours and renames it `{target}_future` grouped by Region.
    """
    df_list = []
    for region, group in df.groupby("Region"):
        group = group.copy()
        group[f"{target}_future"] = group[target].shift(-horizon)
        df_list.append(group)
    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=[f"{target}_future"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
    }


class LSTMRegressor:
    def __init__(self, seq_len=6, epochs=12, batch_size=128, lr=0.005, hidden_dim=32, num_layers=2):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    class _LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out.squeeze(-1)

    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_seq, y_seq = [], []
        if self.seq_len > 1:
            for i in range(self.seq_len - 1, len(X_scaled)):
                X_seq.append(X_scaled[i - self.seq_len + 1 : i + 1])
                y_seq.append(y_scaled[i])
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
        else:
            X_seq = X_scaled[:, np.newaxis, :]
            y_seq = y_scaled

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X.shape[1]
        self.model = self._LSTMModel(input_dim, self.hidden_dim, self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.scaler_x.transform(X)
        
        X_seq = []
        if self.seq_len > 1:
            for i in range(len(X_scaled)):
                if i < self.seq_len - 1:
                    pad_len = self.seq_len - 1 - i
                    pad = [X_scaled[0]] * pad_len
                    seq = pad + list(X_scaled[0 : i + 1])
                else:
                    seq = X_scaled[i - self.seq_len + 1 : i + 1]
                X_seq.append(seq)
            X_seq = np.array(X_seq)
        else:
            X_seq = X_scaled[:, np.newaxis, :]

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        with torch.no_grad():
            preds_scaled = self.model(X_tensor).numpy()
            
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return preds


# --------------------------------------------------------------------------------------
# Walk-forward validation
# --------------------------------------------------------------------------------------

MODEL_BUILDERS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1),
    "LightGBM": lambda: LGBMRegressor(n_estimators=400, max_depth=-1, learning_rate=0.05,
                                       num_leaves=63, random_state=42, n_jobs=-1, verbosity=-1),
    "LSTM": lambda: LSTMRegressor(seq_len=6, epochs=12, batch_size=128, lr=0.005, hidden_dim=32, num_layers=2),
}



def run_walk_forward(df: pd.DataFrame, feature_cols, target_col: str, n_splits: int = 5):
    """
    TimeSeriesSplit gives expanding-window folds: each fold trains on all data up to a
    point and tests on the next chunk, which respects temporal order and avoids the
    look-ahead leakage that a random 80/20 split on hourly data would introduce.
    """
    X = df[feature_cols]
    y = df[target_col].values
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = {name: [] for name in MODEL_BUILDERS}
    fold_predictions = {name: [] for name in MODEL_BUILDERS}
    fold_truth = []

    last_models = {}

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fold_truth.append(y_test)

        for name, builder in MODEL_BUILDERS.items():
            model = builder()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            metrics["fold"] = fold_idx
            fold_results[name].append(metrics)
            fold_predictions[name].append(preds)
            last_models[name] = model  # keep the most recent fold's model for SHAP/inspection

        print(f"[fold {fold_idx}/{n_splits}] done "
              f"(train={len(train_idx)}, test={len(test_idx)})")

    return fold_results, fold_predictions, fold_truth, last_models


def summarize_results(fold_results: dict) -> pd.DataFrame:
    rows = []
    for name, folds in fold_results.items():
        dfm = pd.DataFrame(folds)
        rows.append({
            "model": name,
            "RMSE_mean": dfm["RMSE"].mean(), "RMSE_std": dfm["RMSE"].std(),
            "MAE_mean": dfm["MAE"].mean(),
            "R2_mean": dfm["R2"].mean(),
            "MAPE_%_mean": dfm["MAPE_%"].mean(),
        })
    return pd.DataFrame(rows).sort_values("RMSE_mean")


# --------------------------------------------------------------------------------------
# Significance test between the top two models (simple paired test on fold errors)
# --------------------------------------------------------------------------------------

def paired_significance_test(fold_predictions: dict, fold_truth: list, model_a: str, model_b: str):
    """
    A lightweight Diebold-Mariano-style test: compare squared errors of two models across
    the same test folds with a paired t-test. Reports whether the better model's lower
    error is statistically significant, not just a point-estimate difference.
    """
    errors_a, errors_b = [], []
    for preds_a, preds_b, truth in zip(fold_predictions[model_a], fold_predictions[model_b], fold_truth):
        errors_a.append((np.asarray(preds_a) - truth) ** 2)
        errors_b.append((np.asarray(preds_b) - truth) ** 2)
    errors_a = np.concatenate(errors_a)
    errors_b = np.concatenate(errors_b)
    t_stat, p_value = stats.ttest_rel(errors_a, errors_b)
    return {"model_a": model_a, "model_b": model_b, "t_stat": float(t_stat), "p_value": float(p_value)}


# --------------------------------------------------------------------------------------
# SHAP explainability for the best model
# --------------------------------------------------------------------------------------

def run_shap_explainability(model, X_sample: pd.DataFrame, feature_cols, out_dir: str, target_col: str = TARGET):
    os.makedirs(out_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample[feature_cols])

    plt.figure()
    shap.summary_plot(shap_values, X_sample[feature_cols], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150)
    plt.close()

    # waterfall for one representative high-balance and one low-balance hour
    high_idx = X_sample[target_col].idxmax() if target_col in X_sample.columns else 0
    low_idx = X_sample[target_col].idxmin() if target_col in X_sample.columns else 1

    base_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

    for label, idx in [("high", high_idx), ("low", low_idx)]:
        row_pos = X_sample.index.get_loc(idx)
        vals = shap_values[row_pos]
        
        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(vals))[::-1]
        top_k = 9
        top_indices = sorted_indices[:top_k]
        other_indices = sorted_indices[top_k:]
        
        waterfall_features = [feature_cols[i] for i in top_indices]
        waterfall_deltas = [vals[i] for i in top_indices]
        
        if len(other_indices) > 0:
            other_sum = np.sum(vals[other_indices])
            waterfall_features.append(f"{len(other_indices)} other features")
            waterfall_deltas.append(other_sum)
            
        # Reverse order for bottom-to-top plotting
        waterfall_features = list(reversed(waterfall_features))
        waterfall_deltas = list(reversed(waterfall_deltas))
        n_feats = len(waterfall_deltas)
        
        # Compute cumulative step positions
        running = base_val
        bottoms = []
        for d in waterfall_deltas:
            if d >= 0:
                bottoms.append(running)
                running += d
            else:
                running += d
                bottoms.append(running)
        final_val = running
        
        fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)
        y_pos = np.arange(n_feats)
        bar_colors = ['#e11d48' if d >= 0 else '#0284c7' for d in waterfall_deltas]
        
        # Horizontal step bars
        bars = ax.barh(y_pos, [abs(d) for d in waterfall_deltas], left=bottoms, height=0.62,
                        color=bar_colors, edgecolor='none', alpha=0.92)
        
        # Vertical reference line for baseline E[f(X)]
        ax.axvline(base_val, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.8)
        
        # Determine span for non-overlapping text placement outside the boxes
        all_vals = bottoms + [b + abs(d) for b, d in zip(bottoms, waterfall_deltas)] + [base_val, final_val]
        x_min, x_max = min(all_vals), max(all_vals)
        span = x_max - x_min
        ax.set_xlim(x_min - span * 0.12, x_max + span * 0.15)
        
        # Annotate values strictly OUTSIDE each bar (never inside the box)
        for i, (d, bottom) in enumerate(zip(waterfall_deltas, bottoms)):
            val_str = f"+{d:,.2f}" if d >= 0 else f"{d:,.2f}"
            text_color = '#be123c' if d >= 0 else '#0369a1'
            if d >= 0:
                x_text = bottom + d + span * 0.02
                ha = 'left'
            else:
                x_text = bottom - span * 0.02
                ha = 'right'
            ax.text(x_text, y_pos[i], val_str, va='center', ha=ha, fontsize=8.5, fontweight='bold', color=text_color)
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(waterfall_features, fontsize=9.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Model Output Net Carbon Balance (kgCO2/h)', fontsize=9.5, fontweight='bold', labelpad=6)
        title_str = "High NCB (Deficit Hour)" if label == "high" else "Low NCB (Surplus Hour)"
        ax.set_title(f"SHAP Waterfall Attribution: {title_str}\nE[f(X)] = {base_val:,.1f}  →  f(x) = {final_val:,.1f}",
                     fontsize=10.0, fontweight='bold', pad=10)
        
        plt.subplots_adjust(left=0.38, right=0.92, top=0.88, bottom=0.14)
        out_fig_path = os.path.join(out_dir, f"shap_waterfall_{label}.png")
        plt.savefig(out_fig_path, dpi=300)
        plt.close()

    print(f"SHAP plots saved to {out_dir}/")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate carbon-balance forecasting models.")
    parser.add_argument("--data", type=str, default="data/processed/multi_region_telemetry.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--report-dir", type=str, default="reports/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["Datetime"])
    
    # Lags and forecast targets must be calculated while we still have the Region column
    df = add_lag_features(df)
    df = add_forecast_target(df)
    target_col = f"{TARGET}_future"
    
    # Now pre-process Region column by one-hot encoding
    df = pd.get_dummies(df, columns=["Region"], drop_first=False)
    for col in ["Region_East", "Region_West", "Region_South"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0
            
    # Sort by Datetime so that regions are interleaved in time-series splits
    df = df.sort_values("Datetime").reset_index(drop=True)

    feature_cols = FEATURE_COLS + [f"{TARGET}_lag1", f"{TARGET}_lag24"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"Loaded {len(df)} rows, {len(feature_cols)} features. Forecasting {FORECAST_HORIZON}h ahead. "
          f"Running walk-forward validation with {args.n_splits} expanding-window folds ...")

    fold_results, fold_predictions, fold_truth, last_models = run_walk_forward(
        df, feature_cols, target_col, n_splits=args.n_splits
    )

    summary = summarize_results(fold_results)
    print("\n=== Model comparison (mean across folds) ===")
    print(summary.to_string(index=False))

    best_two = summary["model"].head(2).tolist()
    if len(best_two) == 2:
        sig = paired_significance_test(fold_predictions, fold_truth, best_two[0], best_two[1])
        print(f"\nSignificance test ({sig['model_a']} vs {sig['model_b']}): "
              f"t={sig['t_stat']:.3f}, p={sig['p_value']:.4f}")
        if sig["p_value"] < 0.05:
            print(f"-> {sig['model_a']}'s error reduction over {sig['model_b']} is statistically significant (p<0.05).")
        else:
            print("-> Difference between top two models is NOT statistically significant at p<0.05.")

    # Save trained models from the final fold
    os.makedirs(args.model_dir, exist_ok=True)
    for name, model in last_models.items():
        joblib.dump(model, os.path.join(args.model_dir, f"{name}_final_fold.joblib"))
    print(f"\nModels saved to {args.model_dir}/")

    # SHAP explainability on the best tree-based model
    tree_models = [m for m in summary["model"] if m in ("RandomForest", "XGBoost", "LightGBM")]
    if tree_models:
        best_tree_model_name = tree_models[0]
        print(f"\nGenerating SHAP explainability plots for the best tree-based model: {best_tree_model_name}...")
        sample = df.tail(min(2000, len(df))).copy()
        run_shap_explainability(last_models[best_tree_model_name], sample, feature_cols, args.report_dir, target_col)
    else:
        print("\nNo tree-based models found in results; skipping SHAP TreeExplainer.")


if __name__ == "__main__":
    main()

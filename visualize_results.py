import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Define constants matching train_models.py
TARGET = "Net_Carbon_Balance"
FORECAST_HORIZON = 1
# Recalculate helper classes or import
from train_models import LSTMRegressor, FEATURE_COLS

def main():
    print("Loading telemetry data...")
    df = pd.read_csv("data/processed/multi_region_telemetry.csv", parse_dates=["Datetime"])
    
    # 1. Recreate feature/target pipeline
    # Add lag features (region-grouped)
    from train_models import add_lag_features, add_forecast_target
    df = add_lag_features(df)
    df = add_forecast_target(df)
    target_col = f"{TARGET}_future"
    
    # Pre-process Region column by one-hot encoding
    df_encoded = pd.get_dummies(df, columns=["Region"], drop_first=False)
    for col in ["Region_East", "Region_West", "Region_South"]:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(int)
        else:
            df_encoded[col] = 0
            
    # Sort by Datetime (matching train_models.py splits)
    df_encoded = df_encoded.sort_values("Datetime").reset_index(drop=True)
    
    feature_cols = FEATURE_COLS + [f"{TARGET}_lag1", f"{TARGET}_lag24"]
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col].values
    
    # Slice final fold test set (matching 1/6th of interleaved dataset)
    test_size = len(df_encoded) // 6
    train_idx = np.arange(len(df_encoded) - test_size)
    test_idx = np.arange(len(df_encoded) - test_size, len(df_encoded))
    
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    test_dates = df_encoded.iloc[test_idx]["Datetime"]
    
    print(f"Loading final fold models from models/ directory...")
    models = {}
    for name in ["LinearRegression", "LightGBM", "LSTM"]:
        model_path = os.path.join("models", f"{name}_final_fold.joblib")
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"  Loaded {name} model.")
        else:
            print(f"  Warning: {name} model not found at {model_path}")
            
    print("Generating predictions...")
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X_test)
        
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plot 1: 72-Hour Actual vs Prediction Time Series Slice
    print("Generating Plot 1: 72-Hour Time Series slice...")
    slice_len = 72
    slice_start = 100  # Pick a slice in the test set
    
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract slices
    t_slice = test_dates.iloc[slice_start : slice_start + slice_len]
    y_true_slice = y_test[slice_start : slice_start + slice_len]
    
    plt.plot(t_slice, y_true_slice, label="Actual Net Carbon Balance", color="black", linewidth=2.5, zorder=5)
    
    colors = {"LinearRegression": "#3498db", "LightGBM": "#e74c3c", "LSTM": "#2ecc71"}
    for name, pred in preds.items():
        pred_slice = pred[slice_start : slice_start + slice_len]
        plt.plot(t_slice, pred_slice, label=f"Predicted ({name})", color=colors.get(name, "gray"), linestyle="--", alpha=0.9, linewidth=1.8)
        
    plt.title("Net Carbon Balance: 72-Hour Forecasting Prediction Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Future Net Carbon Balance (kgCO2)", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/figures/model_predictions_comparison.png", dpi=150)
    plt.close()
    print("  Saved model_predictions_comparison.png")
    
    # Plot 2: Model Performance Metrics Bar Chart
    print("Generating Plot 2: Performance metrics comparison...")
    
    metrics_data = {
        "model": ["LSTM", "LightGBM", "XGBoost", "RandomForest", "LinearRegression"],
        "RMSE": [2887.12, 3636.09, 3666.98, 4587.91, 5899.23],
        "MAE": [2134.15, 2630.98, 2662.02, 3322.59, 4519.28]
    }
    df_metrics = pd.DataFrame(metrics_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    x = np.arange(len(df_metrics))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, df_metrics["RMSE"], width, label="RMSE (kgCO2)", color="#1abc9c", alpha=0.95)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, df_metrics["MAE"], width, label="MAE (kgCO2)", color="#f39c12", alpha=0.85)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('RMSE (kgCO2)', color='#1abc9c', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1abc9c')
    ax2.set_ylabel('MAE (kgCO2)', color='#f39c12', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#f39c12')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_metrics["model"], fontsize=10, fontweight='semibold')
    plt.title("Model Metrics Comparison (Mean across Walk-Forward Folds)", fontsize=14, fontweight='bold', pad=15)
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)
    
    fig.tight_layout()
    plt.savefig("reports/figures/model_metrics_comparison.png", dpi=150)
    plt.close()
    print("  Saved model_metrics_comparison.png")
    
    # Plot 3: Diurnal Carbon Profile (Averages by Hour of Day)
    print("Generating Plot 3: Diurnal carbon and grid emission profile...")
    df["Hour"] = df["Datetime"].dt.hour
    diurnal_means = df.groupby("Hour")[["Grid_Load_MW", "Emission_Factor_kgCO2_per_MWh", "Sequestration_Rate_kgCO2", "Net_Carbon_Balance"]].mean()
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Carbon emissions = Grid_Load_MW * Emission_Factor_kgCO2_per_MWh
    emissions_diurnal = diurnal_means["Grid_Load_MW"] * diurnal_means["Emission_Factor_kgCO2_per_MWh"]
    
    ax1.plot(diurnal_means.index, emissions_diurnal, label="Grid Emissions (kgCO2/h)", color="#e74c3c", linewidth=2.5)
    ax1.plot(diurnal_means.index, diurnal_means["Sequestration_Rate_kgCO2"], label="Zone Sequestration (kgCO2/h)", color="#2ecc71", linewidth=2.5)
    ax1.fill_between(diurnal_means.index, 0, diurnal_means["Sequestration_Rate_kgCO2"], color="#2ecc71", alpha=0.15)
    
    ax2 = ax1.twinx()
    ax2.plot(diurnal_means.index, diurnal_means["Net_Carbon_Balance"], label="Net Carbon Balance (kgCO2/h)", color="#2c3e50", linewidth=3, linestyle="-.")
    
    ax1.set_xlabel("Hour of Day (24h)", fontsize=12)
    ax1.set_ylabel("Emissions & Sequestration (kgCO2/h)", fontsize=12)
    ax2.set_ylabel("Net Carbon Balance (kgCO2/h)", color="#2c3e50", fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#2c3e50')
    
    # Draw horizontal zero line on Net Carbon Balance
    ax2.axhline(0, color="#2c3e50", linestyle=":", alpha=0.5)
    
    # Set titles & legends
    plt.title("Diurnal Carbon Dynamics: Emissions vs Ecological Sequestration", fontsize=14, fontweight='bold', pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
    
    fig.tight_layout()
    plt.savefig("reports/figures/diurnal_carbon_profile.png", dpi=150)
    plt.close()
    print("  Saved diurnal_carbon_profile.png")
    print("All visualizations created successfully!")

if __name__ == "__main__":
    main()

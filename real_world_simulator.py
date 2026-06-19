import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom model wrapper so joblib can deserialize it
from train_models import LSTMRegressor, TARGET, FEATURE_COLS, FORECAST_HORIZON

def main():
    print("Loading telemetry data...")
    df = pd.read_csv("data/processed/multi_region_telemetry.csv", parse_dates=["Datetime"])
    
    # Re-calculate lag features and forecast target
    # Lags and forecast targets must be calculated while we still have the Region column
    from train_models import add_lag_features, add_forecast_target
    df_encoded = add_lag_features(df)
    df_encoded = add_forecast_target(df_encoded)
    
    # Now pre-process Region column by one-hot encoding (matching train_models.py exactly)
    df_encoded = pd.get_dummies(df_encoded, columns=["Region"], drop_first=False)
    for col in ["Region_East", "Region_West", "Region_South"]:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(int)
        else:
            df_encoded[col] = 0
            
    # Sort by Datetime so that regions are interleaved in time-series splits
    df_encoded = df_encoded.sort_values("Datetime").reset_index(drop=True)
    
    target_col = f"{TARGET}_future"
    feature_cols = FEATURE_COLS + [f"{TARGET}_lag1", f"{TARGET}_lag24"]
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]
    
    # We will simulate on the last week of the dataset (168 hours) for the "East" region
    # to show a specific localized real-world case study.
    print("Extracting test week slice for East region...")
    east_test = df_encoded[df_encoded["Region_East"] == 1].tail(168).copy().reset_index(drop=True)
    
    if len(east_test) < 168:
        print(f"Warning: Only {len(east_test)} rows available for test slice. Using all available.")
        
    X_test = east_test[feature_cols]
    y_true = east_test[target_col].values
    
    model_path = os.path.join("models", "LSTM_final_fold.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained LSTM model not found at {model_path}. Run train_models.py first.")
        
    print("Loading trained LSTM model...")
    model = joblib.load(model_path)
    
    print("Running forecast predictions...")
    preds = model.predict(X_test)
    
    # ----------------------------------------------------------------------------------
    # Simulation Parameters
    # ----------------------------------------------------------------------------------
    # Battery parameters
    BESS_CAPACITY_MWH = 100.0      # 100 MWh total capacity
    BESS_MAX_POWER_MW = 30.0       # Max charge/discharge rate = 30 MW
    bess_soc = BESS_CAPACITY_MWH / 2.0  # Start at 50% state of charge (50 MWh)
    bess_efficiency = 0.90         # 90% roundtrip efficiency
    
    # Carbon offset parameters
    # Discharging 1 MW for 1 hour saves (1 MWh * Grid Emission Factor) of carbon
    
    # Tracking variables
    std_carbon = []
    smart_carbon = []
    
    std_cost = []
    smart_cost = []
    
    cum_std_carbon = 0.0
    cum_smart_carbon = 0.0
    cum_std_cost = 0.0
    cum_smart_cost = 0.0
    
    soc_history = []
    
    # For cost hedging, keep a rolling window of carbon prices to determine if current price is "cheap"
    price_window = list(east_test["Carbon_Credit_Price_INR"].head(24).values)
    
    print("Executing hourly simulation...")
    for t in range(len(east_test)):
        row = east_test.iloc[t]
        actual_ncb = y_true[t]            # Actual net carbon balance at t+1 (kgCO2)
        predicted_ncb = preds[t]          # Forecasted net carbon balance at t+1 (kgCO2)
        
        ef = row["Emission_Factor_kgCO2_per_MWh"]  # kgCO2/MWh
        carbon_price = row["Carbon_Credit_Price_INR"] / 1000.0  # Convert INR/tonne to INR/kgCO2
        
        # Update rolling price average
        price_window.append(row["Carbon_Credit_Price_INR"])
        if len(price_window) > 48:
            price_window.pop(0)
        avg_price = np.mean(price_window)
        
        # ------------------------------------------------------------------------------
        # Standard Operation (No Battery, No Hedging)
        # ------------------------------------------------------------------------------
        # Emits exactly the actual net carbon balance
        actual_emit_std = max(0.0, actual_ncb)
        cum_std_carbon += actual_emit_std
        std_carbon.append(cum_std_carbon)
        
        # Cost is standard carbon credits purchased at the instant price
        cum_std_cost += actual_emit_std * carbon_price
        std_cost.append(cum_std_cost)
        
        # ------------------------------------------------------------------------------
        # Smart Operation (Model-Driven Battery Dispatch & CCTS Hedging)
        # ------------------------------------------------------------------------------
        battery_action = 0.0  # positive = discharging, negative = charging
        
        # A. Battery Dispatch Decision based on forecast
        if predicted_ncb > 1000.0:
            # We predict a high carbon deficit next hour -> discharge battery!
            # Determine how much we can discharge based on SOC and Max Power
            max_discharge_energy = (bess_soc - 0.1 * BESS_CAPACITY_MWH)  # Keep 10% safety buffer
            discharge_rate = min(BESS_MAX_POWER_MW, max_discharge_energy)
            discharge_rate = max(0.0, discharge_rate)
            
            bess_soc -= discharge_rate
            battery_action = discharge_rate
            
        elif predicted_ncb < -500.0:
            # We predict clean surplus (negative carbon balance) -> recharge battery!
            max_charge_energy = (0.95 * BESS_CAPACITY_MWH - bess_soc)    # Charge up to 95%
            charge_rate = min(BESS_MAX_POWER_MW, max_charge_energy)
            charge_rate = max(0.0, charge_rate)
            
            bess_soc += charge_rate * bess_efficiency
            battery_action = -charge_rate
            
        soc_history.append(bess_soc)
        
        # Actual carbon emitted after battery dispatch
        # Discharging reduces grid load, thus reducing emissions
        offset_carbon = battery_action * ef  # kgCO2 offset
        actual_emit_smart = max(0.0, actual_ncb - offset_carbon)
        cum_smart_carbon += actual_emit_smart
        smart_carbon.append(cum_smart_carbon)
        
        # B. Cost Hedging Decision
        # If model predicts we will need credits next hour, and current price is cheap (< rolling average)
        # buy credits now. Otherwise, buy only what we need at t+1.
        if predicted_ncb > 0.0 and row["Carbon_Credit_Price_INR"] < avg_price:
            # Hedged purchase: buy next hour's forecasted need at current lower price
            purchase_price = carbon_price
            cum_smart_cost += actual_emit_smart * purchase_price
        else:
            # Spot purchase: buy at actual price next hour
            cum_smart_cost += actual_emit_smart * carbon_price
            
        smart_cost.append(cum_smart_cost)
        
    print("Simulation finished.")
    
    # ----------------------------------------------------------------------------------
    # Results & Metrics
    # ----------------------------------------------------------------------------------
    total_carbon_saved = cum_std_carbon - cum_smart_carbon
    pct_carbon_saved = (total_carbon_saved / cum_std_carbon) * 100
    
    total_cost_saved = cum_std_cost - cum_smart_cost
    pct_cost_saved = (total_cost_saved / cum_std_cost) * 100
    
    print("\n================== SIMULATION RESULTS ==================")
    print(f"Standard Total Carbon Liability:    {cum_std_carbon:,.2f} kgCO2")
    print(f"Smart Model-Driven Carbon:          {cum_smart_carbon:,.2f} kgCO2")
    print(f"--> Carbon Emissions Saved:         {total_carbon_saved:,.2f} kgCO2 ({pct_carbon_saved:.2f}%)")
    print(f"Standard Carbon Compliance Cost:    {cum_std_cost:,.2f} INR")
    print(f"Smart Model-Driven Cost:            {cum_smart_cost:,.2f} INR")
    print(f"--> Financial Savings:              {total_cost_saved:,.2f} INR ({pct_cost_saved:.2f}%)")
    print("========================================================\n")
    
    # ----------------------------------------------------------------------------------
    # Plotting the Dashboard
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(14, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Subplot 1: Cumulative Carbon Emissions Comparison
    plt.subplot(3, 1, 1)
    time_steps = np.arange(len(east_test))
    plt.plot(time_steps, std_carbon, label="Standard Operations (No Forecasting)", color="#e74c3c", linewidth=2.5)
    plt.plot(time_steps, smart_carbon, label="Model-Driven Smart Dispatch (LSTM)", color="#2ecc71", linewidth=2.5)
    plt.fill_between(time_steps, smart_carbon, std_carbon, color="#2ecc71", alpha=0.15)
    plt.title("Cumulative Carbon Emissions Avoided", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Emissions (kgCO2)")
    plt.legend(loc="upper left")
    
    # Subplot 2: Cumulative Financial Cost Comparison
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, std_cost, label="Standard Compliance Costs", color="#c0392b", linewidth=2.5)
    plt.plot(time_steps, smart_cost, label="Smart Hedged CCTS Costs (LSTM)", color="#27ae60", linewidth=2.5)
    plt.fill_between(time_steps, smart_cost, std_cost, color="#2ecc71", alpha=0.15)
    plt.title("Cumulative Carbon Compliance Costs (CCTS)", fontsize=12, fontweight='bold')
    plt.ylabel("Cumulative Cost (INR)")
    plt.legend(loc="upper left")
    
    # Subplot 3: Battery State of Charge (SOC) History
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, soc_history, label="Battery State of Charge (SOC)", color="#3498db", linewidth=2)
    plt.axhline(BESS_CAPACITY_MWH, color="gray", linestyle="--", alpha=0.7)
    plt.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    plt.title("Battery Energy Storage System (BESS) Dynamics", fontsize=12, fontweight='bold')
    plt.ylabel("SOC (MWh)")
    plt.xlabel("Simulation Hour (Test Week in December)")
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    chart_path = "reports/figures/real_world_savings_dashboard.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    
    print(f"Dashboard visualization saved to: {chart_path}")

if __name__ == "__main__":
    main()

# recommendation_engine_irish.py
# AI Vehicle Recommendation Tool (Irish Scenario) – using ai_ready_vehicle_dataset.csv
# ------------------------------------------------------------------------------------
# In this script, I load my AI-ready dataset, apply Irish cost assumptions,train a prediction model, and recommend vehicles to users based on budget and driving needs.


#importing the necessary dependencies first
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# my Irish cost assumptions
# here I use actual Irish average costs for electricity, fuel, and maintenance.
ELECTRICITY_PRICE_PER_KWH = 0.34     # EUR/kWh – average Irish residential rate (2024)
FUEL_PRICE_PER_L = 1.75              # EUR/L – average petrol/diesel price in Ireland (2024)
MAINTENANCE_PER_YEAR_ICE = 550       # EUR/year – ICE maintenance average in Ireland
MAINTENANCE_PER_YEAR_EV = 350        # EUR/year – EV maintenance average in Ireland
DEPRECIATION_RATE_5YR = 0.35         # 35% depreciation over 5 years
BATTERY_REPLACEMENT_COST_DEFAULT = 7000  # EUR – estimated for Irish market
DEFAULT_ANNUAL_KM = 15000            # km/year – average annual distance in Ireland

# here I am Loading my AI-ready dataset
df = pd.read_csv("ai_ready_vehicle_dataset.csv")

# My deterministic TCO calculator
# This function calculates a vehicle's exact 5-year Total Cost of Ownership for a specific monthly distance, using Irish cost assumptions.
def deterministic_5yr_tco(row, monthly_km):
    annual_km = monthly_km * 12
    purchase = row['purchase_price_eur']

    veh_type = str(row.get('veh_type', '')).lower()

    if veh_type == 'ev':
        # for EVs, I convert Wh/km to kWh/km and apply electricity price
        kwh_per_km = row.get('el_consumpt_whkm', np.nan) / 1000.0
        energy_cost_5yr = kwh_per_km * ELECTRICITY_PRICE_PER_KWH * annual_km * 5
        maintenance_5yr = MAINTENANCE_PER_YEAR_EV * 5
    else:
        # for ICE/hybrid, I use fuel consumption in L/100km
        litres_per_km = row.get('fuel_consumption', np.nan) / 100.0
        energy_cost_5yr = litres_per_km * FUEL_PRICE_PER_L * annual_km * 5
        maintenance_5yr = MAINTENANCE_PER_YEAR_ICE * 5

    # Depreciation calculation
    depreciation_5yr = purchase * DEPRECIATION_RATE_5YR

    # Battery replacement check for EVs with shorter range
    battery_cost = 0
    if veh_type == 'ev':
        electric_range = row.get('electric_range_(km)', row.get('electric_range_km', np.nan))
        if not pd.isna(electric_range) and electric_range < 300:
            battery_cost = BATTERY_REPLACEMENT_COST_DEFAULT

    # Total TCO
    tco_5yr = purchase + energy_cost_5yr + maintenance_5yr + depreciation_5yr + battery_cost

    return {
        'tco_5yr': float(tco_5yr),
        'energy_5yr': float(energy_cost_5yr),
        'maintenance_5yr': float(maintenance_5yr),
        'depreciation_5yr': float(depreciation_5yr),
        'battery_cost': float(battery_cost),
        'annual_km': annual_km
    }

# now I have to prep data for my ML model
# I create an 'annual_km' column so my model knows the assumed Irish average.
df['annual_km'] = DEFAULT_ANNUAL_KM

# only numeric features are needed for training
features = [col for col in [
    'purchase_price_eur', 'el_consumpt_whkm', 'fuel_consumption',
    'electric_range_(km)', 'power_kw'
] if col in df.columns]

# Target column is the 5-year TCO from my dataset
target_col = '5yr_tco_eur'

# Removing rows with missing values in my features or target
ml_df = df[features + [target_col]].dropna().copy()

X = ml_df[features]
y = ml_df[target_col]

# now the data has to be split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Training my prediction model

# I chose RandomForest because it handles mixed data well and is robust to outliers.
rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# now to evaluate the model accuracy
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, y_pred)

# and now to save the trained model
joblib.dump(rf, "rf_tco_model_irish.joblib")

print(f"✅ RandomForest trained (Irish rates) -- R2: {r2:.3f}, RMSE: {rmse:.2f} EUR")






# My vehicle recommendation function
# this returns affordable and stretch options based on the user budget and monthly distance.
def recommend_vehicles(budget_eur, monthly_km, stretch_pct=0.10, top_n=10):
    results = []
    for _, row in df.iterrows():
        det = deterministic_5yr_tco(row, monthly_km)

        # ML prediction for comparison
        feat = {f: (monthly_km * 12 if f == 'annual_km' else row.get(f, np.nan)) for f in features}
        ml_pred = rf.predict(pd.DataFrame([feat]))[0]

        results.append({
            'brand': row.get('brand'),
            'model': row.get('veh_model', row.get('version')),
            'veh_type': row.get('veh_type'),
            'energy': row.get('energy'),
            'power_kw': row.get('power_kw'),
            'el_consumpt_whkm': row.get('el_consumpt_whkm'),
            'fuel_consumption': row.get('fuel_consumption'),
            'electric_range_(km)': row.get('electric_range_(km)'),
            'purchase_price_eur': row.get('purchase_price_eur'),
            'det_5yr_tco': det['tco_5yr'],
            'ml_pred_5yr_tco': ml_pred
        })

    res_df = pd.DataFrame(results).sort_values(by='det_5yr_tco')

    affordable = res_df[res_df['det_5yr_tco'] <= budget_eur].head(top_n)
    stretch_budget = budget_eur * (1 + stretch_pct)
    stretch = res_df[(res_df['det_5yr_tco'] > budget_eur) & (res_df['det_5yr_tco'] <= stretch_budget)].head(top_n)

    return affordable.reset_index(drop=True), stretch.reset_index(drop=True), {
        'budget': budget_eur,
        'stretch_budget': stretch_budget,
        'monthly_km': monthly_km,
        'model_r2': r2,
        'model_rmse': rmse
    }

# Example usage for demonstration
if __name__ == "__main__":
    print("🚗 AI Vehicle Recommendation Tool (Irish Scenario)")
    print("--------------------------------------------------")

    try:
        budget = float(input("Enter your vehicle budget in EUR: "))
        monthly_km = float(input("Enter your average monthly distance (in km): "))
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")
        exit()

    print("\n🔍 Calculating best options based on your input...\n")
    affordable, stretch, info = recommend_vehicles(budget, monthly_km, stretch_pct=0.15, top_n=10)

    print("--- Recommendation Summary ---")
    print(info)

    if not affordable.empty:
        print("\n✅ Affordable Options:")
        print(affordable.to_string(index=False))
    else:
        print("\n⚠️ No vehicles found within your budget.")

    if not stretch.empty:
        print("\n💡 Stretch Options (within 15% over budget):")
        print(stretch.to_string(index=False))
    else:
        print("\nℹ️ No stretch options found either.")

    
    # just to save the recommendations to CSV for using in my report, along with the budget and km in the filename included automatically so that everytime I run a new scenario, a unique file is created without overwriting the existing file
    affordable_filename = f"affordable_recommendations_budget{budget}_km{monthly_km}.csv"
    stretch_filename = f"stretch_recommendations_budget{budget}_km{monthly_km}.csv"

    affordable.to_csv(affordable_filename, index=False)
    stretch.to_csv(stretch_filename, index=False)

    print(f"\n📂 Saved recommendations as '{affordable_filename}' and '{stretch_filename}'")


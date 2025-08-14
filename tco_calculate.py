import pandas as pd

# load the primary dataset
df = pd.read_csv("vehicles.csv")


electricity_price_per_kwh = 0.30   # EUR/kWh
fuel_price_per_litre = 1.80        # EUR/L
maintenance_per_year_ice = 500
maintenance_per_year_ev = 300
depreciation_rate_5yr = 0.35       # 35% value loss after 5 years
battery_replacement_cost = 5000    # EUR (only for EVs)



# Calculate 5-year energy cost
def calc_energy_cost(row):
    if row["Veh_type"].lower() == "ev":
        # Convert Wh/km to kWh/100km → cost per km
        kwh_per_km = row["El_Consumpt_whkm"] / 1000
        return kwh_per_km * electricity_price_per_kwh * (12000 * 5)  # 12,000 km/year × 5
    else:
        # Fuel consumption is in L/100km → cost per km
        litres_per_km = row["Fuel consumption"] / 100
        return litres_per_km * fuel_price_per_litre * (12000 * 5)

df["5yr_energy_cost_eur"] = df.apply(calc_energy_cost, axis=1)

# Calculate 5-year maintenance cost
def calc_maintenance(row):
    if row["Veh_type"].lower() == "ev":
        return maintenance_per_year_ev * 5
    else:
        return maintenance_per_year_ice * 5

df["5yr_maintenance_cost_eur"] = df.apply(calc_maintenance, axis=1)

# Calculate 5-year depreciation loss
df["5yr_depreciation_loss_eur"] = df["purchase_price_eur"] * depreciation_rate_5yr

# Battery replacement cost (only for EV, else 0)
def calc_battery_cost(row):
    if row["Veh_type"].lower() == "ev" and row["Electric range (km)"] < 300:
        return battery_replacement_cost
    return 0

df["battery_replacement_cost_eur"] = df.apply(calc_battery_cost, axis=1)

# Calculate 5-year TCO
df["5yr_TCO_eur"] = (
    df["purchase_price_eur"]
    + df["5yr_energy_cost_eur"]
    + df["5yr_maintenance_cost_eur"]
    + df["5yr_depreciation_loss_eur"]
    + df["battery_replacement_cost_eur"]
)

# Save to new CSV
df.to_csv("dataset_with_tco.csv", index=False)
print("✅ Dataset saved as dataset_with_tco.csv with new cost columns.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# first I load the dataset and trained model
df = pd.read_csv("ai_ready_vehicle_dataset.csv")
rf = joblib.load("rf_tco_model_irish.joblib")

# now to define the features and target
features = [
    "purchase_price_eur", "el_consumpt_whkm", "fuel_consumption",
    "electric_range_(km)", "power_kw"
]
target_col = "5yr_tco_eur"

# add default annual km for a consistent baseline
df["annual_km"] = 15000

# preparing the data
ml_df = df[features + [target_col, "energy"]].dropna().copy()
X = ml_df[features]
y = ml_df[target_col]
y_pred = rf.predict(X)
ml_df["predicted_tco"] = y_pred



# Plot 1: Actual vs Predicted TCO
plt.figure(figsize=(8,6))
sns.scatterplot(x=y, y=y_pred)
plt.xlabel("Actual 5yr TCO (EUR)")
plt.ylabel("Predicted 5yr TCO (EUR)")
plt.title("Model Performance: Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.grid(True)
plt.tight_layout()
plt.savefig("1.model_performance_actual_vs_predicted.png")

# Plot 2: Feature Importance
importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("2.feature_importance.png")

# Plot 3: Distribution of Predicted TCO
plt.figure(figsize=(8,5))
sns.histplot(y_pred, kde=True, bins=30)
plt.title("Distribution of Predicted 5yr TCO")
plt.xlabel("Predicted 5yr TCO (EUR)")
plt.tight_layout()
plt.savefig("3.predicted_tco_distribution.png")

# 4 Scatter plot: Actual vs Predicted TCO by Energy Type
plt.figure(figsize=(10, 6))
sns.scatterplot(data=ml_df, x=target_col, y="predicted_tco", hue="energy", palette="Set2", alpha=0.7, edgecolor="w", s=80)
plt.plot([ml_df[target_col].min(), ml_df[target_col].max()],
         [ml_df[target_col].min(), ml_df[target_col].max()],
         'r--', label='Ideal Prediction Line')

plt.title("Actual vs Predicted 5-Year TCO by Energy Type")
plt.xlabel("Actual 5-Year TCO (EUR)")
plt.ylabel("Predicted 5-Year TCO (EUR)")
plt.legend()
plt.tight_layout()
plt.savefig("4.actual_vs_predicted_by_energy.png")
plt.show()


# Plot 5: Predicted TCO Distribution by Energy Type
plt.figure(figsize=(10, 6))
sns.histplot(data=ml_df, x="predicted_tco", hue="energy", kde=True, bins=30, palette="tab10")
plt.title("Predicted 5-Year TCO Distribution by Energy Type")
plt.xlabel("Predicted 5-Year TCO (EUR)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("5.predicted_tco_by_energy_type.png")


ml_df = df[features + [target_col, "veh_type"]].dropna().copy()



# Predict TCO and add it
X = ml_df[features]
y = ml_df[target_col]
y_pred = rf.predict(X)
ml_df["predicted_tco"] = y_pred

# Plot 6: Predicted TCO by Vehicle Type
valid_types = ["SUV", "Hatchback", "Sedan", "Crossover", "MPV", "City", "Supermini", "Saloon", "Estate", "Liftback"]
ml_df = ml_df[ml_df["veh_type"].isin(valid_types)]
plt.figure(figsize=(10, 6))
sns.histplot(data=ml_df, x="predicted_tco", hue="veh_type", kde=True, bins=30, palette="colorblind")
plt.title("Predicted 5-Year TCO Distribution by Vehicle Type")
plt.xlabel("Predicted 5-Year TCO (EUR)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("6.predicted_tco_by_veh_type.png")



print("All visualizations saved as PNG files.")

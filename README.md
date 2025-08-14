AI Vehicle Recommendation Tool (Irish Scenario)

This project uses cost assumptions and machine learning to recommend vehicles in Ireland based on a user's budget and monthly driving distance. It predicts the 5-year Total Cost of Ownership (TCO) using both rule-based and data-driven approaches.

📁 Project Structure
.
├── vehicles.csv                         # Raw vehicle data
├── tco_calculate.py                     # Calculates 5-year TCO and cost components
├── dataset_with_tco.csv                 # Dataset after TCO calculation
├── clean_dataset.py                     # Removes duplicates and fills missing values
├── cleaned_vehicle_dataset.csv          # Intermediate cleaned dataset
├── preprocess_dataset.py                # Normalizes columns and finalizes data
├── ai_ready_vehicle_dataset.csv         # Final dataset used for ML and predictions
├── recommendation_engine.py             # Main script (ML training + recommendation tool)
├── rf_tco_model_irish.joblib            # Trained RandomForest model
├── affordable_recommendations_*.csv     # Outputs: vehicles within user budget
├── stretch_recommendations_*.csv        # Outputs: vehicles slightly above budget


⚙️ How to Run the Project

Clone the repository:

git clone https://github.com/AnanthuPSankar/DissertationDBS.git
cd DissertationDBS


Install required Python libraries:

pip install pandas numpy scikit-learn joblib


Run the preprocessing pipeline:

python tco_calculate.py
python clean_dataset.py
python preprocess_dataset.py


Launch the recommender tool:

python recommendation_engine.py


Enter your inputs when prompted:

Budget (EUR)

Average monthly driving distance (in km)

View your recommended vehicles:

Affordable and stretch options are printed in the terminal

Recommendations are saved as CSV files with dynamic filenames

📊 Features

Calculates realistic 5-year TCO (Total Cost of Ownership)

Irish market assumptions for energy, fuel, maintenance

Predictive ML model (Random Forest Regressor)

Dynamic user inputs (budget + distance)

Saves output for easy reporting and comparison

*  Irish Assumptions Used

Electricity: €0.34/kWh

Fuel (petrol/diesel): €1.75/L

Maintenance: €550/year (ICE), €350/year (EV)

Battery replacement: €7,000 (for EVs < 300 km range)

Depreciation: 35% over 5 years

Default driving: 15,000 km/year (modifiable)

🔍 Output Sample

Each run generates:

affordable_recommendations_budget<BUDGET>_km<MONTHLYKM>.csv

stretch_recommendations_budget<BUDGET>_km<MONTHLYKM>.csv

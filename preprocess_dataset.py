import pandas as pd

#firsdt load my saved dataset
df = pd.read_csv("cleaned_vehicle_dataset.csv")

# now i have to run a quick check
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# if any duplicates exist further, they have to be dropped
df.drop_duplicates(inplace=True)

# the column names have to be fixed/normalized into a single format for all (lowercase and replace spaces with underscores)
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# if any unrealistic values are present, they have to be removed
df = df[df['purchase_price_eur'] > 0]             # price must be positive
df = df[df['electric_range_(km)'] > 0]          # range must be positive as well

# to identify the numeric and text columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
text_cols = df.select_dtypes(include=['object']).columns

#now the missing numbers to be filled with the column mean
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))


# to fill missing text with 'Unknown'
df[text_cols] = df[text_cols].apply(lambda x: x.fillna("Unknown"))

# finally, save the cleaned & normalized dataset as a new file
df.to_csv("ai_ready_vehicle_dataset.csv", index=False)
print("AI-ready dataset saved as ai_ready_vehicle_dataset.csv")

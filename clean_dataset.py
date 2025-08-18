import pandas as pd

#to load my dataset from the file
df=pd.read_csv("dataset_with_tco.csv")

#now, the first rows should be shown
print("Original Data:\n", df.head())


#although I have refined the datasets abit, there could be duplicates, now to drop them
df.drop_duplicates(inplace=True)

#to fill missing numbers with column mean, and to fill missing text with 'unknown' to reduce errors
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna("unknown")



# now to save my cleaned dataset
df.to_csv("cleaned_vehicle_dataset.csv", index=False)
print("Cleaned dataset saved as cleaned_vehicle_dataset.csv") 



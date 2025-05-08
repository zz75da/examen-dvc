import pandas as pd

# Load original data
df = pd.read_csv("data/raw/raw.csv")

# Drop the date column (adjust name if needed)
df = df.drop(columns=["date"])

# Save cleaned version
df.to_csv("data/processed/raw_cleaned.csv", index=False)

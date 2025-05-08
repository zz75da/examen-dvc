import pandas as pd
import os

# Load original data
df = pd.read_csv("data/raw/raw.csv")

# Drop the date column (adjust name if needed)
df = df.drop(columns=["date"])

# Save cleaned version
os.makedirs("data/preprocessed", exist_ok=True)
df.to_csv("data/preprocessed/raw_cleaned.csv", index=False)

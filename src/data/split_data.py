import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/processed/raw_cleaned.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("data/split", exist_ok=True)
X_train.to_csv("data/split/X_train.csv", index=False)
X_test.to_csv("data/split/X_test.csv", index=False)
y_train.to_csv("data/split/y_train.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)

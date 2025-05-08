import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

X_train = pd.read_csv("data/split/X_train.csv")
X_test = pd.read_csv("data/split/X_test.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("data/normalized", exist_ok=True)
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/normalized/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/normalized/X_test_scaled.csv", index=False)

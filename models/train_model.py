import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

X_train = pd.read_csv("/home/ubuntu/masterDVCexam/examen-dvc/data/preprocessed/X_train_scaled.csv")
y_train = pd.read_csv("/home/ubuntu/masterDVCexam/examen-dvc/data/preprocessed/y_train.csv").values.ravel()
best_params = joblib.load("/home/ubuntu/masterDVCexam/examen-dvc/models/best_params.pkl")

model = RandomForestRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

joblib.dump(model, "/home/ubuntu/masterDVCexam/examen-dvc/models/trained_model.pkl")

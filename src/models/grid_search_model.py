import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

X_train = pd.read_csv("/home/ubuntu/masterDVCexam/examen-dvc/data/preprocessed/X_train_scaled.csv")
y_train = pd.read_csv("/home/ubuntu/masterDVCexam/examen-dvc/data/preprocessed/y_train.csv").values.ravel()

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

os.makedirs("/home/ubuntu/masterDVCexam/examen-dvc/models", exist_ok=True)
joblib.dump(grid.best_params_, "/home/ubuntu/masterDVCexam/examen-dvc/models/best_params.pkl")

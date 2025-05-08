import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os
import yaml

# Load params from YAML
with open("params.yaml") as f:
    params_yaml = yaml.safe_load(f)

params = {
    "n_estimators": params_yaml["train"]["n_estimators"],
    "max_depth": params_yaml["train"]["max_depth"],
    "min_samples_split": params_yaml["train"]["min_samples_split"],
}


X_train = pd.read_csv("data/normalized/X_train_scaled.csv")
y_train = pd.read_csv("data/split/y_train.csv").values.ravel()



grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(grid.best_params_, "models/best_params.pkl")

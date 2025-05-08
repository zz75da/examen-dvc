import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

X_test = pd.read_csv("data/normalized/X_test_scaled.csv")
y_test = pd.read_csv("data/split/y_test.csv").values.ravel()
model = joblib.load("models/trained_model.pkl")

predictions = model.predict(X_test)

# Enregistrement des prédictions
os.makedirs("data/prediction", exist_ok=True)
pd.DataFrame({
    "y_true": y_test,
    "y_pred": predictions
}).to_csv("data/prediction/predictions.csv", index=False)

# Évaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f, indent=4)

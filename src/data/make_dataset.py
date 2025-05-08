import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

df = pd.read_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/raw/raw_cleaned.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed", exist_ok=True)
X_train.to_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/X_train.csv", index=False)
X_test.to_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/X_test.csv", index=False)
y_train.to_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/y_train.csv", index=False)
y_test.to_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/y_test.csv", index=False)


X_train = pd.read_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/X_train.csv")
X_test = pd.read_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/X_test.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/X>pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("/home/ubuntu/masterDVCexam/examen_dvc_Zeghoud/data/preprocessed/X_t>

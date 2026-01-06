import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("preprocessed_data.csv")

X = df.drop(columns=["C6H6(GT)"])
y = df["C6H6(GT)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("AirQuality_Baseline_Model")

mlflow.sklearn.autolog()

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
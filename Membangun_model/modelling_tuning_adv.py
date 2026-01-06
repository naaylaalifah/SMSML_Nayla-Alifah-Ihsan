import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


os.environ["MLFLOW_TRACKING_USERNAME"] = "naaylaalifah"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bf3f2e3af436622f67f11029e89b7ce43f4027b5"

from dagshub import init
init(
    repo_owner="USERNAME_DAGSHUB",
    repo_name="NAMA_REPO_DAGSHUB",
    mlflow=True
)

df = pd.read_csv("preprocessed_data.csv")
X = df.drop(columns=["C6H6(GT)"])
y = df["C6H6(GT)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("AirQuality_Advanced_Tuning")

params = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 15, 25],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=params,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
best_params = search.best_params_

y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        input_example=X_test.iloc[:5]
    )

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual C6H6")
    plt.ylabel("Predicted C6H6")
    plt.title("Actual vs Predicted Benzene Concentration")
    plt.tight_layout()

    plot_name = "prediction_scatter.png"
    plt.savefig(plot_name)
    mlflow.log_artifact(plot_name)

print("Best Parameters:", best_params)
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2 Score: {r2}")
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_csv("winequality_red_preprocessing.csv")

    X = df.drop("quality_label", axis=1)
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }

    base_model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="rf_tuning"):
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("best_n_estimators", best_model.n_estimators)
        mlflow.log_param("best_max_depth", best_model.max_depth)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("cv_best_score", grid.best_score_)

        mlflow.sklearn.log_model(best_model, "model")

if __name__ == "__main__":
    main()